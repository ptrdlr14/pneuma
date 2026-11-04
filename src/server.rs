//! PolicyServer：Zenoh IO 泵 + 单会话准入状态机。
//! 对齐 lerobot `remote_inference/server.py` 的通道布局：
//!   `<deployment>/describe`        queryable  DESCRIBE → DESCRIPTOR
//!   `<instance>/open`              queryable  OPEN → ACCEPTED（`session_id` 由服务端生成）
//!   `<instance>/alive`             liveliness token（服务端在场）
//!   `<session>/control`            queryable  CONTROL → ACK
//!   `<session>/obs`                subscriber OBSERVATION → 发布 ACTION 到 `<session>/act`
//!   `<session>/alive`              客户端在场令牌，history 订阅；缺席宽限见 `expire_session`
//!
//! 支持范围：plain chunk（`chunk_merge=append`）。aligned/blending、RTC、language 尚未实现；
//! 观测和 status 要求 generation 精确匹配。
//!
//! 并发模型：有界回调入口汇入单任务泵；泵独占会话和发布，
//! 至多一个 blocking worker 执行同步策略，控制无需等待推理。
//! 清理遵循 LeRobot：先标记 closing，再有在途推理时等它结束，之后才释放所有权。

use std::io::Cursor;
use std::sync::Arc;

use rmpv::Value;
use serde::Serialize;
use std::time::{Duration, Instant};

use tokio::sync::{
    OwnedSemaphorePermit, Semaphore,
    mpsc::{Receiver, Sender, channel},
    watch,
};
use tokio::task::JoinHandle;
use zenoh::Session;
use zenoh::query::Query;
use zenoh::sample::{Sample, SampleKind};

// callback 形态的声明句柄：接收端类型为 ()，Drop 即撤销声明。
type QueryableHandle = zenoh::query::Queryable<()>;
type SubscriberHandle = zenoh::pubsub::Subscriber<()>;

use crate::codec::{self, Limits};
use crate::config::ServerConfig;
use crate::contract::FeatureSpec;
use crate::inference::PolicyRunner;
use crate::protocol::{
    Capabilities, Envelope, ErrorCode, MessageType, ProtocolError, body_get, deployment_prefix,
    instance_prefix, map, session_prefix,
};

const LANGUAGE_DEADLINE_S: f64 = 60.0;
const MAX_INPUT_CHARS: u64 = 4096;
const MAX_OUTPUT_CHARS: u64 = 8192;
/// 在场事件只更新 watch；泵按此周期读取最新值并检查缺席宽限。
const PRESENCE_POLL: Duration = Duration::from_millis(50);

/// 有界回调积压；观测和 language 另用名额约束大 payload。
const INGRESS_CAPACITY: usize = 32;

enum Ingress {
    Describe(Query),
    Open(Query),
    Control(Query),
    Observation {
        payload: Vec<u8>,
        arrived: Instant,
        permit: OwnedSemaphorePermit,
        overloaded: bool,
    },
    Language(Vec<u8>, OwnedSemaphorePermit),
}

#[derive(Clone)]
struct IngressSender {
    tx: Sender<Ingress>,
    observation: Arc<Semaphore>,
    overload: Arc<Semaphore>,
    language: Arc<Semaphore>,
    max_payload_bytes: usize,
}

impl IngressSender {
    fn channel() -> (Self, Receiver<Ingress>) {
        let (tx, rx) = channel(INGRESS_CAPACITY);
        (
            Self {
                tx,
                observation: Arc::new(Semaphore::new(1)),
                overload: Arc::new(Semaphore::new(1)),
                language: Arc::new(Semaphore::new(1)),
                max_payload_bytes: Limits::default().max_payload_bytes,
            },
            rx,
        )
    }

    fn send(&self, ingress: Ingress) {
        let _ = self.tx.try_send(ingress);
    }

    fn observation(&self, sample: &Sample) {
        let arrived = Instant::now();
        if sample.payload().len() > self.max_payload_bytes {
            return;
        }
        let (permit, overloaded) = match self.observation.clone().try_acquire_owned() {
            Ok(permit) => (permit, false),
            Err(_) => match self.overload.clone().try_acquire_owned() {
                Ok(permit) => (permit, true),
                Err(_) => return,
            },
        };
        self.send(Ingress::Observation {
            payload: sample.payload().to_bytes().into_owned(),
            arrived,
            permit,
            overloaded,
        });
    }
}

struct Completion {
    envelope: Envelope,
    result: Result<crate::inference::ActionChunk, ProtocolError>,
    queue_s: f64,
    worker_s: f64,
    _permit: OwnedSemaphorePermit,
}

struct ActiveSession {
    id: String,
    generation: i64,
    /// 最近一次在场事件；`None` 表示尚未收到。
    present: Option<bool>,
    /// 首帧 Put 之后才区分「客户端离开」和「初始在场超时」。
    presence_established: bool,
    absent_since: Option<Instant>,
    closing: bool,
    cleanup_reason: Option<&'static str>,
    /// 在途推理期间收到的 close 查询；推理完成后再回 ACK 并释放。
    pending_close: Option<(Query, Envelope)>,
    /// liveliness 回调写入，泵周期读取；只保留最新值。
    presence: watch::Receiver<Option<bool>>,
    /// 保留通道句柄；会话关闭时随 Drop 撤销 zenoh 声明。
    _obs: SubscriberHandle,
    _control: QueryableHandle,
    /// 骨架禁用 language，但客户端准入要求该订阅在场，收到请求回 UNSUPPORTED。
    _language: SubscriberHandle,
    _alive: SubscriberHandle,
}

impl ActiveSession {
    /// 把 liveliness 的最新状态合并进会话；恢复在场会取消宽限。
    fn sync_presence(&mut self, now: Instant) {
        if !self.presence.has_changed().unwrap_or(false) {
            return;
        }
        let Some(present) = *self.presence.borrow_and_update() else {
            return;
        };
        if present {
            if !self.presence_established {
                tracing::info!(session = self.id, "initial client presence established");
            } else if self.present == Some(false) {
                tracing::info!(session = self.id, "client presence restored");
            }
            self.present = Some(true);
            self.presence_established = true;
            self.absent_since = None;
        } else {
            self.present = Some(false);
            if self.absent_since.is_none() {
                self.absent_since = Some(now);
                tracing::info!(session = self.id, "client presence lost");
            }
        }
    }
}

pub struct Server {
    cfg: ServerConfig,
    caps: Capabilities,
    /// caps 的 msgpack 形态；描述符每次回带，启动时算一次。
    caps_value: Value,
    policy: Arc<PolicyRunner>,
    instance_id: String,
    limits: Limits,
}

fn to_value<T: Serialize>(v: &T) -> Value {
    let bytes = rmp_serde::to_vec_named(v).expect("descriptor is msgpack-serializable");
    rmpv::decode::read_value(&mut Cursor::new(bytes)).expect("msgpack round-trip")
}

fn body_str<'a>(body: &'a Value, key: &str) -> Option<&'a str> {
    body_get(body, key).and_then(Value::as_str)
}

fn body_f64(body: &Value, key: &str) -> Option<f64> {
    match body_get(body, key)? {
        // msgpack float32 解成 F32，与 F64 同类接受，否则合法窄浮点被当类型错误。
        Value::F32(f) => Some(f64::from(*f)),
        Value::F64(f) => Some(*f),
        Value::Integer(i) => i.as_i64().map(|x| x as f64),
        _ => None,
    }
}

fn body_array<'a>(body: &'a Value, key: &str) -> Result<&'a [Value], ProtocolError> {
    match body_get(body, key) {
        None => Ok(&[]),
        Some(Value::Array(items)) => Ok(items),
        Some(_) => Err(ProtocolError::new(
            ErrorCode::Malformed,
            format!("{key} must be an array"),
        )),
    }
}

/// msgpack 值 → Rust 类型经 rmp-serde 往返；body 子结构与通告合约比对用。
fn from_value<T: serde::de::DeserializeOwned>(value: &Value) -> Option<T> {
    let bytes = rmp_serde::to_vec_named(value).ok()?;
    rmp_serde::from_slice(&bytes).ok()
}

impl Server {
    fn artifact_identity(&self) -> String {
        self.cfg
            .artifact_identity
            .clone()
            .unwrap_or_else(|| match self.cfg.model.policy {
                crate::policy::PolicyKind::Zero => "pneuma-zero-v0".into(),
                crate::policy::PolicyKind::Act => {
                    format!("pneuma-act:{}", self.cfg.model.repo_or_path.display())
                }
                crate::policy::PolicyKind::Smolvla => {
                    format!("pneuma-smolvla:{}", self.cfg.model.repo_or_path.display())
                }
                crate::policy::PolicyKind::Pi05 => {
                    format!("pneuma-pi05:{}", self.cfg.model.repo_or_path.display())
                }
                crate::policy::PolicyKind::Cosmos3Edge => {
                    format!(
                        "pneuma-cosmos3-edge:{}",
                        self.cfg.model.repo_or_path.display()
                    )
                }
            })
    }

    pub fn new(cfg: ServerConfig) -> anyhow::Result<Self> {
        cfg.validate()?;
        let (policy, caps) = crate::deployment::build(&cfg)?;
        Ok(Self {
            caps_value: to_value(&caps),
            caps,
            policy,
            instance_id: uuid::Uuid::new_v4().simple().to_string(),
            limits: Limits::default(),
            cfg,
        })
    }

    pub async fn run(self) -> anyhow::Result<()> {
        use anyhow::Context;
        let deployment = self.cfg.deployment.clone();
        let zconfig = self.zenoh_config()?;
        let zenoh = zenoh::open(zconfig)
            .await
            .map_err(anyhow::Error::from_boxed)
            .context("opening zenoh session")?;
        let instance_key = instance_prefix(&deployment, &self.instance_id)?;
        let describe_key = format!("{}/describe", deployment_prefix(&deployment)?);
        let (tx, rx) = IngressSender::channel();

        // 实例在场令牌：客户端据此筛选存活的服务端启动代。
        let _token = zenoh
            .liveliness()
            .declare_token(format!("{instance_key}/alive"))
            .await
            .map_err(anyhow::Error::from_boxed)?;

        let _describe = zenoh
            .declare_queryable(describe_key)
            .callback({
                let tx = tx.clone();
                move |query: Query| {
                    tx.send(Ingress::Describe(query));
                }
            })
            .await
            .map_err(anyhow::Error::from_boxed)?;
        let _open = zenoh
            .declare_queryable(format!("{instance_key}/open"))
            .callback({
                let tx = tx.clone();
                move |query: Query| {
                    tx.send(Ingress::Open(query));
                }
            })
            .await
            .map_err(anyhow::Error::from_boxed)?;

        tracing::info!(
            policy = ?self.cfg.model.policy,
            "policy server ready: deployment={} instance={} mode={}; awaiting one client",
            deployment,
            self.instance_id,
            self.cfg.zenoh.mode
        );
        // queryable 句柄留在本作用域存活；rx 只有随它们一起 Drop 才会结束。
        self.pump(zenoh, rx, tx).await;
        Ok(())
    }

    fn zenoh_config(&self) -> anyhow::Result<zenoh::Config> {
        use anyhow::Context;
        // JSON5 底座保留 TLS/证书/ACL；连接拓扑由 TOML 显式覆盖。
        let mut config = match &self.cfg.zenoh.config_file {
            Some(path) => zenoh::Config::from_file(path)
                .map_err(anyhow::Error::from_boxed)
                .with_context(|| format!("reading zenoh.config_file {}", path.display()))?,
            None => zenoh::Config::default(),
        };
        // 官方客户端/服务端都只用显式 endpoints，不走 multicast/gossip。
        config
            .insert_json5("scouting/multicast/enabled", "false")
            .map_err(anyhow::Error::from_boxed)?;
        config
            .insert_json5("scouting/gossip/enabled", "false")
            .map_err(anyhow::Error::from_boxed)?;
        config
            .insert_json5("mode", &serde_json::to_string(&self.cfg.zenoh.mode)?)
            .map_err(anyhow::Error::from_boxed)?;
        config
            .insert_json5(
                "connect/endpoints",
                &serde_json::to_string(&self.cfg.zenoh.connect_endpoints)?,
            )
            .map_err(anyhow::Error::from_boxed)?;
        config
            .insert_json5(
                "listen/endpoints",
                &serde_json::to_string(&self.cfg.zenoh.listen_endpoints)?,
            )
            .map_err(anyhow::Error::from_boxed)?;
        Ok(config)
    }

    async fn pump(self, zenoh: Session, mut rx: Receiver<Ingress>, tx: IngressSender) {
        let mut session: Option<ActiveSession> = None;
        let mut worker: Option<JoinHandle<Completion>> = None;
        let mut ticker = tokio::time::interval(PRESENCE_POLL);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            let ingress = tokio::select! {
                biased;
                // 先处理入口再收完成结果：在途推理期间收到的 close 必须先入账，
                // 否则完成分支会释放会话，使 close 查询拿到 STALE。
                ingress = rx.recv() => match ingress {
                    Some(ingress) => ingress,
                    None => break,
                },
                result = async { worker.as_mut().expect("worker present").await }, if worker.is_some() => {
                    worker = None;
                    match result {
                        Ok(completion) => self.complete_observation(&zenoh, session.as_ref(), completion).await,
                        Err(error) => tracing::error!(%error, "inference worker failed"),
                    }
                    if session.as_ref().is_some_and(|active| active.closing) {
                        self.release_session(&mut session).await;
                    }
                    continue;
                }
                _ = ticker.tick() => {
                    // 没有在途推理时宽限到期即可释放；否则继续持有到完成分支。
                    if self.expire_session(&mut session, worker.is_some()) {
                        self.release_session(&mut session).await;
                    }
                    continue;
                }
            };
            match ingress {
                Ingress::Describe(query) => {
                    let mut reply = self.control_reply(&query, MessageType::Descriptor, |me| {
                        me.descriptor_body(session.as_ref(), worker.is_some())
                    });
                    // 客户端断言 descriptor.instance_id == 信封 instance_id；
                    // DESCRIBE 请求允许空身份，因此回复必须盖上服务端身份。
                    reply.instance_id = self.instance_id.clone();
                    self.reply(&query, &reply).await;
                }
                Ingress::Open(query) => {
                    session = self
                        .handle_open(&zenoh, &query, session, &tx, worker.is_some())
                        .await;
                }
                Ingress::Control(query) => {
                    self.handle_control(query, &mut session, worker.is_some())
                        .await;
                    if worker.is_none() && session.as_ref().is_some_and(|active| active.closing) {
                        self.release_session(&mut session).await;
                    }
                }
                Ingress::Observation {
                    payload,
                    arrived,
                    permit,
                    overloaded,
                } => {
                    if let Some(task) = self
                        .handle_observation(
                            &zenoh,
                            session.as_ref(),
                            &payload,
                            arrived,
                            permit,
                            overloaded || worker.is_some(),
                        )
                        .await
                    {
                        worker = Some(task);
                    }
                }
                Ingress::Language(payload, _permit) => {
                    self.handle_language(&zenoh, session.as_ref(), &payload)
                        .await;
                }
            }
        }
    }

    /// 在场宽限到期后标记 closing；返回 true 表示当前没有在途推理，可以立即释放。
    fn expire_session(&self, session: &mut Option<ActiveSession>, worker_busy: bool) -> bool {
        let Some(active) = session.as_mut() else {
            return false;
        };
        active.sync_presence(Instant::now());
        if active.closing {
            return false;
        }
        let Some(absent_since) = active.absent_since else {
            return false;
        };
        if absent_since.elapsed() < Duration::from_secs_f64(self.cfg.execution.idle_timeout_s) {
            return false;
        }
        let reason = if active.presence_established {
            "client_absence"
        } else {
            "initial_presence_timeout"
        };
        active.closing = true;
        active.cleanup_reason = Some(reason);
        tracing::info!(
            session = active.id,
            reason,
            inference_pending = worker_busy,
            "session cleanup queued"
        );
        !worker_busy
    }

    /// 撤销会话所有权；通道句柄和未完成的 close 查询随 Drop 结束。
    async fn release_session(&self, session: &mut Option<ActiveSession>) {
        let Some(active) = session.take() else {
            return;
        };
        if let Some((query, ack)) = active.pending_close.as_ref() {
            self.reply(query, ack).await;
        }
        tracing::info!(
            session = active.id,
            reason = active.cleanup_reason.unwrap_or("client_close"),
            "session ownership released"
        );
    }

    // ---- descriptor -------------------------------------------------------

    /// 描述符字段对齐 Python `_descriptor_locked`，会话诊断取泵持有的当前会话。
    fn descriptor_body(&self, session: Option<&ActiveSession>, inference_pending: bool) -> Value {
        let busy = session.is_some();
        let absent_since = session.and_then(|active| active.absent_since);
        let grace_remaining = absent_since.map(|since| {
            (self.cfg.execution.idle_timeout_s - since.elapsed().as_secs_f64()).max(0.0)
        });
        let session_status = match session {
            None => "idle",
            Some(active) if active.closing => "closing",
            Some(_) if inference_pending => "busy",
            Some(_) => "active",
        };
        map([
            (
                "deployment",
                Value::String(self.cfg.deployment.clone().into()),
            ),
            (
                "instance_id",
                Value::String(self.instance_id.clone().into()),
            ),
            (
                "artifact_identity",
                Value::String(self.artifact_identity().into()),
            ),
            (
                "software",
                map([(
                    "lerobot_version",
                    Value::String("pneuma-0.1.0 (protocol v1)".into()),
                )]),
            ),
            (
                "semantics",
                Value::String(self.cfg.semantics.clone().into()),
            ),
            ("capabilities", self.caps_value.clone()),
            // 骨架未实现 aligned cursor 语义，故不宣告 chunk_alignment_v2；
            // 客户端 auto→aligned 会因此提前收到明确的 UNSUPPORTED。
            ("execution_contracts", Value::Array(Vec::new())),
            ("ready", Value::Boolean(true)),
            ("available", Value::Boolean(!busy)),
            ("failure", Value::Nil),
            (
                "session",
                map([
                    (
                        "owner",
                        match session {
                            Some(active) => Value::String(active.id.clone().into()),
                            None => Value::Nil,
                        },
                    ),
                    (
                        "client_present",
                        match session.and_then(|active| active.present) {
                            Some(present) => Value::Boolean(present),
                            None => Value::Nil,
                        },
                    ),
                    (
                        "absence_grace_remaining_s",
                        match grace_remaining {
                            Some(remaining) => Value::from(remaining),
                            None => Value::Nil,
                        },
                    ),
                    (
                        "cleanup_pending",
                        Value::Boolean(session.is_some_and(|active| active.closing)),
                    ),
                    (
                        "cleanup_reason",
                        match session.and_then(|active| active.cleanup_reason) {
                            Some(reason) => Value::String(reason.into()),
                            None => Value::Nil,
                        },
                    ),
                    ("inference_pending", Value::Boolean(inference_pending)),
                    (
                        "admission_blocker",
                        match self.admission_blocker(session, inference_pending) {
                            Some(blocker) => Value::String(blocker.into()),
                            None => Value::Nil,
                        },
                    ),
                ]),
            ),
            ("session_status", Value::String(session_status.into())),
            (
                "limits",
                map([
                    (
                        "action_deadline_s",
                        Value::from(self.cfg.execution.action_deadline_s),
                    ),
                    ("language_deadline_s", Value::from(LANGUAGE_DEADLINE_S)),
                    (
                        "idle_timeout_s",
                        Value::from(self.cfg.execution.idle_timeout_s),
                    ),
                    ("max_input_chars", Value::from(MAX_INPUT_CHARS)),
                    ("max_output_chars", Value::from(MAX_OUTPUT_CHARS)),
                    ("codec", self.limits.descriptor_value()),
                    ("inflight_inference", Value::from(1u64)),
                ]),
            ),
        ])
    }

    /// 与 Python `_admission_blocker_locked` 同序：清理优先，其次在场宽限。
    fn admission_blocker(
        &self,
        session: Option<&ActiveSession>,
        inference_pending: bool,
    ) -> Option<&'static str> {
        let active = session?;
        if active.closing {
            return Some(if inference_pending {
                "unfinished_inference"
            } else {
                "worker_cleanup_pending"
            });
        }
        if active.absent_since.is_some() {
            return Some(if active.presence_established {
                "absence_grace"
            } else {
                "awaiting_initial_presence"
            });
        }
        Some("active_session")
    }

    // ---- admission --------------------------------------------------------

    /// OPEN 校验对齐 `_validate_open`；骨架只放行 append 合并，aligned 显式拒绝。
    /// 成功时先装通道再回 ACCEPTED，客户端收到回复即可发观测。
    async fn handle_open(
        &self,
        zenoh: &Session,
        query: &Query,
        session: Option<ActiveSession>,
        tx: &IngressSender,
        worker_busy: bool,
    ) -> Option<ActiveSession> {
        let envelope = match self.decode_query(query) {
            Ok(envelope) if envelope.message_type == MessageType::Open => envelope,
            Ok(envelope) => {
                self.reply(
                    query,
                    &envelope.error(ErrorCode::Malformed, "Wrong control surface"),
                )
                .await;
                return session;
            }
            Err(error) => {
                self.reply(query, &Envelope::bare_error(error.code, &error.message))
                    .await;
                return session;
            }
        };
        if envelope.instance_id != self.instance_id {
            self.reply(
                query,
                &envelope.error(ErrorCode::Stale, "Server instance does not match"),
            )
            .await;
            return session;
        }
        if envelope.request_id.is_empty() {
            self.reply(
                query,
                &envelope.error(ErrorCode::Malformed, "Open operation ID is required"),
            )
            .await;
            return session;
        }
        if session.is_some() {
            // 独占准入：忙时按 Python 的 details.admission_blocker 约定回复。
            let blocker = self
                .admission_blocker(session.as_ref(), worker_busy)
                .unwrap_or("active_session");
            let mut busy = envelope.error(
                ErrorCode::Busy,
                "an active session owns this deployment; the previous owner must finish cleanup",
            );
            if let Value::Map(entries) = &mut busy.body {
                let mut details = map([("admission_blocker", Value::String(blocker.into()))]);
                if let Some(remaining) = session
                    .as_ref()
                    .and_then(|active| active.absent_since)
                    .map(|since| {
                        (self.cfg.execution.idle_timeout_s - since.elapsed().as_secs_f64()).max(0.0)
                    })
                    && let Value::Map(fields) = &mut details
                {
                    fields.push((
                        Value::String("absence_grace_remaining_s".into()),
                        Value::from(remaining),
                    ));
                }
                entries.push(("details".into(), details));
            }
            self.reply(query, &busy).await;
            return session;
        }
        if let Err(error) = self.validate_open(&envelope.body) {
            self.reply(query, &envelope.error(error.code, &error.message))
                .await;
            return session;
        }

        let session_id = uuid::Uuid::new_v4().simple().to_string();
        let key = session_prefix(&self.cfg.deployment, &self.instance_id, &session_id)
            .expect("uuid is a valid segment");
        let active = match self.declare_channels(zenoh, &key, &session_id, tx).await {
            Ok(active) => active,
            Err(error) => {
                self.reply(query, &envelope.error(error.code, &error.message))
                    .await;
                return session;
            }
        };
        // ACCEPTED = 完整描述符（含新 owner）+ 协商结果。
        let mut accepted_body = self.descriptor_body(Some(&active), false);
        if let Value::Map(entries) = &mut accepted_body {
            entries.push((Value::String("mode".into()), Value::String("chunk".into())));
            entries.push((
                Value::String("chunk_settings".into()),
                map([("chunk_merge", Value::String("append".into()))]),
            ));
        }
        let mut accepted = envelope.reply(MessageType::Accepted, accepted_body);
        // reply() 只保留请求关联；ACCEPTED 必须携带服务端新分配的 session_id。
        accepted.session_id = session_id.clone();
        self.reply(query, &accepted).await;
        tracing::info!(
            instance_id = self.instance_id,
            session_id,
            "session admitted"
        );
        Some(active)
    }

    /// 为已准入会话声明 obs/control 通道，回调复用泵的发送端。
    /// zenoh 声明失败回错误给客户端（Execution），不能让单会话的建立失败带走进程。
    async fn declare_channels(
        &self,
        zenoh: &Session,
        key: &str,
        session_id: &str,
        tx: &IngressSender,
    ) -> Result<ActiveSession, ProtocolError> {
        let obs = zenoh
            .declare_subscriber(format!("{key}/obs"))
            .callback({
                let tx = tx.clone();
                move |sample: Sample| {
                    tx.observation(&sample);
                }
            })
            .await
            .map_err(|e| {
                ProtocolError::new(ErrorCode::Execution, format!("declare obs subscriber: {e}"))
            })?;
        let control = zenoh
            .declare_queryable(format!("{key}/control"))
            .callback({
                let tx = tx.clone();
                move |query: Query| {
                    tx.send(Ingress::Control(query));
                }
            })
            .await
            .map_err(|e| {
                ProtocolError::new(
                    ErrorCode::Execution,
                    format!("declare control queryable: {e}"),
                )
            })?;
        let language = zenoh
            .declare_subscriber(format!("{key}/language/request"))
            .callback({
                let tx = tx.clone();
                move |sample| {
                    if sample.payload().len() <= tx.max_payload_bytes
                        && let Ok(permit) = tx.language.clone().try_acquire_owned()
                    {
                        tx.send(Ingress::Language(
                            sample.payload().to_bytes().into_owned(),
                            permit,
                        ));
                    }
                }
            })
            .await
            .map_err(|e| {
                ProtocolError::new(
                    ErrorCode::Execution,
                    format!("declare language subscriber: {e}"),
                )
            })?;
        // history(true) 让客户端在 OPEN 之前声明的令牌也能立即送达；
        // 回调只更新 watch，避免有界入口丢弃在场变化。
        let (presence_tx, presence) = watch::channel(None);
        let alive = zenoh
            .liveliness()
            .declare_subscriber(format!("{key}/alive"))
            .history(true)
            .callback(move |sample: Sample| {
                let _ = presence_tx.send(Some(sample.kind() == SampleKind::Put));
            })
            .await
            .map_err(|e| {
                ProtocolError::new(
                    ErrorCode::Execution,
                    format!("declare presence subscriber: {e}"),
                )
            })?;
        tracing::debug!("channels declared for session {session_id}");
        Ok(ActiveSession {
            id: session_id.to_string(),
            generation: 0,
            present: None,
            presence_established: false,
            // 初始在场也受宽限约束：客户端须在 idle_timeout_s 内声明令牌。
            absent_since: Some(Instant::now()),
            closing: false,
            cleanup_reason: None,
            pending_close: None,
            presence,
            _obs: obs,
            _control: control,
            _language: language,
            _alive: alive,
        })
    }

    fn validate_open(&self, body: &Value) -> Result<(), ProtocolError> {
        // required_capabilities 必须与 chunk_settings 推导出的集合完全一致。
        let required = body_array(body, "required_capabilities")?
            .iter()
            .enumerate()
            .map(|(index, value)| {
                value.as_str().ok_or_else(|| {
                    ProtocolError::new(
                        ErrorCode::Malformed,
                        format!("required_capabilities[{index}] must be a string"),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let merge = body_get(body, "chunk_settings")
            .and_then(|v| body_str(v, "chunk_merge"))
            .unwrap_or("append");
        if !matches!(merge, "append" | "aligned") {
            return Err(ProtocolError::new(
                ErrorCode::Incompatible,
                "chunk_merge must be append or aligned",
            ));
        }
        if merge == "aligned" {
            return Err(ProtocolError::new(
                ErrorCode::Incompatible,
                "aligned chunk merge is not implemented by this server skeleton",
            ));
        }
        if !required.is_empty() {
            return Err(ProtocolError::new(
                ErrorCode::Incompatible,
                "Required capabilities differ from chunk_settings",
            ));
        }
        if body_str(body, "mode") != Some("chunk") {
            return Err(ProtocolError::new(
                ErrorCode::Unsupported,
                "Requested execution mode is unavailable",
            ));
        }
        if body_str(body, "semantics").unwrap_or("") != self.cfg.semantics {
            return Err(ProtocolError::new(
                ErrorCode::Incompatible,
                "Robot/action semantic profile differs",
            ));
        }
        if let Some(expected) = body_str(body, "expected_artifact")
            && expected != self.artifact_identity()
        {
            return Err(ProtocolError::new(
                ErrorCode::Incompatible,
                "Deployed artifact identity differs",
            ));
        }
        let interval = body_f64(body, "action_interval").ok_or_else(|| {
            ProtocolError::new(ErrorCode::Malformed, "action_interval is required")
        })?;
        if (interval - self.caps.action_interval).abs() > self.caps.action_interval * 1e-6 {
            return Err(ProtocolError::new(
                ErrorCode::Incompatible,
                "Policy action interval differs",
            ));
        }
        let features = body_array(body, "features")?
            .iter()
            .enumerate()
            .map(|(index, value)| {
                from_value::<FeatureSpec>(value).ok_or_else(|| {
                    ProtocolError::new(
                        ErrorCode::Malformed,
                        format!("features[{index}] must be a FeatureSpec"),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let action: Option<FeatureSpec> = body_get(body, "action_feature").and_then(from_value);
        if features != self.caps.features || action.as_ref() != Some(&self.caps.action_feature) {
            return Err(ProtocolError::new(
                ErrorCode::Incompatible,
                "Ordered feature schema or semantics differ",
            ));
        }
        if body_str(body, "encoding") != Some("raw") {
            return Err(ProtocolError::new(
                ErrorCode::Unsupported,
                "Unsupported image encoding",
            ));
        }
        Ok(())
    }

    // ---- control ----------------------------------------------------------

    /// 控制面对齐 Python `SessionWorker` 的 generation 规则：
    /// reset 把 generation 置为请求值（单调不回退），status 要求精确匹配，
    /// close 始终应用；ACK 必须回带 `applied_generation`（客户端逐字断言）。
    /// 在途推理期间收到的 close 延后到 worker 结束再 ACK，所有权同刻释放。
    async fn handle_control(
        &self,
        query: Query,
        session: &mut Option<ActiveSession>,
        worker_busy: bool,
    ) {
        let envelope = match self.decode_query(&query) {
            Ok(envelope) => envelope,
            Err(error) => {
                self.reply(&query, &Envelope::bare_error(error.code, &error.message))
                    .await;
                return;
            }
        };
        if envelope.message_type != MessageType::Control {
            self.reply(
                &query,
                &envelope.error(ErrorCode::Malformed, "Wrong control surface"),
            )
            .await;
            return;
        }
        if envelope.instance_id != self.instance_id {
            self.reply(
                &query,
                &envelope.error(ErrorCode::Stale, "Server instance does not match"),
            )
            .await;
            return;
        }
        let Some(active) = session.as_mut() else {
            self.reply(&query, &envelope.error(ErrorCode::Stale, "Session closed"))
                .await;
            return;
        };
        if envelope.session_id != active.id {
            self.reply(
                &query,
                &envelope.error(ErrorCode::Stale, "Session ownership changed"),
            )
            .await;
            return;
        }
        let operation = body_str(&envelope.body, "operation").unwrap_or("");
        if active.closing && operation != "close" {
            self.reply(
                &query,
                &envelope.error(ErrorCode::Stale, "Session is closing"),
            )
            .await;
            return;
        }
        let request_generation = envelope.generation;
        // close 之外，过旧的 generation 一律 STALE。
        if operation != "close" && request_generation < active.generation {
            self.reply(
                &query,
                &envelope.error(ErrorCode::Stale, "Control generation became obsolete"),
            )
            .await;
            return;
        }
        let applied = match operation {
            "status" if request_generation != active.generation => {
                self.reply(
                    &query,
                    &envelope.error(ErrorCode::Stale, "Status generation does not match"),
                )
                .await;
                return;
            }
            "status" | "reset" => {
                active.generation = request_generation;
                request_generation
            }
            "close" => request_generation,
            other => {
                self.reply(
                    &query,
                    &envelope.error(
                        ErrorCode::Unsupported,
                        &format!("control operation {other:?} is not supported by the skeleton"),
                    ),
                )
                .await;
                return;
            }
        };
        let ack = envelope.reply(
            MessageType::Ack,
            map([
                ("operation", Value::String(operation.into())),
                ("applied_generation", Value::from(applied)),
            ]),
        );
        if operation == "close" {
            if !active.closing {
                active.closing = true;
                active.cleanup_reason = Some("client_close");
                tracing::info!(
                    session = active.id,
                    inference_pending = worker_busy,
                    "session close requested"
                );
            }
            // 忙时留给完成分支 ACK；泵在没有在途推理时立即释放所有权。
            if worker_busy && active.pending_close.is_none() {
                active.pending_close = Some((query, ack));
            } else {
                self.reply(&query, &ack).await;
            }
            return;
        }
        self.reply(&query, &ack).await;
    }

    // ---- inference --------------------------------------------------------

    /// 观测 → 推理 → 把 ACTION（或错误）发布到 <session>/act。
    /// Python 里校验失败也走 /act 发布错误信封，这里保持一致。
    async fn handle_observation(
        &self,
        zenoh: &Session,
        session: Option<&ActiveSession>,
        payload: &[u8],
        arrived: Instant,
        permit: OwnedSemaphorePermit,
        overloaded: bool,
    ) -> Option<JoinHandle<Completion>> {
        let envelope = match codec::decode_envelope(payload, &self.limits) {
            Ok(envelope) => envelope,
            Err(error) => {
                // Python 对无法解码的信封直接丢弃，不回包。
                tracing::warn!("rejected malformed inference envelope: {error}");
                return None;
            }
        };
        if envelope.message_type != MessageType::Observation {
            return None;
        }
        let active = session?;
        let stale = envelope.instance_id != self.instance_id
            || envelope.session_id != active.id
            || envelope.generation != active.generation;
        if stale {
            self.publish_act(
                zenoh,
                active,
                &envelope.error(ErrorCode::Stale, "Session or generation became obsolete"),
            )
            .await;
            return None;
        }
        if active.closing {
            self.publish_act(
                zenoh,
                active,
                &envelope.error(ErrorCode::Stale, "Session is not accepting inference"),
            )
            .await;
            return None;
        }
        // 观测特征键必须与通告合约完全一致。
        let supplied_keys: Vec<&str> = body_get(&envelope.body, "features")
            .and_then(|v| v.as_map())
            .map_or_else(Vec::new, |m| {
                m.iter().filter_map(|(k, _)| k.as_str()).collect()
            });
        let expected: Vec<&str> = self.caps.features.iter().map(|f| f.name.as_str()).collect();
        if supplied_keys.len() != expected.len()
            || !expected.iter().all(|name| supplied_keys.contains(name))
        {
            self.publish_act(
                zenoh,
                active,
                &envelope.error(ErrorCode::Incompatible, "Observation feature keys differ"),
            )
            .await;
            return None;
        }
        // 逐特征 shape/dtype/浮点有限性校验，与 Python `_validate_data` 同位；
        // 上面的键合约已保证 features 存在且为 map，直接借用免得深拷贝图像负载。
        let features = body_get(&envelope.body, "features")?;
        if let Err(error) = codec::validate_features(features, &self.caps.features) {
            self.publish_act(zenoh, active, &envelope.error(error.code, &error.message))
                .await;
            return None;
        }
        if overloaded {
            self.publish_act(
                zenoh,
                active,
                &envelope.error(ErrorCode::Busy, "Inference capacity is full"),
            )
            .await;
            return None;
        }
        let policy = self.policy.clone();
        Some(tokio::task::spawn_blocking(move || {
            let started = Instant::now();
            let result = codec::observation_batch(
                body_get(&envelope.body, "features").expect("validated features"),
                body_str(&envelope.body, "task"),
            )
            .and_then(|batch| {
                policy.predict(batch).map_err(|error| {
                    ProtocolError::new(ErrorCode::Execution, format!("policy inference: {error:#}"))
                })
            });
            let finished = Instant::now();
            Completion {
                envelope,
                result,
                queue_s: started.duration_since(arrived).as_secs_f64(),
                worker_s: finished.duration_since(started).as_secs_f64(),
                _permit: permit,
            }
        }))
    }

    async fn complete_observation(
        &self,
        zenoh: &Session,
        session: Option<&ActiveSession>,
        completion: Completion,
    ) {
        let Some(active) = session else {
            return;
        };
        // 清理中的会话不再发布结果；客户端已离开或正在关闭。
        if active.closing {
            return;
        }
        let Completion {
            envelope,
            result,
            queue_s,
            worker_s,
            _permit,
        } = completion;
        if envelope.session_id != active.id || envelope.generation != active.generation {
            return;
        }
        let chunk = match result {
            Ok(chunk) => chunk,
            Err(error) => {
                self.publish_act(zenoh, active, &envelope.error(error.code, &error.message))
                    .await;
                return;
            }
        };
        let dim = self.caps.action_feature.shape[0];
        let steps = chunk.steps as u64;
        let action_bytes = codec::f32_le(&chunk.values);
        // plain chunk 仅使用 canonical 表示；协议两字段沿用同值。
        let mut entries: Vec<(Value, Value)> = codec::identity_subset(&envelope.body)
            .into_iter()
            .map(|(k, v)| (Value::String(k.into()), v))
            .collect();
        entries.push((
            Value::String("canonical_actions".into()),
            codec::tensor_record("float32", &[steps, dim], action_bytes.clone()),
        ));
        entries.push((
            Value::String("model_actions".into()),
            codec::tensor_record("float32", &[steps, dim], action_bytes),
        ));
        entries.push((Value::String("execution_steps".into()), Value::from(steps)));
        tracing::info!(
            session = active.id,
            steps,
            worker_ms = worker_s * 1000.0,
            "chunk produced"
        );
        entries.push((
            Value::String("server_durations".into()),
            map([
                ("queue", Value::from(queue_s)),
                ("worker", Value::from(worker_s)),
            ]),
        ));
        let action = envelope.reply(MessageType::Action, Value::Map(entries));
        self.publish_act(zenoh, active, &action).await;
    }

    async fn publish_act(&self, zenoh: &Session, session: &ActiveSession, envelope: &Envelope) {
        let key = session_prefix(&self.cfg.deployment, &self.instance_id, &session.id)
            .expect("session segments validated at admission");
        match codec::encode_envelope(envelope, &self.limits) {
            Ok(payload) => {
                if let Err(error) = zenoh.put(format!("{key}/act"), payload).await {
                    tracing::warn!("failed to publish action reply: {error}");
                }
            }
            Err(error) => tracing::warn!("action reply could not be encoded: {error}"),
        }
    }

    /// 骨架禁用 language：任何请求按 Python 同位回 UNSUPPORTED 错误到 /language/result。
    async fn handle_language(
        &self,
        zenoh: &Session,
        session: Option<&ActiveSession>,
        payload: &[u8],
    ) {
        let Ok(envelope) = codec::decode_envelope(payload, &self.limits) else {
            return; // 无法解码的信封不回包（与 Python 一致）。
        };
        let Some(active) = session else { return };
        let reply = if envelope.message_type != MessageType::LanguageRequest {
            envelope.error(ErrorCode::Malformed, "Wrong language surface")
        } else if envelope.instance_id != self.instance_id
            || envelope.session_id != active.id
            || envelope.generation != active.generation
        {
            envelope.error(ErrorCode::Stale, "Session or generation became obsolete")
        } else if active.closing {
            envelope.error(ErrorCode::Stale, "Session is not accepting inference")
        } else {
            envelope.error(ErrorCode::Unsupported, "language generation is disabled")
        };
        let key = session_prefix(&self.cfg.deployment, &self.instance_id, &active.id)
            .expect("session segments validated at admission");
        if let Ok(bytes) = codec::encode_envelope(&reply, &self.limits)
            && let Err(error) = zenoh.put(format!("{key}/language/result"), bytes).await
        {
            tracing::warn!("failed to publish language error: {error}");
        }
    }

    // ---- plumbing ---------------------------------------------------------

    fn decode_query(&self, query: &Query) -> Result<Envelope, ProtocolError> {
        let payload = query
            .payload()
            .map_or_else(Vec::new, |p| p.to_bytes().into_owned());
        codec::decode_envelope(&payload, &self.limits)
    }

    /// 查询回复：DESCRIBE 允许空关联身份，协商期字段直接回带。
    fn control_reply(
        &self,
        query: &Query,
        kind: MessageType,
        body: impl Fn(&Self) -> Value,
    ) -> Envelope {
        match self.decode_query(query) {
            Ok(request) if request.message_type == MessageType::Describe => {
                request.reply(kind, body(self))
            }
            Ok(request) => request.error(ErrorCode::Malformed, "Wrong control surface"),
            Err(error) => Envelope::bare_error(error.code, &error.message),
        }
    }

    async fn reply(&self, query: &Query, envelope: &Envelope) {
        let key = query.key_expr().clone();
        match codec::encode_envelope(envelope, &self.limits) {
            Ok(payload) => {
                if let Err(error) = query.reply(key, payload).await {
                    tracing::warn!("reply failed: {error}");
                }
            }
            Err(error) => tracing::warn!("reply could not be encoded: {error}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::PROTOCOL_VERSION;
    use std::time::Duration;

    fn server() -> Server {
        let mut cfg: ServerConfig = toml::from_str(include_str!("../examples/so101.toml")).unwrap();
        cfg.features.truncate(1);
        Server::new(cfg).unwrap()
    }

    fn request(
        kind: MessageType,
        instance: &str,
        session: &str,
        generation: i64,
        body: Value,
    ) -> Envelope {
        Envelope {
            version: PROTOCOL_VERSION,
            message_type: kind,
            instance_id: instance.into(),
            session_id: session.into(),
            generation,
            request_id: "req1".into(),
            body,
        }
    }

    fn open_body(server: &Server) -> Value {
        map([
            ("required_capabilities", Value::Array(vec![])),
            ("mode", "chunk".into()),
            ("semantics", server.cfg.semantics.clone().into()),
            ("action_interval", server.caps.action_interval.into()),
            ("features", to_value(&server.caps.features)),
            ("action_feature", to_value(&server.caps.action_feature)),
            ("encoding", "raw".into()),
        ])
    }

    fn replace(body: &mut Value, key: &str, value: Value) {
        let Value::Map(entries) = body else {
            panic!("expected map")
        };
        let (_, old) = entries
            .iter_mut()
            .find(|(k, _)| k.as_str() == Some(key))
            .unwrap();
        *old = value;
    }

    fn assert_error(reply: &Envelope, code: ErrorCode) {
        assert_eq!(reply.message_type, MessageType::Error, "{reply:?}");
        assert_eq!(
            body_str(&reply.body, "code"),
            Some(code.as_str()),
            "{reply:?}"
        );
    }

    /// 轮询描述符直到条件成立；在场事件经 zenoh 异步送达。
    async fn wait_descriptor(wire: &Wire, predicate: impl Fn(&Value) -> bool) -> Value {
        for _ in 0..200 {
            let body = wire.describe().await.body;
            if predicate(&body) {
                return body;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("descriptor condition was not reached");
    }

    /// 脱离 `Wire` 借用发起查询，便于在等待 ACK 期间继续探测描述符。
    fn detached_query(zenoh: Session, key: String, envelope: Envelope) -> JoinHandle<Envelope> {
        let payload = codec::encode_envelope(&envelope, &Limits::default()).unwrap();
        tokio::spawn(async move {
            let replies = zenoh
                .get(key)
                .payload(payload)
                .timeout(Duration::from_secs(5))
                .await
                .unwrap();
            let reply = replies.recv_async().await.unwrap();
            codec::decode_envelope(
                &reply.result().as_ref().unwrap().payload().to_bytes(),
                &Limits::default(),
            )
            .unwrap()
        })
    }

    // 本地路由走真实编码和回调，无需端口、发现等待或外部进程。
    struct Wire {
        zenoh: Session,
        instance: String,
        describe_key: String,
        open_key: String,
        body: Value,
        ingress: IngressSender,
        _describe: QueryableHandle,
        _open: QueryableHandle,
        pump: tokio::task::JoinHandle<()>,
    }
    impl Drop for Wire {
        fn drop(&mut self) {
            self.pump.abort();
        }
    }
    impl Wire {
        async fn start() -> Self {
            Self::start_with(server()).await
        }
        async fn start_with(server: Server) -> Self {
            let mut cfg = server.zenoh_config().unwrap();
            cfg.insert_json5("listen/endpoints", "[]").unwrap();
            let zenoh = zenoh::open(cfg).await.unwrap();
            let instance = server.instance_id.clone();
            let describe_key = format!(
                "{}/describe",
                deployment_prefix(&server.cfg.deployment).unwrap()
            );
            let open_key = format!(
                "{}/open",
                instance_prefix(&server.cfg.deployment, &instance).unwrap()
            );
            let body = open_body(&server);
            let (tx, rx) = IngressSender::channel();
            let sender = tx.clone();
            let describe = zenoh
                .declare_queryable(describe_key.clone())
                .callback(move |q| {
                    sender.send(Ingress::Describe(q));
                })
                .await
                .unwrap();
            let sender = tx.clone();
            let open = zenoh
                .declare_queryable(open_key.clone())
                .callback(move |q| {
                    sender.send(Ingress::Open(q));
                })
                .await
                .unwrap();
            let pump = tokio::spawn(server.pump(zenoh.clone(), rx, tx.clone()));
            Self {
                zenoh,
                instance,
                describe_key,
                open_key,
                body,
                ingress: tx,
                _describe: describe,
                _open: open,
                pump,
            }
        }
        async fn query(&self, key: &str, envelope: &Envelope) -> Envelope {
            let replies = self
                .zenoh
                .get(key.to_owned())
                .payload(codec::encode_envelope(envelope, &Limits::default()).unwrap())
                .timeout(Duration::from_secs(2))
                .await
                .unwrap();
            let reply = tokio::time::timeout(Duration::from_secs(3), replies.recv_async())
                .await
                .unwrap()
                .unwrap();
            codec::decode_envelope(
                &reply.result().as_ref().unwrap().payload().to_bytes(),
                &Limits::default(),
            )
            .unwrap()
        }
        async fn describe(&self) -> Envelope {
            self.query(
                &self.describe_key,
                &request(MessageType::Describe, "", "", 0, map([])),
            )
            .await
        }
        async fn admit(&self) -> Envelope {
            let reply = self
                .query(
                    &self.open_key,
                    &request(MessageType::Open, &self.instance, "", 0, self.body.clone()),
                )
                .await;
            assert_eq!(reply.message_type, MessageType::Accepted, "{reply:?}");
            reply
        }
        fn key(&self, session: &str, suffix: &str) -> String {
            format!(
                "{}/{suffix}",
                session_prefix("manipulation", &self.instance, session).unwrap()
            )
        }
        async fn publish_reply(
            &self,
            session: &str,
            input: &str,
            output: &str,
            envelope: &Envelope,
        ) -> Envelope {
            let subscriber = self
                .zenoh
                .declare_subscriber(self.key(session, output))
                .await
                .unwrap();
            self.zenoh
                .put(
                    self.key(session, input),
                    codec::encode_envelope(envelope, &Limits::default()).unwrap(),
                )
                .await
                .unwrap();
            let sample = tokio::time::timeout(Duration::from_secs(2), subscriber.recv_async())
                .await
                .unwrap()
                .unwrap();
            codec::decode_envelope(&sample.payload().to_bytes(), &Limits::default()).unwrap()
        }
    }

    struct TaskPolicy;

    impl crate::policy::Policy for TaskPolicy {
        fn predict(
            &self,
            mut batch: crate::processor::Batch,
        ) -> anyhow::Result<crate::processor::Batch> {
            let value = match batch.tasks.first().map(String::as_str) {
                Some("pick red") => 1f32,
                Some("pick blue") => 2f32,
                _ => anyhow::bail!("task was not forwarded"),
            };
            batch.values.clear();
            batch.values.insert(
                "action".into(),
                crate::processor::Data::Float3(burn::prelude::Tensor::full(
                    [1, 50, 6],
                    value,
                    &burn::prelude::Device::flex(),
                )),
            );
            Ok(batch)
        }
    }

    fn with_policy(
        server: &Server,
        policy: impl crate::policy::Policy + 'static,
    ) -> Arc<PolicyRunner> {
        use crate::processor::{DType, Field, Layout, Pipeline, Schema};
        let steps = server.caps.prediction_steps as usize;
        let dim = server.caps.action_feature.shape[0] as usize;
        let output = Schema::from([(
            "action".into(),
            Field {
                shape: vec![Some(1), Some(steps), Some(dim)],
                dtype: DType::F32,
                layout: Layout::Flat,
                optional: false,
            },
        )]);
        Arc::new(
            PolicyRunner::new(
                Box::new(policy),
                (
                    Pipeline::new(server.policy.preprocessor.input_schema().clone(), vec![])
                        .unwrap(),
                    Pipeline::new(output, vec![]).unwrap(),
                ),
                burn::prelude::Device::flex(),
                steps,
                server.caps.execution_steps as usize,
                dim,
            )
            .unwrap(),
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn observation_task_conditions_each_prediction_and_is_returned_with_actions() {
        let mut server = server();
        server.policy = with_policy(&server, TaskPolicy);
        let wire = Wire::start_with(server).await;
        let accepted = wire.admit().await;
        for (task, expected) in [("pick red", 1f32), ("pick blue", 2f32)] {
            let envelope = request(
                MessageType::Observation,
                &wire.instance,
                &accepted.session_id,
                0,
                map([
                    (
                        "features",
                        map([(
                            "observation.state",
                            codec::tensor_record("float32", &[6], vec![0; 24]),
                        )]),
                    ),
                    ("task", task.into()),
                ]),
            );
            let reply = wire
                .publish_reply(&accepted.session_id, "obs", "act", &envelope)
                .await;
            assert_eq!(reply.message_type, MessageType::Action, "{reply:?}");
            assert_eq!(body_str(&reply.body, "task"), Some(task));
            let actions =
                codec::decode_record(body_get(&reply.body, "canonical_actions").unwrap()).unwrap();
            assert!(
                actions
                    .data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|bytes| f32::from_le_bytes(*bytes) == expected)
            );
        }
    }

    struct SlowPolicy {
        entered: tokio::sync::mpsc::UnboundedSender<()>,
        gate: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
        fails: bool,
    }

    struct Release(Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>);
    impl Drop for Release {
        fn drop(&mut self) {
            if let Ok(mut released) = self.0.0.lock() {
                *released = true;
            }
            // poisoned gate 会让 worker 失败；清理仍须唤醒，且不能二次 panic。
            self.0.1.notify_all();
        }
    }

    impl crate::policy::Policy for SlowPolicy {
        fn predict(
            &self,
            batch: crate::processor::Batch,
        ) -> anyhow::Result<crate::processor::Batch> {
            self.entered.send(()).unwrap();
            let released = self.gate.0.lock().expect("gate poisoned");
            drop(
                self.gate
                    .1
                    .wait_while(released, |released| !*released)
                    .expect("gate poisoned"),
            );
            if self.fails {
                anyhow::bail!("gated failure")
            } else {
                crate::policy::Policy::predict(
                    &crate::policy::zero::ZeroActionPolicy {
                        steps: 50,
                        dim: 6,
                        device: burn::prelude::Device::flex(),
                    },
                    batch,
                )
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn invalidated_workers_keep_capacity_and_drop_success_and_errors() {
        for operation in ["reset", "close"] {
            for fails in [false, true] {
                let mut server = server();
                let (entered, mut entries) = tokio::sync::mpsc::unbounded_channel();
                let gate = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
                let release = Release(gate.clone());
                server.policy = with_policy(
                    &server,
                    SlowPolicy {
                        entered,
                        gate,
                        fails,
                    },
                );
                let wire = Wire::start_with(server).await;
                let old = wire.admit().await.session_id;
                let old_actions = wire
                    .zenoh
                    .declare_subscriber(wire.key(&old, "act"))
                    .await
                    .unwrap();
                let mut observation = request(
                    MessageType::Observation,
                    &wire.instance,
                    &old,
                    0,
                    map([(
                        "features",
                        map([(
                            "observation.state",
                            codec::tensor_record("float32", &[6], vec![0; 24]),
                        )]),
                    )]),
                );
                wire.zenoh
                    .put(
                        wire.key(&old, "obs"),
                        codec::encode_envelope(&observation, &Limits::default()).unwrap(),
                    )
                    .await
                    .unwrap();
                tokio::time::timeout(Duration::from_secs(3), entries.recv())
                    .await
                    .unwrap()
                    .unwrap();
                let pending = wire.describe().await;
                assert_eq!(
                    body_get(
                        body_get(&pending.body, "session").unwrap(),
                        "inference_pending"
                    ),
                    Some(&true.into())
                );
                // reset 立即 ACK 并保留会话；close 推迟 ACK 到推理结束。
                let close = if operation == "close" {
                    Some(detached_query(
                        wire.zenoh.clone(),
                        wire.key(&old, "control"),
                        request(
                            MessageType::Control,
                            &wire.instance,
                            &old,
                            0,
                            map([("operation", "close".into())]),
                        ),
                    ))
                } else {
                    let ack = wire
                        .query(
                            &wire.key(&old, "control"),
                            &request(
                                MessageType::Control,
                                &wire.instance,
                                &old,
                                1,
                                map([("operation", "reset".into())]),
                            ),
                        )
                        .await;
                    assert_eq!(ack.message_type, MessageType::Ack);
                    None
                };
                observation.request_id = "next".into();
                if operation == "reset" {
                    observation.generation = 1;
                    // 会话保留，容量被在途推理占满：新观测 Busy。
                    let busy = wire.publish_reply(&old, "obs", "act", &observation).await;
                    assert_error(&busy, ErrorCode::Busy);
                    old_actions.recv_async().await.unwrap();
                } else {
                    let closing = wait_descriptor(&wire, |body| {
                        body_get(body, "session_status") == Some(&"closing".into())
                    })
                    .await;
                    assert!(
                        !close.as_ref().unwrap().is_finished(),
                        "close ACK was sent before inference"
                    );
                    let session = body_get(&closing, "session").unwrap();
                    assert_eq!(body_get(session, "cleanup_pending"), Some(&true.into()));
                    assert_eq!(
                        body_str(session, "admission_blocker"),
                        Some("unfinished_inference")
                    );
                    let busy = wire
                        .query(
                            &wire.open_key,
                            &request(MessageType::Open, &wire.instance, "", 0, wire.body.clone()),
                        )
                        .await;
                    assert_error(&busy, ErrorCode::Busy);
                    let stale = wire.publish_reply(&old, "obs", "act", &observation).await;
                    assert_error(&stale, ErrorCode::Stale);
                    // 该 STALE 错误也会投递到 old_actions；先取走再检查泄漏。
                    old_actions.recv_async().await.unwrap();
                }
                assert!(
                    entries.try_recv().is_err(),
                    "second predict entered before release"
                );
                assert_eq!(wire.ingress.observation.available_permits(), 0);
                drop(release);
                // 名额在 completion 发布/丢弃之后归还，作为处理完成屏障。
                let permit = tokio::time::timeout(
                    Duration::from_secs(3),
                    wire.ingress.observation.clone().acquire_owned(),
                )
                .await
                .unwrap()
                .unwrap();
                assert!(
                    old_actions.try_recv().unwrap().is_none(),
                    "obsolete result leaked: {operation}/{fails}"
                );
                drop(permit);
                let (current, generation) = match close {
                    Some(close) => {
                        let ack = close.await.unwrap();
                        assert_eq!(ack.message_type, MessageType::Ack);
                        (wire.admit().await.session_id, 0)
                    }
                    None => (old.clone(), 1),
                };
                observation.session_id = current.clone();
                observation.generation = generation;
                let result = wire
                    .publish_reply(&current, "obs", "act", &observation)
                    .await;
                if fails {
                    assert_error(&result, ErrorCode::Execution);
                } else {
                    assert_eq!(result.message_type, MessageType::Action);
                }
                assert_eq!(result.request_id, "next");
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn describe_tracks_active_owner() {
        let wire = Wire::start().await;
        assert_eq!(
            body_get(&wire.describe().await.body, "available"),
            Some(&true.into())
        );
        let accepted = wire.admit().await;
        let active = wire.describe().await;
        assert_eq!(active.instance_id, wire.instance);
        assert_eq!(body_get(&active.body, "available"), Some(&false.into()));
        assert_eq!(body_str(&active.body, "session_status"), Some("active"));
        let session = body_get(&active.body, "session").unwrap();
        assert_eq!(
            body_str(session, "owner"),
            Some(accepted.session_id.as_str())
        );
        // 客户端令牌尚未声明：准入被初始在场宽限阻塞，但不是清理。
        assert_eq!(body_get(session, "client_present"), Some(&Value::Nil));
        assert!(body_f64(session, "absence_grace_remaining_s").is_some());
        assert_eq!(body_get(session, "cleanup_pending"), Some(&false.into()));
        assert_eq!(
            body_str(session, "admission_blocker"),
            Some("awaiting_initial_presence")
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn client_absence_releases_session_after_grace() {
        let mut server = server();
        server.cfg.execution.idle_timeout_s = 0.2;
        let wire = Wire::start_with(server).await;
        let accepted = wire.admit().await;
        let token = wire
            .zenoh
            .liveliness()
            .declare_token(wire.key(&accepted.session_id, "alive"))
            .await
            .unwrap();
        let active = wait_descriptor(&wire, |body| {
            body_get(body_get(body, "session").unwrap(), "client_present") == Some(&true.into())
        })
        .await;
        assert_eq!(body_str(&active, "session_status"), Some("active"));
        assert_eq!(
            body_str(body_get(&active, "session").unwrap(), "admission_blocker"),
            Some("active_session")
        );

        drop(token);
        let idle = wait_descriptor(&wire, |body| {
            body_get(body, "session_status") == Some(&"idle".into())
        })
        .await;
        assert_eq!(body_get(&idle, "available"), Some(&true.into()));
        assert_eq!(
            body_get(body_get(&idle, "session").unwrap(), "owner"),
            Some(&Value::Nil)
        );
        // 宽限释放后所有权可被新会话取得。
        wire.admit().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn describe_checks_type_and_preserves_protocol_diagnostic() {
        let wire = Wire::start().await;
        assert_error(
            &wire
                .query(
                    &wire.describe_key,
                    &request(MessageType::Open, "", "", 0, map([])),
                )
                .await,
            ErrorCode::Malformed,
        );
        let mut wrong = request(MessageType::Describe, "", "", 0, map([]));
        wrong.version = 2;
        let reply = wire.query(&wire.describe_key, &wrong).await;
        assert_error(&reply, ErrorCode::Protocol);
        assert_eq!(
            body_str(&reply.body, "message"),
            Some("Unsupported protocol version: 2")
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn open_checks_instance_without_taking_ownership() {
        let wire = Wire::start().await;
        let wrong = request(MessageType::Open, "other", "", 0, wire.body.clone());
        let reply = wire.query(&wire.open_key, &wrong).await;
        assert_error(&reply, ErrorCode::Stale);
        assert_eq!(reply.request_id, wrong.request_id);
        assert_eq!(
            body_get(&wire.describe().await.body, "available"),
            Some(&true.into())
        );
        let mut features = match body_get(&wire.body, "features") {
            Some(Value::Array(items)) => items.clone(),
            _ => panic!("open body must carry features"),
        };
        features.push(true.into());
        for (field, value, code) in [
            ("encoding", "jpeg".into(), ErrorCode::Unsupported),
            (
                "required_capabilities",
                Value::Array(vec![42.into()]),
                ErrorCode::Malformed,
            ),
            ("required_capabilities", true.into(), ErrorCode::Malformed),
            ("features", true.into(), ErrorCode::Malformed),
            ("features", Value::Array(features), ErrorCode::Malformed),
        ] {
            let mut body = wire.body.clone();
            replace(&mut body, field, value);
            let invalid = request(MessageType::Open, &wire.instance, "", 0, body);
            assert_error(&wire.query(&wire.open_key, &invalid).await, code);
            assert_eq!(
                body_get(&wire.describe().await.body, "available"),
                Some(&true.into())
            );
        }
        let accepted = wire.admit().await;
        let busy = wire
            .query(
                &wire.open_key,
                &request(MessageType::Open, &wire.instance, "", 0, wire.body.clone()),
            )
            .await;
        assert_error(&busy, ErrorCode::Busy);
        assert_eq!(
            body_str(
                body_get(&busy.body, "details").unwrap(),
                "admission_blocker"
            ),
            Some("awaiting_initial_presence")
        );
        let descriptor = wire.describe().await;
        assert_eq!(
            body_str(body_get(&descriptor.body, "session").unwrap(), "owner"),
            Some(accepted.session_id.as_str())
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn control_guards_preserve_owner_and_close_generation_exception() {
        let wire = Wire::start().await;
        let accepted = wire.admit().await;
        let key = wire.key(&accepted.session_id, "control");
        for (kind, instance, session, code) in [
            (
                MessageType::Open,
                wire.instance.as_str(),
                accepted.session_id.as_str(),
                ErrorCode::Malformed,
            ),
            (
                MessageType::Control,
                "other",
                accepted.session_id.as_str(),
                ErrorCode::Stale,
            ),
            (
                MessageType::Control,
                wire.instance.as_str(),
                "other",
                ErrorCode::Stale,
            ),
        ] {
            assert_error(
                &wire
                    .query(
                        &key,
                        &request(
                            kind,
                            instance,
                            session,
                            5,
                            map([("operation", "reset".into())]),
                        ),
                    )
                    .await,
                code,
            );
        }
        for (operation, generation) in [("status", 0), ("reset", 3), ("close", 0)] {
            let reply = wire
                .query(
                    &key,
                    &request(
                        MessageType::Control,
                        &wire.instance,
                        &accepted.session_id,
                        generation,
                        map([("operation", operation.into())]),
                    ),
                )
                .await;
            assert_eq!(reply.message_type, MessageType::Ack, "{reply:?}");
            assert_eq!(
                body_get(&reply.body, "applied_generation"),
                Some(&generation.into())
            );
            assert_eq!(reply.request_id, "req1");
            assert_eq!(reply.session_id, accepted.session_id);
        }
        let idle = wire.describe().await;
        assert_eq!(body_str(&idle.body, "session_status"), Some("idle"));
        assert_eq!(
            body_get(body_get(&idle.body, "session").unwrap(), "owner"),
            Some(&Value::Nil)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn observation_requires_exact_generation() {
        let wire = Wire::start().await;
        let accepted = wire.admit().await;
        let body = map([(
            "features",
            map([(
                "observation.state",
                codec::tensor_record("float32", &[6], vec![0; 24]),
            )]),
        )]);
        for generation in [1, 0] {
            let envelope = request(
                MessageType::Observation,
                &wire.instance,
                &accepted.session_id,
                generation,
                body.clone(),
            );
            let reply = wire
                .publish_reply(&accepted.session_id, "obs", "act", &envelope)
                .await;
            if generation == 1 {
                assert_error(&reply, ErrorCode::Stale);
            } else {
                assert_eq!(reply.message_type, MessageType::Action);
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn language_checks_identity_before_unsupported() {
        let wire = Wire::start().await;
        let accepted = wire.admit().await;
        for (kind, instance, session, generation, code) in [
            (
                MessageType::Open,
                wire.instance.as_str(),
                accepted.session_id.as_str(),
                0,
                ErrorCode::Malformed,
            ),
            (
                MessageType::LanguageRequest,
                "other",
                accepted.session_id.as_str(),
                0,
                ErrorCode::Stale,
            ),
            (
                MessageType::LanguageRequest,
                wire.instance.as_str(),
                "other",
                0,
                ErrorCode::Stale,
            ),
            (
                MessageType::LanguageRequest,
                wire.instance.as_str(),
                accepted.session_id.as_str(),
                1,
                ErrorCode::Stale,
            ),
            (
                MessageType::LanguageRequest,
                wire.instance.as_str(),
                accepted.session_id.as_str(),
                0,
                ErrorCode::Unsupported,
            ),
        ] {
            let envelope = request(kind, instance, session, generation, map([]));
            let reply = wire
                .publish_reply(
                    &accepted.session_id,
                    "language/request",
                    "language/result",
                    &envelope,
                )
                .await;
            assert_error(&reply, code);
            assert_eq!(reply.request_id, envelope.request_id);
        }
    }
}
