//! PolicyServer：Zenoh IO 泵 + 单会话准入状态机。
//! 对齐 lerobot `remote_inference/server.py` 的通道布局：
//!   <deployment>/describe        queryable  DESCRIBE → DESCRIPTOR
//!   <instance>/open              queryable  OPEN     → ACCEPTED（session_id 由服务端生成）
//!   <instance>/alive             liveliness token（服务端在场）
//!   <session>/control            queryable  CONTROL  → ACK
//!   <session>/obs                subscriber OBSERVATION → 发布 ACTION 到 <session>/act
//!   <session>/alive              客户端在场令牌（骨架未接，缺席宽限为 TODO）
//!
//! 骨架范围：plain chunk（chunk_merge=append）。aligned/blending、RTC、language、
//! 缺席宽限与 generation 细粒度规则均未实现，入口处按协议返回显式错误。
//!
//! 并发模型：所有 zenoh 回调经 mpsc 汇入单任务泵，会话状态因此无需加锁，
//! 准入/关闭与推理发布天然串行。

use std::io::Cursor;
use std::sync::Arc;

use rmpv::Value;
use serde::Serialize;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
use zenoh::Session;
use zenoh::query::Query;
use zenoh::sample::Sample;

// callback 形态的声明句柄：接收端类型为 ()，Drop 即撤销声明。
type QueryableHandle = zenoh::query::Queryable<()>;
type SubscriberHandle = zenoh::pubsub::Subscriber<()>;

use crate::codec::{self, Limits};
use crate::config::ServerConfig;
use crate::policy::PolicyRunner;
use crate::protocol::*;

const LANGUAGE_DEADLINE_S: f64 = 60.0;
const MAX_INPUT_CHARS: u64 = 4096;
const MAX_OUTPUT_CHARS: u64 = 8192;

enum Ingress {
    Describe(Query),
    Open(Query),
    Control(Query),
    Observation(Vec<u8>),
    Language(Vec<u8>),
}

struct ActiveSession {
    id: String,
    generation: i64,
    /// 保留通道句柄；会话关闭时随 Drop 撤销 zenoh 声明。
    _obs: SubscriberHandle,
    _control: QueryableHandle,
    /// 骨架禁用 language，但客户端准入要求该订阅在场，收到请求回 UNSUPPORTED。
    _language: SubscriberHandle,
}

pub struct Server {
    cfg: ServerConfig,
    caps: Capabilities,
    /// caps 的 msgpack 形态；描述符每次回带，启动时算一次。
    caps_value: Value,
    policy: Arc<dyn PolicyRunner>,
    instance_id: String,
    limits: Limits,
}

/// zenoh 的错误类型是 `Box<dyn Error + Send + Sync>`，不经 anyhow 的 blanket From；
/// 所有 zenoh 调用点统一走这一条转换路径。
fn zresult<T>(r: Result<T, Box<dyn std::error::Error + Send + Sync>>) -> anyhow::Result<T> {
    r.map_err(anyhow::Error::from_boxed)
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
        Value::F32(f) => Some(*f as f64),
        Value::F64(f) => Some(*f),
        Value::Integer(i) => i.as_i64().map(|x| x as f64),
        _ => None,
    }
}

fn body_array<'a>(body: &'a Value, key: &str) -> Option<Vec<&'a Value>> {
    body_get(body, key)?
        .as_array()
        .map(|items| items.iter().collect())
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
                crate::config::Policy::Echo => "pneuma-echo-v0".into(),
                crate::config::Policy::Act => {
                    format!("pneuma-act:{}", self.cfg.model.repo_or_path.display())
                }
            })
    }

    pub fn new(cfg: ServerConfig) -> anyhow::Result<Self> {
        cfg.validate()?;
        let (policy, caps) = crate::policy::build(&cfg)?;
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
        let zenoh = zresult(zenoh::open(zconfig).await).context("opening zenoh session")?;
        let instance_key = instance_prefix(&deployment, &self.instance_id)?;
        let describe_key = format!("{}/describe", deployment_prefix(&deployment)?);
        let (tx, rx) = unbounded_channel::<Ingress>();

        // 实例在场令牌：客户端据此筛选存活的服务端启动代。
        let _token = zresult(
            zenoh
                .liveliness()
                .declare_token(format!("{instance_key}/alive"))
                .await,
        )?;

        let _describe = zresult(
            zenoh
                .declare_queryable(describe_key)
                .callback({
                    let tx = tx.clone();
                    move |query: Query| {
                        let _ = tx.send(Ingress::Describe(query));
                    }
                })
                .await,
        )?;
        let _open = zresult(
            zenoh
                .declare_queryable(format!("{instance_key}/open"))
                .callback({
                    let tx = tx.clone();
                    move |query: Query| {
                        let _ = tx.send(Ingress::Open(query));
                    }
                })
                .await,
        )?;

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
            Some(path) => zresult(zenoh::Config::from_file(path))
                .with_context(|| format!("reading zenoh.config_file {}", path.display()))?,
            None => zenoh::Config::default(),
        };
        // 官方客户端/服务端都只用显式 endpoints，不走 multicast/gossip。
        zresult(config.insert_json5("scouting/multicast/enabled", "false"))?;
        zresult(config.insert_json5("scouting/gossip/enabled", "false"))?;
        zresult(config.insert_json5("mode", &serde_json::to_string(&self.cfg.zenoh.mode)?))?;
        zresult(config.insert_json5(
            "connect/endpoints",
            &serde_json::to_string(&self.cfg.zenoh.connect_endpoints)?,
        ))?;
        zresult(config.insert_json5(
            "listen/endpoints",
            &serde_json::to_string(&self.cfg.zenoh.listen_endpoints)?,
        ))?;
        Ok(config)
    }

    async fn pump(
        self,
        zenoh: Session,
        mut rx: UnboundedReceiver<Ingress>,
        tx: UnboundedSender<Ingress>,
    ) {
        let mut session: Option<ActiveSession> = None;
        while let Some(ingress) = rx.recv().await {
            match ingress {
                Ingress::Describe(query) => {
                    let mut reply = self.control_reply(&query, MessageType::Descriptor, |me| {
                        me.descriptor_body(None)
                    });
                    // 客户端断言 descriptor.instance_id == 信封 instance_id；
                    // DESCRIBE 请求允许空身份，因此回复必须盖上服务端身份。
                    reply.instance_id = self.instance_id.clone();
                    self.reply(&query, &reply).await;
                }
                Ingress::Open(query) => {
                    session = self.handle_open(&zenoh, &query, session, &tx).await;
                }
                Ingress::Control(query) => {
                    self.handle_control(&query, &mut session).await;
                }
                Ingress::Observation(payload) => {
                    self.handle_observation(&zenoh, &mut session, &payload)
                        .await;
                }
                Ingress::Language(payload) => {
                    self.handle_language(&zenoh, &session, &payload).await;
                }
            }
        }
    }

    // ---- descriptor -------------------------------------------------------

    /// 描述符字段照抄 Python `_descriptor_locked`；骨架把会话诊断填成静态快照。
    fn descriptor_body(&self, owner: Option<&str>) -> Value {
        let busy = owner.is_some();
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
                        match owner {
                            Some(id) => Value::String(id.into()),
                            None => Value::Nil,
                        },
                    ),
                    ("client_present", Value::Nil),
                    ("absence_grace_remaining_s", Value::Nil),
                    ("cleanup_pending", Value::Boolean(false)),
                    ("cleanup_reason", Value::Nil),
                    ("inference_pending", Value::Boolean(false)),
                    // TODO: 接入客户端 liveliness 后填真实的 absence_grace 系列值。
                    (
                        "admission_blocker",
                        if busy {
                            Value::String("active_session".into())
                        } else {
                            Value::Nil
                        },
                    ),
                ]),
            ),
            (
                "session_status",
                Value::String(if busy { "active" } else { "idle" }.into()),
            ),
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

    // ---- admission --------------------------------------------------------

    /// OPEN 校验对齐 `_validate_open`；骨架只放行 append 合并，aligned 显式拒绝。
    /// 成功时先装通道再回 ACCEPTED，客户端收到回复即可发观测。
    async fn handle_open(
        &self,
        zenoh: &Session,
        query: &Query,
        session: Option<ActiveSession>,
        tx: &UnboundedSender<Ingress>,
    ) -> Option<ActiveSession> {
        if session.is_some() {
            // 独占准入：忙时按 Python 的 details.admission_blocker 约定回复。
            let envelope = self.decode_query(query);
            let busy = match envelope {
                Ok(envelope) if envelope.message_type == MessageType::Open => {
                    let mut busy = envelope.error(ErrorCode::Busy, "deployment is busy");
                    busy.body = map([
                        ("code", Value::String("busy".into())),
                        (
                            "message",
                            Value::String("an active session owns this deployment".into()),
                        ),
                        (
                            "details",
                            map([("admission_blocker", Value::String("active_session".into()))]),
                        ),
                    ]);
                    busy
                }
                _ => Envelope::bare_error(ErrorCode::Busy, "deployment is busy"),
            };
            self.reply(query, &busy).await;
            return session;
        }
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
        let mut accepted_body = self.descriptor_body(Some(&session_id));
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
        tx: &UnboundedSender<Ingress>,
    ) -> Result<ActiveSession, ProtocolError> {
        let obs = zenoh
            .declare_subscriber(format!("{key}/obs"))
            .callback({
                let tx = tx.clone();
                move |sample: Sample| {
                    let _ = tx.send(Ingress::Observation(
                        sample.payload().to_bytes().into_owned(),
                    ));
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
                    let _ = tx.send(Ingress::Control(query));
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
                    let _ = tx.send(Ingress::Language(sample.payload().to_bytes().into_owned()));
                }
            })
            .await
            .map_err(|e| {
                ProtocolError::new(
                    ErrorCode::Execution,
                    format!("declare language subscriber: {e}"),
                )
            })?;
        tracing::debug!("channels declared for session {session_id}");
        Ok(ActiveSession {
            id: session_id.to_string(),
            generation: 0,
            _obs: obs,
            _control: control,
            _language: language,
        })
    }

    fn validate_open(&self, body: &Value) -> Result<(), ProtocolError> {
        // required_capabilities 必须与 chunk_settings 推导出的集合完全一致。
        let required: Vec<&str> = body_array(body, "required_capabilities")
            .map(|items| items.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
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
        let features: Vec<FeatureSpec> = body_array(body, "features")
            .map(|items| items.iter().filter_map(|v| from_value(v)).collect())
            .unwrap_or_default();
        let action: Option<FeatureSpec> = body_get(body, "action_feature").and_then(from_value);
        if features != self.caps.features || action.as_ref() != Some(&self.caps.action_feature) {
            return Err(ProtocolError::new(
                ErrorCode::Incompatible,
                "Ordered feature schema or semantics differ",
            ));
        }
        if !matches!(body_str(body, "encoding"), Some("raw") | Some("jpeg")) {
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
    async fn handle_control(&self, query: &Query, session: &mut Option<ActiveSession>) {
        let envelope = match self.decode_query(query) {
            Ok(envelope) => envelope,
            Err(error) => {
                self.reply(query, &Envelope::bare_error(error.code, &error.message))
                    .await;
                return;
            }
        };
        let Some(active) = session.as_mut() else {
            self.reply(query, &envelope.error(ErrorCode::Stale, "Session closed"))
                .await;
            return;
        };
        if envelope.session_id != active.id {
            self.reply(
                query,
                &envelope.error(ErrorCode::Stale, "Session ownership changed"),
            )
            .await;
            return;
        }
        let operation = body_str(&envelope.body, "operation").unwrap_or("");
        let request_generation = envelope.generation;
        // close 之外，过旧的 generation 一律 STALE。
        if operation != "close" && request_generation < active.generation {
            self.reply(
                query,
                &envelope.error(ErrorCode::Stale, "Control generation became obsolete"),
            )
            .await;
            return;
        }
        let applied = match operation {
            "status" if request_generation != active.generation => {
                self.reply(
                    query,
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
                    query,
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
        self.reply(query, &ack).await;
        if operation == "close" {
            tracing::info!("session {} closed by client", active.id);
            *session = None;
        }
    }

    // ---- inference --------------------------------------------------------

    /// 观测 → 推理 → 把 ACTION（或错误）发布到 <session>/act。
    /// Python 里校验失败也走 /act 发布错误信封，这里保持一致。
    async fn handle_observation(
        &self,
        zenoh: &Session,
        session: &mut Option<ActiveSession>,
        payload: &[u8],
    ) {
        let envelope = match codec::decode_envelope(payload, &self.limits) {
            Ok(envelope) => envelope,
            Err(error) => {
                // Python 对无法解码的信封直接丢弃，不回包。
                tracing::warn!("rejected malformed inference envelope: {error}");
                return;
            }
        };
        if envelope.message_type != MessageType::Observation {
            return;
        }
        let Some(active) = session.as_ref() else {
            return;
        };
        let stale = envelope.instance_id != self.instance_id
            || envelope.session_id != active.id
            || envelope.generation < active.generation;
        if stale {
            self.publish_act(
                zenoh,
                active,
                &envelope.error(ErrorCode::Stale, "Session or generation became obsolete"),
            )
            .await;
            return;
        }
        // 观测特征键必须与通告合约完全一致。
        let supplied_keys: Vec<&str> = body_get(&envelope.body, "features")
            .and_then(|v| v.as_map())
            .map(|m| m.iter().filter_map(|(k, _)| k.as_str()).collect())
            .unwrap_or_default();
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
            return;
        }
        // 逐特征 shape/dtype/浮点有限性校验，与 Python `_validate_data` 同位；
        // 上面的键合约已保证 features 存在且为 map，直接借用免得深拷贝图像负载。
        let Some(features) = body_get(&envelope.body, "features") else {
            return;
        };
        if let Err(error) = codec::validate_features(features, &self.caps.features) {
            self.publish_act(zenoh, active, &envelope.error(error.code, &error.message))
                .await;
            return;
        }
        let infer_started = std::time::Instant::now();
        let chunk = match self.policy.predict(&self.caps, features) {
            Ok(chunk) => chunk,
            Err(error) => {
                self.publish_act(zenoh, active, &envelope.error(error.code, &error.message))
                    .await;
                return;
            }
        };
        let dim = self.caps.action_feature.shape[0];
        let steps = chunk.execution_steps;
        // canonical 与 model 同一表示（echo 桩）：model_actions 与 canonical 同值。
        let mut entries: Vec<(Value, Value)> = codec::identity_subset(&envelope.body)
            .into_iter()
            .map(|(k, v)| (Value::String(k.into()), v))
            .collect();
        entries.push((
            Value::String("canonical_actions".into()),
            codec::tensor_record("float32", &[steps, dim], chunk.canonical_actions.clone()),
        ));
        entries.push((
            Value::String("model_actions".into()),
            codec::tensor_record("float32", &[steps, dim], chunk.canonical_actions),
        ));
        entries.push((Value::String("execution_steps".into()), Value::from(steps)));
        // worker = 本次前向实际耗时；单请求内联处理，无排队 → queue=0。
        let worker_s = infer_started.elapsed().as_secs_f64();
        tracing::info!(
            session = active.id,
            steps,
            worker_ms = worker_s * 1000.0,
            "chunk produced"
        );
        entries.push((
            Value::String("server_durations".into()),
            map([
                ("queue", Value::from(0.0f64)),
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
        session: &Option<ActiveSession>,
        payload: &[u8],
    ) {
        let Ok(envelope) = codec::decode_envelope(payload, &self.limits) else {
            return; // 无法解码的信封不回包（与 Python 一致）。
        };
        let Some(active) = session else { return };
        if envelope.session_id != active.id {
            return;
        }
        let key = session_prefix(&self.cfg.deployment, &self.instance_id, &active.id)
            .expect("session segments validated at admission");
        let reply = envelope.error(ErrorCode::Unsupported, "language generation is disabled");
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
            .map(|p| p.to_bytes().into_owned())
            .unwrap_or_default();
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
            Ok(request) => request.reply(kind, body(self)),
            Err(_) => Envelope::bare_error(ErrorCode::Malformed, "Malformed control envelope"),
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
