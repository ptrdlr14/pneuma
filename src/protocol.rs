//! 线协议 v1 的寻址、信封与描述符契约。
//! 对齐 lerobot `remote_inference/protocol.py` 与 `inference/contracts.py`，
//! 消息类型为 snake_case 的线格式字符串，与 Python StrEnum 逐字一致。

use rmpv::Value;
use serde::{Deserialize, Serialize};

pub const PROTOCOL_VERSION: u32 = 1;

/// 观测/动作体内必须原样回带的关联身份键（对齐 Python IDENTITY_KEYS）。
pub const IDENTITY_KEYS: [&str; 5] = [
    "artifact_identity",
    "observation_id",
    "capture_time",
    "task",
    "task_version",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageType {
    Describe,
    Descriptor,
    Open,
    Accepted,
    Control,
    Ack,
    Observation,
    Action,
    LanguageRequest,
    LanguageResult,
    Error,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    Unsupported,
    Malformed,
    Busy,
    Stale,
    Incompatible,
    Execution,
    Timeout,
    Protocol,
}

impl ErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::Unsupported => "unsupported",
            ErrorCode::Malformed => "malformed",
            ErrorCode::Busy => "busy",
            ErrorCode::Stale => "stale",
            ErrorCode::Incompatible => "incompatible",
            ErrorCode::Execution => "execution",
            ErrorCode::Timeout => "timeout",
            ErrorCode::Protocol => "protocol",
        }
    }
}

impl std::fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{code}: {message}")]
pub struct ProtocolError {
    pub code: ErrorCode,
    pub message: String,
}

impl ProtocolError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

/// Zenoh key 的单段约束：`[A-Za-z0-9][A-Za-z0-9_.-]{0,127}`，越界段一律拒绝，
/// 防止构造出跨 deployment 的注入 key。
fn validate_segment(value: &str, label: &str) -> Result<(), ProtocolError> {
    let bytes = value.as_bytes();
    let ok = !bytes.is_empty()
        && bytes.len() <= 128
        && bytes[0].is_ascii_alphanumeric()
        && bytes
            .iter()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-'));
    if !ok {
        return Err(ProtocolError::new(
            ErrorCode::Malformed,
            format!("Invalid {label}: expected a single safe key segment"),
        ));
    }
    Ok(())
}

pub fn deployment_prefix(deployment: &str) -> Result<String, ProtocolError> {
    validate_segment(deployment, "deployment")?;
    Ok(format!(
        "lerobot/inference/v{PROTOCOL_VERSION}/deployments/{deployment}"
    ))
}

pub fn instance_prefix(deployment: &str, instance: &str) -> Result<String, ProtocolError> {
    validate_segment(instance, "instance")?;
    Ok(format!(
        "{}/instances/{instance}",
        deployment_prefix(deployment)?
    ))
}

pub fn session_prefix(
    deployment: &str,
    instance: &str,
    session: &str,
) -> Result<String, ProtocolError> {
    validate_segment(session, "session")?;
    Ok(format!(
        "{}/sessions/{session}",
        instance_prefix(deployment, instance)?
    ))
}

/// 这些类型必须携带完整 instance/session/request 关联身份。
fn requires_session_ids(kind: MessageType) -> bool {
    matches!(
        kind,
        MessageType::Observation
            | MessageType::Action
            | MessageType::LanguageRequest
            | MessageType::LanguageResult
            | MessageType::Control
            | MessageType::Ack
    )
}

/// 每条数据消息的关联信封；body 保持未解码的 msgpack 值，按消息类型再解释。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Envelope {
    pub version: u32,
    pub message_type: MessageType,
    pub instance_id: String,
    pub session_id: String,
    pub generation: i64,
    pub request_id: String,
    pub body: Value,
}

impl Envelope {
    /// 先于大 body 校验关联字段与版本，与 Python `__post_init__` 同序。
    pub fn validate(&self) -> Result<(), ProtocolError> {
        if self.version != PROTOCOL_VERSION {
            return Err(ProtocolError::new(
                ErrorCode::Protocol,
                format!("Unsupported protocol version: {}", self.version),
            ));
        }
        // Python 界为 0 <= generation < 2**63，i64 上界天然满足，只需查下界。
        if self.generation < 0 {
            return Err(ProtocolError::new(
                ErrorCode::Malformed,
                "Invalid execution generation",
            ));
        }
        for (label, value) in [
            ("instance_id", &self.instance_id),
            ("session_id", &self.session_id),
            ("request_id", &self.request_id),
        ] {
            if !value.is_empty() {
                validate_segment(value, label)?;
            }
        }
        if requires_session_ids(self.message_type)
            && (self.instance_id.is_empty()
                || self.session_id.is_empty()
                || self.request_id.is_empty())
        {
            return Err(ProtocolError::new(
                ErrorCode::Malformed,
                "Session messages require instance, session and request IDs",
            ));
        }
        if !self.body.is_map() {
            return Err(ProtocolError::new(
                ErrorCode::Malformed,
                "Message body must be a string-keyed map",
            ));
        }
        Ok(())
    }

    /// 回复保留请求的完整关联身份。
    pub fn reply(&self, message_type: MessageType, body: Value) -> Envelope {
        Envelope {
            version: PROTOCOL_VERSION,
            message_type,
            instance_id: self.instance_id.clone(),
            session_id: self.session_id.clone(),
            generation: self.generation,
            request_id: self.request_id.clone(),
            body,
        }
    }

    pub fn error(&self, code: ErrorCode, message: &str) -> Envelope {
        // 诊断文本截断到 2048 字节，details 属于后续对齐项，此处省略。
        let text: String = message.chars().take(2048).collect();
        self.reply(
            MessageType::Error,
            map([
                ("code", Value::String(code.as_str().into())),
                ("message", Value::String(text.into())),
            ]),
        )
    }

    /// 解码失败时没有可信关联身份，只能回空身份信封；Python 侧会 drop，
    /// 这里选择带 malformed 的显式错误以便对端日志定位。
    pub fn bare_error(code: ErrorCode, message: &str) -> Envelope {
        Envelope {
            version: PROTOCOL_VERSION,
            message_type: MessageType::Error,
            instance_id: String::new(),
            session_id: String::new(),
            generation: 0,
            request_id: String::new(),
            body: map([
                ("code", Value::String(code.as_str().into())),
                ("message", Value::String(message.into())),
            ]),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FeatureSpec {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: String,
    pub kind: String,
    pub names: Vec<String>,
    pub semantics: String,
}

impl Default for FeatureSpec {
    fn default() -> Self {
        Self {
            name: String::new(),
            shape: Vec::new(),
            dtype: String::new(),
            kind: "tensor".to_string(),
            names: Vec::new(),
            semantics: String::new(),
        }
    }
}

/// 准入前通告的不可变服务合约，字段与 Python `PolicyCapabilities` 的 asdict 对齐。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Capabilities {
    pub modes: Vec<String>,
    pub prediction_steps: u64,
    pub execution_steps: u64,
    pub action_interval: f64,
    pub features: Vec<FeatureSpec>,
    pub action_feature: FeatureSpec,
    pub language: bool,
    pub training_max_delay: u64,
    pub rtc_horizon: u64,
    pub current_observation_only: bool,
    pub retains_session_state: bool,
    pub action_representation: String,
    /// model_action_dim 与 action 维一致或缺省时省略（与 Python 弹出逻辑一致）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_action_dim: Option<u64>,
}

/// 在 MessagePack map 中查找第一个匹配的字符串键。
pub(crate) fn body_get<'a>(body: &'a Value, key: &str) -> Option<&'a Value> {
    body.as_map()?
        .iter()
        .find_map(|(k, v)| k.as_str().filter(|s| *s == key).map(|_| v))
}

/// 用键值对快速构造 msgpack map。
pub fn map<const N: usize>(entries: [(&str, Value); N]) -> Value {
    Value::Map(
        entries
            .into_iter()
            .map(|(k, v)| (Value::String(k.into()), v))
            .collect(),
    )
}
