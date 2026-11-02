//! 有界 `MessagePack` 编解码：信封收发与显式类型的 tensor/RGB 记录。
//! 对齐 lerobot `remote_inference/codec.py`：拒绝扩展类型、非有限浮点、
//! 重复 map 键，所有分配先过预算；tensor/RGB 记录当前均仅支持 raw。

use std::io::Cursor;

use rmpv::Value;

use crate::protocol::{Envelope, ErrorCode, IDENTITY_KEYS, ProtocolError, body_get, map};

pub const MARKER: &str = "__lerobot_type__";

/// 与 Python `CodecLimits` 同名同默认值的防御性上界。
pub struct Limits {
    pub max_payload_bytes: usize,
    pub max_string_bytes: usize,
    pub max_container_items: usize,
    pub max_nodes: usize,
    pub max_depth: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_payload_bytes: 32 * 1024 * 1024,
            max_string_bytes: 16 * 1024,
            max_container_items: 4096,
            max_nodes: 32768,
            max_depth: 16,
        }
    }
}

impl Limits {
    /// 描述符 limits.codec 字段直接用 Python 全量默认值，供客户端预判分配。
    pub fn descriptor_value(&self) -> Value {
        map([
            (
                "max_payload_bytes",
                Value::from(self.max_payload_bytes as u64),
            ),
            ("max_decoded_bytes", Value::from(64u64 * 1024 * 1024)),
            ("max_tensor_bytes", Value::from(32u64 * 1024 * 1024)),
            ("max_image_pixels", Value::from(4096u64 * 4096)),
            ("max_dimension", Value::from(16384u64)),
            ("max_rank", Value::from(8u64)),
            (
                "max_string_bytes",
                Value::from(self.max_string_bytes as u64),
            ),
            (
                "max_container_items",
                Value::from(self.max_container_items as u64),
            ),
            ("max_nodes", Value::from(self.max_nodes as u64)),
            ("max_depth", Value::from(self.max_depth as u64)),
        ])
    }
}

fn malformed<T>(msg: &str) -> Result<T, ProtocolError> {
    Err(ProtocolError::new(ErrorCode::Malformed, msg))
}

/// 解码后全树走查：Python 用 unpack 钩子边解边数，这里等价地先整读再走查，
/// 骨架阶段接受多一次解析的开销。
fn walk(
    value: &Value,
    depth: usize,
    limits: &Limits,
    nodes: &mut usize,
) -> Result<(), ProtocolError> {
    *nodes += 1;
    if depth > limits.max_depth || *nodes > limits.max_nodes {
        return malformed("Message nesting or node count exceeds limit");
    }
    match value {
        // msgpack float32 解成 Value::F32，与 F64 同闸门，否则 NaN/Inf 从窄浮点绕行。
        Value::F32(f) if !f.is_finite() => malformed("Non-finite scalar"),
        Value::F64(f) if !f.is_finite() => malformed("Non-finite scalar"),
        Value::String(s) => {
            // rmpv 的 String 只装 UTF-8 文本，二进制一律走 Binary 变体；
            // 字节数与 Python 的 len(value.encode('utf-8')) 语义一致。
            if s.as_bytes().len() > limits.max_string_bytes {
                return malformed("String exceeds byte limit");
            }
            Ok(())
        }
        Value::Binary(bin) => {
            if bin.len() > limits.max_payload_bytes {
                return malformed("Byte string exceeds limit");
            }
            Ok(())
        }
        Value::Array(items) => {
            if items.len() > limits.max_container_items {
                return malformed("Sequence exceeds item limit");
            }
            for item in items {
                walk(item, depth + 1, limits, nodes)?;
            }
            Ok(())
        }
        Value::Map(pairs) => {
            if pairs.len() > limits.max_container_items {
                return malformed("Invalid map size or reserved codec marker");
            }
            let mut seen: Vec<&Value> = Vec::with_capacity(pairs.len());
            for (key, item) in pairs {
                if !key.is_str() {
                    return malformed("Map keys must be strings");
                }
                if seen.contains(&key) {
                    return malformed("Invalid or duplicate map key");
                }
                seen.push(key);
                walk(item, depth + 1, limits, nodes)?;
            }
            Ok(())
        }
        // 扩展类型与非法标量（如 rmpv 的自定义类型）一律拒绝。
        Value::Ext(_, _) => malformed("MessagePack extensions are unsupported"),
        _ => Ok(()),
    }
}

/// 有界解码信封：尺寸 → 语法走查 → 结构反序列化 → 关联字段校验。
pub fn decode_envelope(payload: &[u8], limits: &Limits) -> Result<Envelope, ProtocolError> {
    if payload.len() > limits.max_payload_bytes {
        return malformed("Encoded message exceeds byte limit");
    }
    let raw = rmpv::decode::read_value(&mut Cursor::new(payload))
        .map_err(|_| ProtocolError::new(ErrorCode::Malformed, "Malformed MessagePack envelope"))?;
    let mut nodes = 0usize;
    walk(&raw, 0, limits, &mut nodes)?;
    let envelope: Envelope = rmp_serde::from_slice(payload)
        .map_err(|_| ProtocolError::new(ErrorCode::Malformed, "Incomplete message envelope"))?;
    envelope.validate()?;
    Ok(envelope)
}

pub fn encode_envelope(envelope: &Envelope, limits: &Limits) -> Result<Vec<u8>, ProtocolError> {
    let bytes = rmp_serde::to_vec_named(envelope)
        .map_err(|e| ProtocolError::new(ErrorCode::Execution, format!("encode failed: {e}")))?;
    if bytes.len() > limits.max_payload_bytes {
        return malformed("Encoded message exceeds byte limit");
    }
    Ok(bytes)
}

/// numpy dtype 名 → 字节宽；与 Python `_DTYPES` 表一致。
pub fn dtype_size(dtype: &str) -> Option<usize> {
    Some(match dtype {
        "bool" | "uint8" | "int8" => 1,
        "uint16" | "int16" | "float16" => 2,
        "uint32" | "int32" | "float32" => 4,
        "uint64" | "int64" | "float64" => 8,
        _ => return None,
    })
}

#[derive(Debug)]
pub struct TensorRecord<'a> {
    pub dtype: String,
    pub shape: Vec<u64>,
    pub data: &'a [u8],
    /// rgb 记录附带 `channel_order` 校验后的标记，骨架只支持 raw。
    pub rgb: bool,
}

/// 解出 tensor/RGB 记录的原始字节；形状与字节数必须严格吻合。
pub fn decode_record(value: &Value) -> Result<TensorRecord<'_>, ProtocolError> {
    let marker = body_get(value, MARKER)
        .and_then(Value::as_str)
        .unwrap_or("");
    let dtype = body_get(value, "dtype")
        .and_then(Value::as_str)
        .unwrap_or("");
    let dims = body_get(value, "shape")
        .and_then(Value::as_array)
        .ok_or_else(|| ProtocolError::new(ErrorCode::Malformed, "shape must be an array"))?;
    let shape = dims
        .iter()
        .enumerate()
        .map(|(index, dimension)| {
            dimension.as_u64().ok_or_else(|| {
                ProtocolError::new(
                    ErrorCode::Malformed,
                    format!("shape[{index}] must be a non-negative integer"),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let data = body_get(value, "data")
        .and_then(|d| d.as_slice())
        .unwrap_or(&[]);
    let item = dtype_size(dtype)
        .ok_or_else(|| ProtocolError::new(ErrorCode::Malformed, "Unsupported tensor dtype"))?;
    // 形状乘积必须走 checked：release 下 [2^32, 2^32] 会回绕成 0，空 payload 即过字节校验。
    let expected = shape
        .iter()
        .try_fold(1u64, |n, &d| n.checked_mul(d))
        .and_then(|n| n.checked_mul(item as u64))
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| ProtocolError::new(ErrorCode::Malformed, "Record shape overflows"))?;
    if marker == "tensor" {
        if body_get(value, "endianness").and_then(Value::as_str) != Some("little") {
            return malformed("Tensor must be little-endian");
        }
    } else if marker == "rgb" {
        if dtype != "uint8"
            || body_get(value, "channel_order").and_then(Value::as_str) != Some("RGB")
        {
            return malformed("Invalid RGB dtype/channel order");
        }
        if body_get(value, "encoding").and_then(Value::as_str) != Some("raw") {
            return malformed("Unsupported RGB encoding");
        }
    } else {
        return malformed("Unknown typed record");
    }
    if data.len() != expected {
        return malformed("Tensor payload size does not match dtype and shape");
    }
    Ok(TensorRecord {
        dtype: dtype.to_string(),
        shape,
        data,
        rgb: marker == "rgb",
    })
}

/// 观测特征逐条对照通告合约：typed-record 标记、shape、dtype、浮点有限性。
/// 对齐 Python `_validate_data`；f16 的有限性检查留待接入真实策略时补。
pub fn validate_features(
    features: &Value,
    specs: &[crate::contract::FeatureSpec],
) -> Result<(), ProtocolError> {
    features
        .as_map()
        .ok_or_else(|| ProtocolError::new(ErrorCode::Malformed, "features must be a map"))?;
    for spec in specs {
        let value = body_get(features, &spec.name).ok_or_else(|| {
            ProtocolError::new(ErrorCode::Incompatible, "Observation feature keys differ")
        })?;
        let record = decode_record(value)?;
        if record.rgb != (spec.kind == "rgb") {
            return malformed("feature kind and record marker differ");
        }
        if record.shape != spec.shape || record.dtype != spec.dtype {
            // 形状/类型不符是合约问题，不是编码损坏。
            return Err(ProtocolError::new(
                ErrorCode::Incompatible,
                "Observation feature shape or dtype differs",
            ));
        }
        if record.dtype == "float32" {
            for chunk in record.data.as_chunks::<4>().0 {
                if !f32::from_le_bytes(*chunk).is_finite() {
                    return malformed("Tensor contains non-finite values");
                }
            }
        }
    }
    Ok(())
}

pub fn tensor_record(dtype: &str, shape: &[u64], data: Vec<u8>) -> Value {
    map([
        (MARKER, Value::String("tensor".into())),
        ("dtype", Value::String(dtype.into())),
        (
            "shape",
            Value::Array(shape.iter().map(|d| Value::from(*d)).collect()),
        ),
        ("endianness", Value::String("little".into())),
        ("data", Value::Binary(data)),
    ])
}

/// f32 小端平铺，供 echo 桩构造 float32 动作块。
pub fn f32_le(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// 将已验证的在线记录转换为与协议无关的主机 batch。
pub fn observation_batch(
    features: &Value,
    task: Option<&str>,
) -> Result<crate::processor::Batch, ProtocolError> {
    use crate::processor::{Batch, Data};
    let mut batch = Batch {
        tasks: task.map(|s| vec![s.to_owned()]).unwrap_or_default(),
        ..Batch::default()
    };
    let entries = features
        .as_map()
        .ok_or_else(|| ProtocolError::new(ErrorCode::Malformed, "features must be a map"))?;
    for (name, value) in entries {
        let name = name.as_str().ok_or_else(|| {
            ProtocolError::new(ErrorCode::Malformed, "feature name must be a string")
        })?;
        let record = decode_record(value)?;
        let shape = record
            .shape
            .iter()
            .map(|&d| usize::try_from(d))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| {
                ProtocolError::new(ErrorCode::Malformed, "feature shape exceeds platform size")
            })?;
        let data = match record.dtype.as_str() {
            "uint8" => Data::U8 {
                values: record.data.to_vec(),
                shape,
            },
            "float32" => Data::F32 {
                values: record
                    .data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|bytes| f32::from_le_bytes(*bytes))
                    .collect(),
                shape,
            },
            other => {
                return Err(ProtocolError::new(
                    ErrorCode::Incompatible,
                    format!("pipeline input dtype {other} is unsupported"),
                ));
            }
        };
        if batch.values.insert(name.to_owned(), data).is_some() {
            return Err(ProtocolError::new(
                ErrorCode::Malformed,
                "duplicate observation feature",
            ));
        }
    }
    Ok(batch)
}

/// 从观测 body 原样摘出身份键回带进动作 body。
pub fn identity_subset(body: &Value) -> Vec<(String, Value)> {
    IDENTITY_KEYS
        .iter()
        .filter_map(|key| body_get(body, key).map(|v| (key.to_string(), v.clone())))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::FeatureSpec;
    use crate::protocol::{Envelope, MessageType};

    fn envelope(kind: MessageType, body: Value) -> Envelope {
        Envelope {
            version: crate::protocol::PROTOCOL_VERSION,
            message_type: kind,
            instance_id: "inst1".into(),
            session_id: "sess1".into(),
            generation: 0,
            request_id: "req1".into(),
            body,
        }
    }

    #[test]
    fn envelope_roundtrip_preserves_correlation_and_body() {
        let original = envelope(
            MessageType::Observation,
            map([
                ("task", Value::String("pick up the cup".into())),
                ("task_version", Value::from(3u64)),
                ("state", tensor_record("float32", &[6], vec![0u8; 24])),
            ]),
        );
        let bytes = encode_envelope(&original, &Limits::default()).unwrap();
        let decoded = decode_envelope(&bytes, &Limits::default()).unwrap();
        assert_eq!(decoded.message_type, MessageType::Observation);
        assert_eq!(decoded.request_id, "req1");
        assert_eq!(decoded.body.as_map().unwrap().len(), 3);
    }

    #[test]
    fn session_message_without_request_id_is_rejected() {
        let mut e = envelope(MessageType::Observation, map([]));
        e.request_id = String::new();
        let bytes = encode_envelope(&e, &Limits::default()).unwrap();
        let error = decode_envelope(&bytes, &Limits::default()).unwrap_err();
        assert_eq!(error.code, ErrorCode::Malformed);
    }

    #[test]
    fn non_finite_float_is_rejected() {
        let e = envelope(
            MessageType::Observation,
            map([("capture_time", Value::F64(f64::NAN))]),
        );
        let bytes = encode_envelope(&e, &Limits::default()).unwrap();
        let error = decode_envelope(&bytes, &Limits::default()).unwrap_err();
        assert_eq!(error.code, ErrorCode::Malformed);
    }

    #[test]
    fn msgpack_f32_non_finite_is_rejected() {
        // msgpack float32 (0xca) 解成 Value::F32，必须与 F64 同走非有限值闸门。
        let e = envelope(
            MessageType::Observation,
            map([("capture_time", Value::F32(f32::NAN))]),
        );
        let bytes = encode_envelope(&e, &Limits::default()).unwrap();
        let error = decode_envelope(&bytes, &Limits::default()).unwrap_err();
        assert_eq!(error.code, ErrorCode::Malformed);
    }

    #[test]
    fn absurd_shape_overflow_is_rejected() {
        // [2^32, 2^32]×float32 的乘积在 release 下回绕成 0，空 payload 不得通过字节校验。
        let record = map([
            (MARKER, Value::String("tensor".into())),
            ("dtype", Value::String("float32".into())),
            (
                "shape",
                Value::Array(vec![Value::from(1u64 << 32), Value::from(1u64 << 32)]),
            ),
            ("endianness", Value::String("little".into())),
            ("data", Value::Binary(Vec::new())),
        ]);
        let error = decode_record(&record).unwrap_err();
        assert_eq!(error.code, ErrorCode::Malformed);
    }

    #[test]
    fn duplicate_map_key_is_rejected() {
        let body = Value::Map(vec![
            (Value::String("task".into()), Value::String("a".into())),
            (Value::String("task".into()), Value::String("b".into())),
        ]);
        let bytes = encode_envelope(
            &envelope(MessageType::Observation, body),
            &Limits::default(),
        )
        .unwrap();
        let error = decode_envelope(&bytes, &Limits::default()).unwrap_err();
        assert_eq!(error.code, ErrorCode::Malformed);
    }

    #[test]
    fn tensor_record_byte_mismatch_is_rejected() {
        let record = tensor_record("float32", &[6], vec![0u8; 23]);
        let error = decode_record(&record).unwrap_err();
        assert_eq!(error.code, ErrorCode::Malformed);
    }

    fn specs() -> Vec<FeatureSpec> {
        vec![FeatureSpec {
            name: "observation.state".into(),
            shape: vec![6],
            dtype: "float32".into(),
            ..FeatureSpec::default()
        }]
    }

    #[test]
    fn validate_features_checks_shape_and_finiteness() {
        let good = map([(
            "observation.state",
            tensor_record("float32", &[6], vec![0u8; 24]),
        )]);
        assert!(validate_features(&good, &specs()).is_ok());

        let wrong_shape = map([(
            "observation.state",
            tensor_record("float32", &[5], vec![0u8; 20]),
        )]);
        let error = validate_features(&wrong_shape, &specs()).unwrap_err();
        assert_eq!(error.code, ErrorCode::Incompatible);

        let mut nan_bytes = vec![0u8; 24];
        nan_bytes[0..4].copy_from_slice(&f32::NAN.to_le_bytes());
        let nan = map([(
            "observation.state",
            tensor_record("float32", &[6], nan_bytes),
        )]);
        let error = validate_features(&nan, &specs()).unwrap_err();
        assert_eq!(error.code, ErrorCode::Malformed);
    }
}

#[cfg(test)]
mod shape_regression_tests {
    use super::*;

    #[test]
    fn malformed_shape_is_never_silently_reduced() {
        for (shape, data, diagnostic) in [
            (
                Some(Value::Array(vec![6.into(), (-1).into()])),
                vec![0; 24],
                "shape[1]",
            ),
            (Some(true.into()), vec![0; 4], "shape"),
            (None, vec![0; 4], "shape"),
        ] {
            let mut record = tensor_record("float32", &[], data);
            let Value::Map(entries) = &mut record else {
                panic!("expected map")
            };
            entries.retain(|(key, _)| key.as_str() != Some("shape"));
            if let Some(shape) = shape {
                entries.push(("shape".into(), shape));
            }
            let error = decode_record(&record).unwrap_err();
            assert_eq!(error.code, ErrorCode::Malformed);
            assert!(error.message.contains(diagnostic), "{error}");
        }
        assert!(decode_record(&tensor_record("float32", &[], vec![0; 4])).is_ok());
    }
}
