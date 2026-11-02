//! 模型、部署与协议共享的特征契约，形状匹配不替代语义约定。

use serde::{Deserialize, Serialize};

/// 一个具名特征的数值表示、组件顺序与物理语义。
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
