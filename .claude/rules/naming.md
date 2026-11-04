# 模块命名

本仓有 ACT、SmolVLA、π₀.₅、Cosmos3 四个策略族。通用名（`TextModel`、`Attention`、`Mlp`、`RmsNorm`、`LayerNorm`、`VisionModel`、`KvCache`、`ExpertModel`）在多个族里重复。burn skill 的「去掉族前缀」约定假设单模型 crate，不适用于本仓。

- 策略目录内的模块类型带族前缀。策略级用 `SmolVla*`、`Pi05*`、`Cosmos3*`、`Act*`；模型组件用 `SmolVlm*` 等。
- 同一族内文本塔和视觉塔共用的概念，同时带塔前缀。例：`SmolVlmTextAttention`、`SmolVlmVisionAttention`。
- 结构体名可以改。字段名是权重键，不因命名风格改动；只有转换层显式重映射时才改。
- 只在单文件内使用、且不与其它族冲突的私有叶类型，可以保留短名。
- 新增策略时先检查是否已有同名类型；有冲突就加族前缀，不加别名。

现有对照：

| 族 | 策略级前缀 | 模型组件前缀 |
|---|---|---|
| SmolVLA | `SmolVla` | `SmolVlm` |
| π₀.₅ | `Pi05` | 待统一 |
| Cosmos3 | `Cosmos3` | 待统一 |
| ACT | `Act` | 待统一 |
