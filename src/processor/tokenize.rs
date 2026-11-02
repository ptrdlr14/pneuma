use std::{collections::BTreeMap, path::Path};

use anyhow::{Context as _, Result, bail, ensure};
use burn::tensor::{Bool, Int, Tensor, TensorData};
use serde_json::Value;
use tokenizers::{
    PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
    TruncationParams,
};

use super::{
    Batch, Context, Data,
    step::{MASK, TOKENS},
};

#[derive(Debug)]
pub(super) struct Prepared {
    pub tokenizer: Tokenizer,
    pub files: BTreeMap<String, Vec<u8>>,
}

pub(super) fn load(
    dir: &Path,
    max_length: usize,
    padding: &str,
    padding_side: &str,
) -> Result<Prepared> {
    let files = ["tokenizer.json", "tokenizer_config.json"]
        .into_iter()
        .map(|name| Ok((name.to_string(), std::fs::read(dir.join(name))?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    from_files(files, max_length, padding, padding_side)
}

pub(super) fn from_files(
    files: BTreeMap<String, Vec<u8>>,
    max_length: usize,
    padding: &str,
    padding_side: &str,
) -> Result<Prepared> {
    let metadata: Value = serde_json::from_slice(
        files
            .get("tokenizer_config.json")
            .context("missing tokenizer config")?,
    )
    .context("parsing tokenizer_config.json")?;
    let mut tokenizer = Tokenizer::from_bytes(
        files
            .get("tokenizer.json")
            .context("missing tokenizer data")?,
    )
    .map_err(|e| anyhow::anyhow!("loading local tokenizer: {e}"))?;
    let pad_token = metadata
        .get("pad_token")
        .and_then(|v| {
            v.as_str()
                .or_else(|| v.get("content").and_then(Value::as_str))
        })
        .context("tokenizer_config.json must define pad_token")?;
    let pad_id = tokenizer
        .token_to_id(pad_token)
        .context("pad_token missing from tokenizer vocabulary")?;
    if let Some(id) = metadata.get("pad_token_id") {
        ensure!(
            id.as_u64() == Some(u64::from(pad_id)),
            "tokenizer pad_token_id disagrees with vocabulary"
        );
    }
    let direction = match metadata
        .get("truncation_side")
        .and_then(Value::as_str)
        .unwrap_or("right")
    {
        "right" => TruncationDirection::Right,
        "left" => TruncationDirection::Left,
        other => bail!("unsupported tokenizer truncation_side={other}"),
    };
    let padding_direction = match padding_side {
        "right" => PaddingDirection::Right,
        "left" => PaddingDirection::Left,
        other => bail!("unsupported tokenizer padding_side={other}"),
    };
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length,
            direction,
            ..Default::default()
        }))
        .map_err(|e| anyhow::anyhow!("configuring tokenizer truncation: {e}"))?;
    let strategy = match padding {
        "max_length" => PaddingStrategy::Fixed(max_length),
        "longest" => PaddingStrategy::BatchLongest,
        other => bail!("unsupported tokenizer padding={other}"),
    };
    tokenizer.with_padding(Some(PaddingParams {
        strategy,
        direction: padding_direction,
        pad_id,
        pad_token: pad_token.into(),
        ..Default::default()
    }));
    Ok(Prepared { tokenizer, files })
}

pub(super) fn run(batch: &mut Batch, context: &Context, tokenizer: &Tokenizer) -> Result<()> {
    ensure!(
        !batch.tasks.is_empty(),
        "tokenizer requires a nonempty task batch"
    );
    for (key, data) in &batch.values {
        ensure!(
            data.shape().first() == Some(&batch.tasks.len()),
            "task count differs from batch dimension of {key}"
        );
    }
    let encodings = tokenizer
        .encode_batch(batch.tasks.clone(), true)
        .map_err(|e| anyhow::anyhow!("tokenizing task batch: {e}"))?;
    let length = encodings.first().context("empty token encoding")?.len();
    ensure!(
        length > 0 && encodings.iter().all(|e| e.len() == length),
        "tokenizer produced an empty or ragged batch"
    );
    let shape = [encodings.len(), length];
    let ids: Vec<i64> = encodings
        .iter()
        .flat_map(|e| e.get_ids().iter().map(|n| i64::from(*n)))
        .collect();
    let mask: Vec<bool> = encodings
        .iter()
        .flat_map(|e| e.get_attention_mask().iter().map(|n| *n != 0))
        .collect();
    batch.values.insert(
        TOKENS.into(),
        Data::Int2(Tensor::<2, Int>::from_data(
            TensorData::new(ids, shape),
            &context.device,
        )),
    );
    batch.values.insert(
        MASK.into(),
        Data::Bool2(Tensor::<2, Bool>::from_data(
            TensorData::new(mask, shape),
            &context.device,
        )),
    );
    Ok(())
}
