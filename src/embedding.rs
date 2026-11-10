use async_trait::async_trait;

use crate::error::Result;

#[async_trait]
pub trait Embedder: Send + Sync {
    fn model_id(&self) -> &str;
    fn dimension(&self) -> usize;
    /// Providers with auto-detected dimensions bind once, then reject drift.
    fn bind_dimension(&self, dim: usize) -> bool {
        self.dimension() == dim
    }
    /// Stable identity of the vector space, including provider/endpoint and settings.
    fn space_id(&self) -> String {
        format!("{}:{}", self.model_id(), self.dimension())
    }
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    async fn embed_query(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> { self.embed(texts).await }
    fn hedge_queries(&self) -> bool { true }
}

#[derive(Debug, Clone)]
pub struct SummarizeInput {
    pub texts: Vec<String>,
    pub entities: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SummarizeOutput {
    pub summary: String,
}

#[async_trait]
pub trait Summarizer: Send + Sync {
    fn summarizer_id(&self) -> &str;
    fn prompt_version(&self) -> &str;
    async fn summarize(&self, input: SummarizeInput) -> Result<SummarizeOutput>;
}

#[derive(Clone)]
pub struct HashEmbedder {
    dim: usize,
}

impl HashEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim: dim.max(8) }
    }
}

#[async_trait]
impl Embedder for HashEmbedder {
    fn model_id(&self) -> &str {
        "hash-ngram-v1"
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| hash_embed(t, self.dim)).collect())
    }
}

fn hash_embed(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return v;
    }
    for window in chars.windows(2) {
        let mut h: u64 = 0xcbf29ce484222325;
        for ch in window {
            h ^= *ch as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        let idx = (h as usize) % dim;
        let sign = if h & 1 == 0 { 1.0 } else { -1.0 };
        v[idx] += sign;
    }
    for ch in &chars {
        let mut h: u64 = 0x9e3779b97f4a7c15;
        h ^= *ch as u64;
        h = h.wrapping_mul(0x100000001b3);
        let idx = (h as usize) % dim;
        v[idx] += 0.5;
    }
    let norm = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x = (*x as f64 / norm) as f32;
        }
    }
    v
}

/// L2 over unit vectors is monotonic with cosine distance. Validate first so
/// non-finite, empty or zero vectors never enter HNSW.
pub fn normalize_vector(mut vector: Vec<f32>) -> Result<Vec<f32>> {
    let norm = vector
        .iter()
        .map(|v| (*v as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    if vector.is_empty() || !norm.is_finite() || norm <= 0.0 {
        return Err(crate::error::MemoraError::Embedding(
            "invalid embedding vector".into(),
        ));
    }
    for v in &mut vector {
        *v = (*v as f64 / norm) as f32;
    }
    Ok(vector)
}

pub fn vector_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

pub fn bytes_to_vector(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_roundtrip() {
        let v = vec![1.0f32, -2.5, 0.0];
        assert_eq!(bytes_to_vector(&vector_to_bytes(&v)), v);
    }

    #[tokio::test]
    async fn hash_embed_is_normalized() {
        let e = HashEmbedder::new(32);
        let v = e.embed(&["SQLite".into()]).await.unwrap();
        let n: f32 = v[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-4);
    }
}
