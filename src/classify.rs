use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;
use crate::types::{MemoryId, MemoryKind};

#[derive(Debug, Clone)]
pub struct ClassifyItem {
    pub id: MemoryId,
    pub content: String,
    pub summary: Option<String>,
    pub kind: MemoryKind,
    pub entities: Vec<String>,
    pub importance: f64,
    pub confidence: f64,
    /// JSON array of roster entries shared by the batch. Empty when no household is registered.
    pub roster_json: String,
}

#[derive(Debug, Clone)]
pub struct ClassifyOutput {
    pub id: MemoryId,
    pub kind: MemoryKind,
    pub importance: f64,
    pub confidence: f64,
    pub entities: Option<Vec<String>>,
    pub archive: bool,
    pub patterns: Vec<String>,
    /// Classifier household subjects, when the model returned them.
    pub subjects: Option<serde_json::Value>,
    /// Classifier life-domain labels, when the model returned them.
    pub domains: Option<serde_json::Value>,
}

#[async_trait]
pub trait MemoryClassifier: Send + Sync {
    fn classifier_id(&self) -> &str;
    async fn classify_batch(&self, items: &[ClassifyItem]) -> Result<Vec<ClassifyOutput>>;
}

/// Keeps provisional scores/kind; marks archive when importance is below threshold.
#[derive(Debug, Clone)]
pub struct HeuristicClassifier {
    archive_importance: f64,
}

impl HeuristicClassifier {
    pub fn new(archive_importance: f64) -> Self {
        Self {
            archive_importance: archive_importance.clamp(0.0, 1.0),
        }
    }

    pub fn classify_items(&self, items: &[ClassifyItem]) -> Vec<ClassifyOutput> {
        items
            .iter()
            .map(|item| {
                let importance = item.importance.clamp(0.0, 1.0);
                let confidence = item.confidence.clamp(0.0, 1.0);
                ClassifyOutput {
                    id: item.id,
                    kind: item.kind,
                    importance,
                    confidence,
                    entities: None,
                    archive: importance < self.archive_importance,
                    patterns: Vec::new(),
                    subjects: None,
                    domains: None,
                }
            })
            .collect()
    }
}

#[async_trait]
impl MemoryClassifier for HeuristicClassifier {
    fn classifier_id(&self) -> &str {
        "heuristic-v1"
    }

    async fn classify_batch(&self, items: &[ClassifyItem]) -> Result<Vec<ClassifyOutput>> {
        Ok(self.classify_items(items))
    }
}

pub fn parse_classify_json(body: &str, fallback: &[ClassifyItem]) -> Result<Vec<ClassifyOutput>> {
    let trimmed = body
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    let value: serde_json::Value = serde_json::from_str(trimmed)
        .map_err(|e| crate::error::MemoraError::Embedding(format!("classify json: {e}")))?;
    let arr = value.as_array().ok_or_else(|| {
        crate::error::MemoraError::Embedding("classify response is not an array".into())
    })?;

    let mut by_id: std::collections::HashMap<MemoryId, &ClassifyItem> =
        fallback.iter().map(|i| (i.id, i)).collect();
    let mut out = Vec::with_capacity(arr.len());
    for entry in arr {
        let id_s = entry.get("id").and_then(|v| v.as_str()).ok_or_else(|| {
            crate::error::MemoraError::Embedding("classify item missing id".into())
        })?;
        let id = Uuid::parse_str(id_s)
            .map_err(|e| crate::error::MemoraError::Embedding(format!("classify id: {e}")))?;
        let provisional = by_id.remove(&id);
        let kind = entry
            .get("kind")
            .and_then(|v| v.as_str())
            .and_then(|s| MemoryKind::parse(s).ok())
            .or_else(|| provisional.map(|p| p.kind))
            .unwrap_or(MemoryKind::Episodic);
        let importance = entry
            .get("importance")
            .and_then(|v| v.as_f64())
            .or_else(|| provisional.map(|p| p.importance))
            .unwrap_or(0.5)
            .clamp(0.0, 1.0);
        let confidence = entry
            .get("confidence")
            .and_then(|v| v.as_f64())
            .or_else(|| provisional.map(|p| p.confidence))
            .unwrap_or(0.8)
            .clamp(0.0, 1.0);
        let entities = entry
            .get("entities")
            .and_then(|v| serde_json::from_value::<Vec<String>>(v.clone()).ok());
        let archive = entry
            .get("archive")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let patterns = entry
            .get("patterns")
            .and_then(|v| serde_json::from_value::<Vec<String>>(v.clone()).ok())
            .unwrap_or_default()
            .into_iter()
            .filter(|s| !s.trim().is_empty())
            .collect();
        let subjects = entry.get("subjects").cloned();
        let domains = entry.get("domains").cloned();
        out.push(ClassifyOutput {
            id,
            kind,
            importance,
            confidence,
            entities,
            archive,
            patterns,
            subjects,
            domains,
        });
    }

    // Preserve order of input for any items the model omitted.
    for item in fallback {
        if out.iter().any(|o| o.id == item.id) {
            continue;
        }
        out.push(ClassifyOutput {
            id: item.id,
            kind: item.kind,
            importance: item.importance.clamp(0.0, 1.0),
            confidence: item.confidence.clamp(0.0, 1.0),
            entities: None,
            archive: false,
            patterns: Vec::new(),
            subjects: None,
            domains: None,
        });
    }
    Ok(out)
}
