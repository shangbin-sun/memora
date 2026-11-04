//! Public knowledge-graph data types.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use super::ontology::{EntityType, FactPolarity, FactStatus, PrivacyLevel, RelationType};

pub type EntityId = String;
pub type FactId = Uuid;

/// Graph node: a referable, typed object with a stable id (and optional aliases).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KgEntity {
    pub id: EntityId,
    pub entity_type: EntityType,
    #[serde(default)]
    pub subtype: Option<String>,
    pub canonical_name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default = "default_entity_status")]
    pub status: String,
    #[serde(default)]
    pub domains: Vec<String>,
    #[serde(default)]
    pub owner_member_id: Option<String>,
    #[serde(default)]
    pub privacy: PrivacyLevel,
    pub first_seen_at_ms: i64,
    pub last_confirmed_at_ms: i64,
    #[serde(default)]
    pub attrs: Value,
}

fn default_entity_status() -> String {
    "active".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FactObject {
    Entity { entity_id: EntityId },
    Value { value_type: String, value: Value },
}

impl FactObject {
    pub fn entity(id: impl Into<String>) -> Self {
        Self::Entity {
            entity_id: id.into(),
        }
    }

    pub fn text(value: impl Into<String>) -> Self {
        Self::Value {
            value_type: "text".into(),
            value: Value::String(value.into()),
        }
    }
}

/// Stored directed edge: `(subject_id, relation, object)` plus provenance/privacy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KgFact {
    pub id: FactId,
    pub subject_id: EntityId,
    /// Open relation label (snake_case or Chinese); not limited to RelationType.
    pub relation: String,
    pub object: FactObject,
    #[serde(default)]
    pub owner_member_id: Option<String>,
    #[serde(default)]
    pub domains: Vec<String>,
    #[serde(default)]
    pub polarity: FactPolarity,
    #[serde(default)]
    pub status: FactStatus,
    #[serde(default)]
    pub valid_from_ms: Option<i64>,
    #[serde(default)]
    pub valid_to_ms: Option<i64>,
    #[serde(default)]
    pub condition_text: Option<String>,
    #[serde(default)]
    pub exception_text: Option<String>,
    #[serde(default)]
    pub reason_text: Option<String>,
    #[serde(default)]
    pub evidence_text: Option<String>,
    #[serde(default)]
    pub evidence_summary: Option<String>,
    #[serde(default)]
    pub source_memory_id: Option<Uuid>,
    #[serde(default)]
    pub confidence: f64,
    #[serde(default)]
    pub privacy: PrivacyLevel,
    #[serde(default)]
    pub revision: i64,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    #[serde(default)]
    pub attrs: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecalledGraphFact {
    pub fact: KgFact,
    pub subject_name: String,
    pub object_label: String,
    pub display_text: String,
    pub score: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PersonGraphQuery {
    /// Household member id (stable roster id) or person entity id.
    pub member_id: String,
    pub namespace: Option<String>,
    pub domains: Vec<String>,
    /// Include ended / corrected / withdrawn facts.
    pub include_ended: bool,
    /// Viewer for privacy filtering; when set, owner-private facts of others are hidden.
    pub viewer_member_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonGraph {
    pub member_id: String,
    pub person: Option<KgEntity>,
    pub entities: Vec<KgEntity>,
    pub facts: Vec<KgFact>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct GraphOverviewQuery {
    pub namespace: Option<String>,
    pub domains: Vec<String>,
    /// Include ended / corrected / withdrawn facts.
    pub include_ended: bool,
    /// Viewer for privacy filtering.
    pub viewer_member_id: Option<String>,
    /// Max facts to return (default 400).
    pub limit: Option<usize>,
}

/// Namespace-wide graph snapshot for Web visualization (no member required).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphOverview {
    pub entities: Vec<KgEntity>,
    pub facts: Vec<KgFact>,
    pub truncated: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestReport {
    pub entities_upserted: usize,
    pub facts_written: usize,
    pub facts_confirmed: usize,
    pub pending: usize,
    #[serde(default)]
    pub messages: Vec<String>,
    #[serde(default)]
    pub extractor_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KgGrant {
    pub id: String,
    pub grantor_member_id: String,
    pub grantee_member_id: String,
    /// fact | relation | domain | all
    pub scope_kind: String,
    /// Fact id / relation name / domain id depending on scope_kind; empty for all.
    #[serde(default)]
    pub scope_value: String,
    #[serde(default)]
    pub operations: Vec<String>,
    #[serde(default)]
    pub valid_from_ms: Option<i64>,
    #[serde(default)]
    pub valid_to_ms: Option<i64>,
    #[serde(default)]
    pub revoked: bool,
    #[serde(default)]
    pub evidence_text: Option<String>,
    pub created_at_ms: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct GrantInput {
    pub grantor_member_id: String,
    pub grantee_member_id: String,
    /// fact | relation | domain | all
    pub scope_kind: String,
    pub scope_value: String,
    pub operations: Vec<String>,
    pub valid_from_ms: Option<i64>,
    pub valid_to_ms: Option<i64>,
    pub evidence_text: Option<String>,
    pub namespace: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RevokeGrantInput {
    pub grant_id: Option<String>,
    pub grantor_member_id: Option<String>,
    pub grantee_member_id: Option<String>,
    pub scope_kind: Option<String>,
    pub scope_value: Option<String>,
    pub namespace: Option<String>,
}

/// Extract-time entity candidate (before resolve / merge). Must be referable + typed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    /// Batch-local id (e1) or roster member_id.
    pub temp_id: String,
    pub name: String,
    #[serde(default)]
    pub entity_type: EntityType,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub member_id: Option<String>,
}

/// Directed knowledge triple `(subject, relation, object)`.
///
/// Both endpoints must be entities (nodes). `relation` is an open directed
/// predicate string. LLM responses reference declared batch-local temp_ids;
/// aliases are persisted as source-linked text values, not separate people.
/// A single extract/ingest call may yield n≥0 triples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedTriple {
    pub subject: String,
    /// Concrete Chinese directed predicate; legacy English input is compatible.
    /// Open vocabulary, not limited to RelationType. Family object is subject's kin.
    pub relation: String,
    pub object: String,
    #[serde(default)]
    pub object_type: Option<EntityType>,
    #[serde(default)]
    pub polarity: FactPolarity,
    #[serde(default)]
    pub status: FactStatus,
    #[serde(default)]
    pub domains: Vec<String>,
    #[serde(default)]
    pub privacy: Option<PrivacyLevel>,
    #[serde(default)]
    pub condition_text: Option<String>,
    #[serde(default)]
    pub evidence_text: String,
    #[serde(default = "default_triple_conf")]
    pub confidence: f64,
    #[serde(default)]
    pub attrs: Value,
}

fn default_triple_conf() -> f64 {
    0.8
}

/// Legacy one-shot fact candidate (mapped into entities+triples on ingest).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedFact {
    pub subject_member_id: Option<String>,
    pub subject_name: String,
    /// Open relation string (known RelationType labels still used by heuristics).
    pub relation: String,
    pub object_type: EntityType,
    pub object_name: String,
    pub polarity: FactPolarity,
    pub status: FactStatus,
    pub domains: Vec<String>,
    pub privacy: PrivacyLevel,
    pub condition_text: Option<String>,
    pub evidence_text: String,
    pub confidence: f64,
    #[serde(default)]
    pub attrs: Value,
}

impl ExtractedFact {
    pub fn with_relation(mut self, rel: RelationType) -> Self {
        self.relation = rel.as_str().to_string();
        self
    }
}
