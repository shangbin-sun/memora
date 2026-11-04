//! Family knowledge graph (finance / medical / education).
//!
//! # Concepts
//!
//! - **Entity** — a graph node: a referable, independently meaningful object with a
//!   type (person, school, condition, medication, …) and stable id. Aliases attach
//!   to the entity; pronouns / particles are not entities.
//! - **Relation** — a directed edge / predicate between two entities, forming a
//!   triple `(subject, relation, object)`. Open string labels are allowed; known
//!   [`RelationType`] values are optional vocabulary, not a hard whitelist.
//! - **Attribute** — scalar descriptors (age, dose, temperature) belong on
//!   `attrs` / `condition_text`, not as junk singleton nodes.
//!
//! Structured facts live alongside ordinary memories. `remember` / `recall`
//! share the same host-facing entry points; use [`service::query_person_graph`]
//! for a full per-person subgraph and [`service::grant_access`] for privacy grants.

pub mod extract;
pub mod extractor;
pub mod ingest;
pub mod ontology;
pub mod policy;
pub mod service;
pub mod store;
pub mod types;

pub use extract::{extract_graph, is_meaningful_person_label, is_meaningless_person_label};
pub use extractor::{
    kg_prompt, kg_prompt_with_context, kg_prompt_with_speaker, merge_facts,
    merge_graph_extract_outputs, parse_and_validate_llm_json, split_source_text,
    EmptyGraphExtractor, GraphExtractInput, GraphExtractOutput, GraphExtractor,
    HeuristicGraphExtractor, PendingItem, SOURCE_CHUNK_CHARS, SOURCE_CHUNK_OVERLAP,
};
pub use ingest::facts_to_graph;
pub use ontology::{
    default_privacy, EntityType, FactPolarity, FactStatus, PrivacyLevel, RelationType,
};
pub use service::{
    apply_extracted, grant_access, ingest_extracted, ingest_graph_batch, ingest_text, list_grants,
    query_graph_overview, query_person_graph, recall_graph, revoke_access,
};
pub use types::{
    EntityId, ExtractedEntity, ExtractedFact, ExtractedTriple, FactId, FactObject, GrantInput,
    GraphOverview, GraphOverviewQuery, IngestReport, KgEntity, KgFact, KgGrant, PersonGraph,
    PersonGraphQuery, RecalledGraphFact, RevokeGrantInput,
};
