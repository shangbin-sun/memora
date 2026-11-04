use moka::future::Cache;
use serde::{Deserialize, Serialize};

use crate::types::MemoryId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StmItem {
    pub key: String,
    /// Linked entries are already in LTM, never promote them as new facts.
    pub memory_id: Option<MemoryId>,
    pub revision: i64,
    pub content: String,
    pub importance: f64,
    pub pinned: bool,
    pub kind: String,
    pub session_id: String,
    pub namespace: String,
    pub entities: Vec<String>,
    pub metadata: serde_json::Value,
}

#[derive(Clone)]
pub struct ShortTermMemory {
    pub session_events: Cache<String, StmItem>,
    pub working_facts: Cache<String, StmItem>,
    pub query_cache: Cache<String, crate::types::RecallResult>,
}

impl ShortTermMemory {
    pub fn new(config: &crate::config::Config) -> Self {
        let session_events = Cache::builder()
            .max_capacity(config.session_events_capacity)
            .time_to_live(config.session_events_ttl)
            .build();
        let working_facts = Cache::builder()
            .max_capacity(config.working_facts_capacity)
            .time_to_live(config.working_facts_ttl)
            .build();
        let query_cache = Cache::builder()
            .max_capacity(config.query_cache_capacity)
            .time_to_live(config.query_cache_ttl)
            .build();
        Self {
            session_events,
            working_facts,
            query_cache,
        }
    }

    pub fn cache_key(namespace: &str, session_id: &str, key: &str) -> String {
        format!("{namespace}/{session_id}/{key}")
    }

    pub async fn put_event(&self, item: StmItem) {
        self.session_events.insert(item.key.clone(), item).await;
    }

    pub async fn put_fact(&self, item: StmItem) {
        self.working_facts.insert(item.key.clone(), item).await;
    }

    pub async fn remove_memories(&self, ids: &[MemoryId]) {
        for (key, item) in self.session_events.iter() {
            if item.memory_id.is_some_and(|id| ids.contains(&id)) {
                self.session_events.invalidate(key.as_str()).await;
            }
        }
        for (key, item) in self.working_facts.iter() {
            if item.memory_id.is_some_and(|id| ids.contains(&id)) {
                self.working_facts.invalidate(key.as_str()).await;
            }
        }
    }

    pub async fn matching_facts(
        &self,
        namespace: &str,
        session_id: Option<&str>,
        query: &str,
    ) -> Vec<StmItem> {
        let q = query.to_lowercase();
        self.working_facts
            .iter()
            .filter(|(_, item)| {
                if item.namespace != namespace {
                    return false;
                }
                if let Some(sid) = session_id {
                    if item.session_id != sid {
                        return false;
                    }
                }
                item.content.to_lowercase().contains(&q)
                    || q.is_empty()
                    || item
                        .entities
                        .iter()
                        .any(|e| e.to_lowercase().contains(&q) || q.contains(&e.to_lowercase()))
            })
            .map(|(_, item)| item)
            .collect()
    }
}
