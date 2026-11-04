use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use hnsw_rs::prelude::*;

use crate::config::Config;
use crate::error::Result;
use crate::types::MemoryId;

pub struct VectorIndex {
    hnsw: Hnsw<'static, f32, DistL2>,
    labels: HashMap<usize, MemoryId>,
    ids: HashMap<MemoryId, usize>,
    deleted: HashSet<usize>,
    next_label: usize,
    path: PathBuf,
    dimension: Option<usize>,
}

impl VectorIndex {
    pub fn new(config: &Config, path: PathBuf) -> Self {
        Self {
            hnsw: Hnsw::<f32, DistL2>::new(
                config.hnsw_m,
                config.hnsw_max_elements,
                config.hnsw_max_layer,
                config.hnsw_ef_construct,
                DistL2 {},
            ),
            labels: HashMap::new(),
            ids: HashMap::new(),
            deleted: HashSet::new(),
            next_label: 1,
            path,
            dimension: None,
        }
    }

    pub fn rebuild(config: &Config, path: PathBuf, rows: Vec<(MemoryId, i64, Vec<f32>)>) -> Self {
        let mut idx = Self::new(config, path);
        let mut max_label = 0usize;
        for (id, label, vec) in rows {
            let Ok(vec) = crate::embedding::normalize_vector(vec) else {
                continue;
            };
            if idx.dimension.is_some_and(|dim| dim != vec.len()) {
                continue;
            }
            idx.dimension = Some(vec.len());
            let label = label as usize;
            max_label = max_label.max(label);
            idx.hnsw.insert((vec.as_slice(), label));
            idx.labels.insert(label, id);
            idx.ids.insert(id, label);
        }
        idx.next_label = max_label + 1;
        idx
    }

    pub fn upsert(&mut self, id: MemoryId, vector: Vec<f32>) -> Result<()> {
        let vector = crate::embedding::normalize_vector(vector)?;
        if self.dimension.is_some_and(|dim| dim != vector.len()) {
            return Err(crate::error::MemoraError::Embedding(
                "index dimension mismatch".into(),
            ));
        }
        self.dimension = Some(vector.len());
        if let Some(old) = self.ids.remove(&id) {
            self.deleted.insert(old);
            self.labels.remove(&old);
        }
        let label = self.next_label;
        self.next_label += 1;
        self.hnsw.insert((vector.as_slice(), label));
        self.labels.insert(label, id);
        self.ids.insert(id, label);
        self.deleted.remove(&label);
        Ok(())
    }

    pub fn remove(&mut self, id: MemoryId) {
        if let Some(label) = self.ids.remove(&id) {
            self.deleted.insert(label);
            self.labels.remove(&label);
        }
    }

    pub fn search(&self, query: &[f32], top_k: usize, ef: usize) -> Vec<(MemoryId, f32)> {
        if self.ids.is_empty() || top_k == 0 || self.dimension != Some(query.len()) {
            return Vec::new();
        }
        let Ok(query) = crate::embedding::normalize_vector(query.to_vec()) else {
            return Vec::new();
        };
        let count = top_k
            .saturating_add(self.deleted.len())
            .min(self.next_label.saturating_sub(1));
        let neighbours = self.hnsw.search(&query, count, ef.max(count));
        neighbours
            .into_iter()
            .filter(|n| !self.deleted.contains(&n.d_id))
            .filter_map(|n| self.labels.get(&n.d_id).map(|id| (*id, n.distance as f32)))
            .take(top_k)
            .collect()
    }

    pub fn path(&self) -> PathBuf {
        self.path.clone()
    }

    pub fn persist(&self) -> Result<()> {
        if let Some(dir) = self.path.parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir)?;
            }
        }
        let dir = self
            .path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        let base = self
            .path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("memora");
        let _ = self.hnsw.file_dump(dir, base);
        Ok(())
    }
}

#[cfg(test)]
mod safety_tests {
    use super::*;
    #[test]
    fn wrong_dimension_and_nonfinite_vectors_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let mut index = VectorIndex::new(&Config::default(), dir.path().join("index"));
        let id = uuid::Uuid::now_v7();
        index.upsert(id, vec![2.0, 0.0]).unwrap();
        assert!(index
            .upsert(uuid::Uuid::now_v7(), vec![1.0, 0.0, 0.0])
            .is_err());
        assert!(index
            .upsert(uuid::Uuid::now_v7(), vec![f32::NAN, 0.0])
            .is_err());
        assert!(index.search(&[1.0, 0.0, 0.0], 5, 64).is_empty());
        assert!(index.search(&[f32::NAN, 0.0], 5, 64).is_empty());
        assert_eq!(index.search(&[10.0, 0.0], 5, 64)[0].0, id);
    }
}
