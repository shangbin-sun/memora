//! Deterministic classifier duplicate regressions; no network or production DB.
use async_trait::async_trait;
use memora::{ClassifyItem, ClassifyOutput, DreamOptions, ListQuery, MemoryClassifier,
    MemoryEngine, MemoryKind, MemoryService, MemoryState, RelationKind, RememberInput};
use rusqlite::{params, Connection};
use serde_json::{json, Value};
use std::sync::Arc;

struct EchoClassifier { kind: MemoryKind, extra: Option<&'static str> }
#[async_trait]
impl MemoryClassifier for EchoClassifier {
    fn classifier_id(&self) -> &str { "echo-regression" }
    async fn classify_batch(&self, items: &[ClassifyItem]) -> memora::Result<Vec<ClassifyOutput>> {
        Ok(items.iter().map(|i| ClassifyOutput {
            id: i.id, kind: self.kind, importance: 0.8, confidence: 0.9,
            entities: None, archive: false,
            patterns: vec![format!("  {}  ", i.content), self.extra.unwrap_or("").into()],
            subjects: None, domains: None,
        }).collect())
    }
}
async fn setup(classifier: Option<EchoClassifier>) -> (tempfile::TempDir, MemoryEngine) {
    let dir = tempfile::tempdir().unwrap();
    let mut builder = MemoryEngine::builder().db_path(dir.path().join("memory.db"));
    if let Some(c) = classifier { builder = builder.classifier(Arc::new(c)); }
    let engine = builder.build().await.unwrap();
    (dir, engine)
}
async fn add(engine: &MemoryEngine, content: &str, kind: MemoryKind, metadata: Value, pending: bool) -> memora::MemoryId {
    let mut input = RememberInput::episodic(content);
    input.kind = kind;
    input.skip_consolidation = !pending;
    input.metadata = metadata;
    engine.remember(input).await.unwrap()
}
fn connection(dir: &tempfile::TempDir) -> Connection { Connection::open(dir.path().join("memory.db")).unwrap() }
async fn active(engine: &MemoryEngine) -> Vec<memora::MemoryRecord> {
    engine.list(ListQuery { state: Some(MemoryState::Active), limit: Some(200), ..Default::default() }).await.unwrap().items
}
fn link(conn: &Connection, child: memora::MemoryId, source: memora::MemoryId) {
    conn.execute("INSERT INTO memory_relations VALUES (?1,?2,'derived_from',1,0)", params![child.to_string(),source.to_string()]).unwrap();
}
async fn legacy_pair(engine: &MemoryEngine, dir: &tempfile::TempDir, kind: MemoryKind) -> (memora::MemoryId,memora::MemoryId) {
    let metadata = json!({"event_at":"2026-10-25", "privacy":true, "subjects":[{"member_id":"father","confidence":1.0}]});
    let source = add(engine,"爸爸经常熬夜加班",kind,metadata,false).await;
    let mut clone_meta = engine.get(source).await.unwrap().unwrap().metadata;
    clone_meta["dream"] = json!(true); clone_meta["from_classifier"] = json!(true);
    let child = add(engine,"爸爸经常熬夜加班",MemoryKind::Pattern,clone_meta,false).await;
    // Simulate the old engine's exact metadata clone, not a fresh annotation.
    let source_meta = engine.get(source).await.unwrap().unwrap().metadata;
    let mut child_meta = source_meta.clone(); child_meta["dream"]=json!(true); child_meta["from_classifier"]=json!(true);
    let conn = connection(dir);
    conn.execute("UPDATE memories SET metadata_json=?2 WHERE id=?1",params![child.to_string(),child_meta.to_string()]).unwrap();
    link(&conn,child,source);
    (source,child)
}

#[tokio::test]
async fn classifier_echo_never_duplicates_promoted_or_semantic_source() {
    for kind in [MemoryKind::Pattern, MemoryKind::Semantic] {
        let (_d,e)=setup(Some(EchoClassifier{kind,extra:None})).await;
        let id=add(&e,"爸爸经常熬夜加班",MemoryKind::Episodic,json!({}),true).await;
        e.dream(DreamOptions::default()).await.unwrap();
        let rows=active(&e).await;
        assert_eq!(rows.len(),1); assert_eq!(rows[0].id,id); assert_eq!(rows[0].kind,kind);
        assert_eq!(e.explain(id).await.unwrap().versions.len(),2);
    }
}

#[tokio::test]
async fn different_pattern_keeps_provenance_and_reclassification_is_idempotent() {
    let (d,e)=setup(Some(EchoClassifier{kind:MemoryKind::Pattern,extra:Some("爸爸工作繁忙")})).await;
    let id=add(&e,"爸爸经常熬夜加班",MemoryKind::Episodic,json!({}),true).await;
    e.dream(DreamOptions::default()).await.unwrap();
    assert_eq!(active(&e).await.len(),2);
    let derived=active(&e).await.into_iter().find(|r|r.id!=id).unwrap();
    assert!(e.explain(derived.id).await.unwrap().relations.iter().any(|r|r.to_id==id && r.relation==RelationKind::DerivedFrom));
    connection(&d).execute("UPDATE memories SET consolidation_pending=1 WHERE id=?1",[id.to_string()]).unwrap();
    e.dream(DreamOptions::default()).await.unwrap();
    assert_eq!(active(&e).await.len(),2);
    assert_eq!(e.deduplicate_exact_memories().await.unwrap(),0);
}

#[tokio::test]
async fn maintenance_repairs_legacy_self_clones_keeps_source_id_and_history() {
    for kind in [MemoryKind::Pattern,MemoryKind::Semantic] {
        let (d,e)=setup(None).await;
        let (source,child)=legacy_pair(&e,&d,kind).await;
        let before=e.explain(source).await.unwrap().versions.len();
        assert_eq!(e.deduplicate_exact_memories().await.unwrap(),1);
        assert_eq!(active(&e).await.iter().map(|r|r.id).collect::<Vec<_>>(),vec![source]);
        assert_eq!(e.get(source).await.unwrap().unwrap().kind,kind);
        assert_eq!(e.get(child).await.unwrap().unwrap().state,MemoryState::Superseded);
        assert_eq!(e.explain(source).await.unwrap().versions.len(),before);
        let explain=e.explain(child).await.unwrap();
        assert!(explain.relations.iter().any(|r|r.from_id==child && r.to_id==source && r.relation==RelationKind::DerivedFrom));
        assert!(!explain.relations.iter().any(|r|r.from_id==source && r.to_id==child));
        assert!(explain.versions.iter().any(|v|v.operation=="deduplicate"));
        assert_eq!(e.deduplicate_exact_memories().await.unwrap(),0);
    }
}

#[tokio::test]
async fn exact_patterns_deduplicate_but_security_and_event_boundaries_do_not() {
    let (d,e)=setup(None).await;
    let a=add(&e,"相同习惯",MemoryKind::Pattern,json!({}),false).await;
    let b=add(&e,"相同习惯",MemoryKind::Pattern,json!({}),false).await;
    assert_eq!(e.deduplicate_exact_memories().await.unwrap(),1);
    assert_eq!([a,b].iter().filter(|id|connection(&d).query_row("SELECT state='active' FROM memories WHERE id=?1",[id.to_string()],|r|r.get::<_,bool>(0)).unwrap()).count(),1);
    for (field,value) in [("privacy",json!(true)),("attributes",json!(["privacy"])),("sensitivity",json!("high")),
        ("event_at",json!("2026-10-26")),("event_at_ms",json!(42)),("periodicity",json!("daily")),
        ("readers",json!(["guest"])),("owners",json!(["other"])),("unknown_acl",json!({"deny":true}))] {
        let (d,e)=setup(None).await;
        let (source,child)=legacy_pair(&e,&d,MemoryKind::Pattern).await;
        let conn=connection(&d);
        let raw:String=conn.query_row("SELECT metadata_json FROM memories WHERE id=?1",[child.to_string()],|r|r.get(0)).unwrap();
        let mut meta:Value=serde_json::from_str(&raw).unwrap();
        // privacy already true in fixture, use false to ensure mismatch.
        meta[field]=if field=="privacy" {json!(false)} else {value};
        conn.execute("UPDATE memories SET metadata_json=?2 WHERE id=?1",params![child.to_string(),meta.to_string()]).unwrap();
        assert_eq!(e.deduplicate_exact_memories().await.unwrap(),0,"field {field}");
        assert_eq!(e.get(source).await.unwrap().unwrap().state,MemoryState::Active);
        assert_eq!(e.get(child).await.unwrap().unwrap().state,MemoryState::Active);
    }
}

#[tokio::test]
async fn graph_supported_sources_and_resource_policies_are_never_withdrawn() {
    for protect_child in [false,true] {
        let (d,e)=setup(None).await;
        let (source,child)=legacy_pair(&e,&d,MemoryKind::Pattern).await;
        let graph_source=if protect_child {child} else {source};
        let conn=connection(&d);
        conn.execute("INSERT INTO kg_entities(namespace,id,entity_type,canonical_name,first_seen_at_ms,last_confirmed_at_ms,created_at_ms,updated_at_ms) VALUES('default','test-person','person','father',0,0,0,0)",[]).unwrap();
        conn.execute("INSERT INTO kg_facts(namespace,id,subject_id,relation,object_value_json,semantic_key,first_seen_at_ms,last_confirmed_at_ms,created_at_ms,updated_at_ms) VALUES('default','test-fact','test-person','habit','true','test-key',0,0,0,0)",[]).unwrap();
        conn.execute("INSERT INTO kg_fact_sources(namespace,fact_id,source_memory_id,created_at_ms,evidence_text) VALUES('default','test-fact',?1,0,'evidence')",[graph_source.to_string()]).unwrap();
        let before:String=conn.query_row("SELECT source_memory_id FROM kg_fact_sources",[],|r|r.get(0)).unwrap();
        assert_eq!(e.deduplicate_exact_memories().await.unwrap(),if protect_child {0}else{1});
        let after:String=conn.query_row("SELECT source_memory_id FROM kg_fact_sources",[],|r|r.get(0)).unwrap();
        assert_eq!(before,after);
        assert_eq!(e.get(graph_source).await.unwrap().unwrap().state,MemoryState::Active);
    }
    for table in ["kg_resource_owners","kg_resource_domains","kg_grants"] {
        let (d,e)=setup(None).await;
        let (_source,child)=legacy_pair(&e,&d,MemoryKind::Pattern).await;
        let conn=connection(&d);
        match table {
            "kg_resource_owners"=> {conn.execute("INSERT INTO kg_resource_owners VALUES('default','memory',?1,'owner')",[child.to_string()]).unwrap();}
            "kg_resource_domains"=> {conn.execute("INSERT INTO kg_resource_domains VALUES('default','memory',?1,'private')",[child.to_string()]).unwrap();}
            _=> {conn.execute("INSERT INTO kg_grants VALUES('default','g','owner','reader',?1,'[\"read\"]',NULL,NULL,0,NULL,0)",[json!({"kind":"memory","value":child}).to_string()]).unwrap();}
        }
        assert_eq!(e.deduplicate_exact_memories().await.unwrap(),0,"table {table}");
        assert_eq!(active(&e).await.len(),2);
    }
}
