//! Live Qianfan checks. Requires `LLM_API_KEY`.
//! Run: `cargo test --test qianfan_live -- --ignored --nocapture`

use memora::{Embedder, QianfanConfig, QianfanEmbedder, QianfanSummarizer, SummarizeInput, Summarizer};

#[tokio::test]
#[ignore]
async fn qianfan_embed_and_summarize() {
    let cfg = QianfanConfig::from_env().expect("LLM_API_KEY");
    let embedder = QianfanEmbedder::new(cfg.clone()).unwrap();
    let vecs = embedder.embed(&["你好".into()]).await.unwrap();
    assert_eq!(vecs.len(), 1);
    assert!(vecs[0].len() > 8);

    let summarizer = QianfanSummarizer::new(cfg).unwrap();
    let out = summarizer
        .summarize(SummarizeInput {
            texts: vec!["用户偏好 SQLite 作为本地存储。".into()],
            entities: vec!["SQLite".into()],
        })
        .await
        .unwrap();
    assert!(!out.summary.is_empty());
}
