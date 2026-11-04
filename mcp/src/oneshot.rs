//! One-shot `recall` / `remember` for Cursor hooks.
//! Stdout is a single JSON object. Logs stay on stderr.

use memora::{MemoryService, RecallQuery, RecallResult, RememberInput};

use crate::boot::Command;

pub async fn run(engine: memora::MemoryEngine, command: Command) -> anyhow::Result<()> {
    match command {
        Command::Recall { text, top_k } => {
            let text = take_text(text, "text")?;
            let top_k = top_k.clamp(1, 20);
            let result = engine
                .recall(RecallQuery {
                    text,
                    top_k: Some(top_k),
                    ..Default::default()
                })
                .await?;
            println!(
                "{}",
                serde_json::json!({
                    "ok": true,
                    "hits": slim_hits(&result, top_k),
                })
            );
        }
        Command::Remember { content } => {
            let content = take_text(content, "content")?;
            let task = tokio::spawn(async move {
                match engine.remember(RememberInput::episodic(content)).await {
                    Ok(id) => tracing::info!(%id, "async remember finished"),
                    Err(e) => tracing::warn!(error = %e, "async remember failed"),
                }
            });
            println!("{}", serde_json::json!({ "ok": true, "accepted": true }));
            let _ = std::io::Write::flush(&mut std::io::stdout());
            // Caller already has the accepted line. Keep this process until
            // the queued write finishes so it is not dropped on exit.
            let _ = task.await;
        }
    }
    Ok(())
}

fn take_text(flag: Option<String>, name: &str) -> anyhow::Result<String> {
    let from_flag = flag.unwrap_or_default();
    let text = if from_flag.trim().is_empty() {
        std::io::read_to_string(std::io::stdin())?
    } else {
        from_flag
    };
    let text = text.trim().to_string();
    if text.is_empty() {
        let err = format!("{name} is required");
        println!("{}", serde_json::json!({ "ok": false, "error": err }));
        anyhow::bail!(err);
    }
    Ok(text)
}

fn slim_hits(result: &RecallResult, top_k: usize) -> Vec<serde_json::Value> {
    let mut hits = Vec::new();
    for item in result.context.items.iter().take(top_k) {
        let text = clip(&item.text);
        if text.is_empty() {
            continue;
        }
        hits.push(serde_json::json!({ "text": text }));
    }
    if hits.is_empty() {
        for hit in result.hits.iter().take(top_k) {
            let raw = if !hit.display_text.trim().is_empty() {
                hit.display_text.as_str()
            } else {
                // Compatibility with results serialized before display_text existed.
                hit.memory
                    .summary
                    .as_deref()
                    .filter(|s| !s.trim().is_empty())
                    .unwrap_or(hit.memory.content.as_str())
            };
            let text = clip(raw);
            if text.is_empty() {
                continue;
            }
            hits.push(serde_json::json!({
                "text": text,
                "score": hit.scores.final_score,
            }));
        }
    }
    hits
}

fn clip(text: &str) -> String {
    let trimmed = text.trim();
    const MAX: usize = 400;
    if trimmed.chars().count() <= MAX {
        return trimmed.to_string();
    }
    let clipped: String = trimmed.chars().take(MAX).collect();
    format!("{clipped}…")
}
