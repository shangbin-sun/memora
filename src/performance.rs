//! Privacy-safe stage timings for host-controlled operations.
//!
//! This module only emits `DEBUG` tracing events; it neither changes responses
//! nor stores user data. Scope/stage labels must be fixed names in call sites,
//! never query text, member IDs, content, or provider error messages. A terminal
//! event is emitted on drop, including early returns, unwinding and future
//! cancellation; it does not claim that the operation succeeded.

use std::time::{Duration, Instant};

/// A lightweight monotonic checkpoint timer. Stage durations cover the time
/// since construction/the previous mark, while total time is cumulative.
pub(crate) struct StageTimer {
    scope: &'static str,
    started: Instant,
    last: Instant,
}

impl StageTimer {
    pub(crate) fn new(scope: &'static str) -> Self {
        let started = Instant::now();
        Self {
            scope,
            started,
            last: started,
        }
    }

    pub(crate) fn mark(&mut self, stage: &'static str) {
        let now = Instant::now();
        emit(Timing {
            scope: self.scope,
            stage,
            elapsed_ms: millis(now.duration_since(self.last)),
            total_elapsed_ms: millis(now.duration_since(self.started)),
            terminal: false,
        });
        self.last = now;
    }
}

impl Drop for StageTimer {
    fn drop(&mut self) {
        let total = millis(self.started.elapsed());
        emit(Timing {
            scope: self.scope,
            stage: "total",
            elapsed_ms: total,
            total_elapsed_ms: total,
            terminal: true,
        });
    }
}

fn millis(duration: Duration) -> u64 {
    duration.as_millis().min(u64::MAX as u128) as u64
}

#[derive(Debug, Clone, Copy)]
struct Timing {
    scope: &'static str,
    stage: &'static str,
    elapsed_ms: u64,
    total_elapsed_ms: u64,
    terminal: bool,
}

#[cfg(not(test))]
fn emit(timing: Timing) {
    tracing::debug!(
        target: "memora::performance",
        scope = timing.scope,
        stage = timing.stage,
        elapsed_ms = timing.elapsed_ms,
        total_elapsed_ms = timing.total_elapsed_ms,
        terminal = timing.terminal,
        "memory operation timing"
    );
}

// This private test sink permits direct `rustc --test src/performance.rs`
// validation without downloading dependencies or running a tracing subscriber.
#[cfg(test)]
std::thread_local! {
    static EVENTS: std::cell::RefCell<Vec<Timing>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
fn emit(timing: Timing) {
    EVENTS.with(|events| events.borrow_mut().push(timing));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn take_events() -> Vec<Timing> {
        EVENTS.with(|events| std::mem::take(&mut *events.borrow_mut()))
    }

    #[test]
    fn checkpoints_are_monotonic_and_drop_emits_one_total() {
        take_events();
        let mut timer = StageTimer::new("recall");
        timer.mark("prepare");
        timer.mark("retrieve");
        drop(timer);
        let events = take_events();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].scope, "recall");
        assert_eq!(events[0].stage, "prepare");
        assert_eq!(events[1].stage, "retrieve");
        assert_eq!(events[2].stage, "total");
        assert!(!events[0].terminal && !events[1].terminal && events[2].terminal);
        assert_eq!(events[0].elapsed_ms, events[0].total_elapsed_ms);
        assert!(events[1].elapsed_ms <= events[1].total_elapsed_ms);
        assert!(events[0].total_elapsed_ms <= events[1].total_elapsed_ms);
        assert!(events[1].total_elapsed_ms <= events[2].total_elapsed_ms);
        assert_eq!(events[2].elapsed_ms, events[2].total_elapsed_ms);
    }

    #[test]
    fn early_return_without_a_checkpoint_still_reports_total() {
        fn operation() -> Result<(), ()> {
            let _timer = StageTimer::new("reconcile");
            Err(())
        }
        take_events();
        assert_eq!(operation(), Err(()));
        let events = take_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].scope, "reconcile");
        assert_eq!(events[0].stage, "total");
        assert!(events[0].terminal);
    }

    #[test]
    fn dropped_pending_operation_does_not_claim_success() {
        take_events();
        // Poll once so the timer is created inside the actual pending future,
        // then cancel it by dropping. No executor or Tokio dependency is needed.
        struct NoopWake;
        impl std::task::Wake for NoopWake {
            fn wake(self: std::sync::Arc<Self>) {}
        }
        let waker = std::task::Waker::from(std::sync::Arc::new(NoopWake));
        let mut context = std::task::Context::from_waker(&waker);
        let mut pending = Box::pin(async {
            let _timer = StageTimer::new("candidate_recall");
            std::future::pending::<()>().await;
        });
        assert!(std::future::Future::poll(pending.as_mut(), &mut context).is_pending());
        drop(pending);
        let events = take_events();
        assert_eq!(events.len(), 1);
        assert!(events[0].terminal);
        assert_eq!(events[0].stage, "total");
    }

    #[test]
    fn milliseconds_are_truncated_and_large_values_saturate() {
        assert_eq!(millis(Duration::from_nanos(999_999)), 0);
        assert_eq!(millis(Duration::from_micros(1_999)), 1);
        assert_eq!(millis(Duration::MAX), u64::MAX);
    }
}
