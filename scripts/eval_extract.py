#!/usr/bin/env python3
"""Memora 记忆提取评测：reconcile 入库 → GET /api/memories 列表打分（不召回）。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = ROOT / "tests" / "eval"
DEFAULT_OUT_DIR = ROOT / "tests" / "eval_out_extract"
DEFAULT_DATASET = "cockpit-extract-dialogue.json"


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        os.environ.setdefault(key, val)


def http_json(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 300.0,
) -> Any:
    data = None
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {url}: {err}") from e


def wait_health(base: str, timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            http_json("GET", f"{base}/api/health", timeout=3)
            return
        except Exception as e:  # noqa: BLE001
            last = str(e)
            time.sleep(0.5)
    raise RuntimeError(f"server health check failed: {last}")


def list_dataset_files(eval_dir: Path, names: list[str] | None) -> list[Path]:
    if names:
        files = []
        for n in names:
            p = Path(n)
            if not p.is_absolute():
                p = eval_dir / n
            if not p.exists():
                p2 = eval_dir / f"{n}.json" if not n.endswith(".json") else eval_dir / n
                p = p2
            if not p.is_file():
                raise FileNotFoundError(f"dataset not found: {n}")
            files.append(p)
        return files
    default = eval_dir / DEFAULT_DATASET
    if default.is_file():
        return [default]
    files = sorted(eval_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no json under {eval_dir}")
    return files


def load_samples(files: list[Path], limit: int | None) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for fp in files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{fp.name}: expected JSON array")
        for i, row in enumerate(data):
            inputs = [str(x).strip() for x in (row.get("input") or []) if str(x).strip()]
            answers_raw = row.get("answers")
            if isinstance(answers_raw, list):
                answers = [str(a) for a in answers_raw]
            elif row.get("answer"):
                answers = [str(row.get("answer"))]
            else:
                answers = []
            # Pad/trim answers to match input length.
            while len(answers) < len(inputs):
                answers.append("")
            answers = answers[: len(inputs)]
            samples.append(
                {
                    "id": f"{fp.stem}#{i}",
                    "dataset": fp.name,
                    "input": inputs,
                    "answers": answers,
                    "recall": str(row.get("recall") or "").strip(),
                    "expect_kw": str(row.get("expect_kw") or "").strip(),
                }
            )
            if limit is not None and len(samples) >= limit:
                return samples
    return samples


def normalize(s: str) -> str:
    return (
        s.lower()
        .replace(" ", "")
        .replace("　", "")
        .replace("℃", "度")
        .replace("°c", "度")
    )


def poll_reconcile_job(base: str, status_url: str, timeout_s: float = 600.0) -> dict[str, Any]:
    if status_url.startswith("/"):
        url = f"{base}{status_url}"
    else:
        url = status_url
    deadline = time.time() + timeout_s
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = http_json("GET", url, timeout=30) or {}
        status = str(last.get("status") or "")
        if status == "completed":
            return last.get("report") or last
        if status == "failed":
            raise RuntimeError(f"reconcile job failed: {last.get('error') or last}")
        time.sleep(1.0)
    raise RuntimeError(f"reconcile job timeout: {last}")


def reconcile_turn(
    base: str,
    query: str,
    answer: str,
    namespace: str,
    session_id: str,
) -> dict[str, Any]:
    body = {
        "query": query,
        "answer": answer or "",
        "namespace": namespace,
        "session_id": session_id,
    }
    resp = http_json("POST", f"{base}/api/reconcile", body=body, timeout=600)
    if not isinstance(resp, dict):
        return {}
    # Async job accepted.
    if "status_url" in resp or resp.get("status") in ("queued", "running"):
        return poll_reconcile_job(base, str(resp.get("status_url") or ""))
    return resp


def list_memories_for_session(
    base: str,
    namespace: str,
    session_id: str,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    """Pull real list API pages and filter by session_id client-side."""
    items: list[dict[str, Any]] = []
    offset = 0
    total = None
    while True:
        qs = urllib.parse.urlencode(
            {
                "namespace": namespace,
                "state": "active",
                "limit": page_size,
                "offset": offset,
            }
        )
        page = http_json("GET", f"{base}/api/memories?{qs}", timeout=60) or {}
        batch = page.get("items") or []
        if total is None:
            total = int(page.get("total") or 0)
        for mem in batch:
            sid = mem.get("session_id")
            if sid == session_id:
                items.append(mem)
        offset += len(batch)
        if not batch or (total is not None and offset >= total):
            break
        # Safety: stop if server returns fewer than page without advancing total.
        if len(batch) < page_size:
            break
    return items


def format_memories(memories: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for mem in memories:
        text = (mem.get("summary") or mem.get("content") or "").strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def score_extract(expect_kw: str, memories: list[dict[str, Any]]) -> tuple[float, str]:
    """Heuristic only: empty expect → zero memories; else keyword in listed text."""
    actual = format_memories(memories)
    n = len(memories)
    if not expect_kw:
        if n == 0:
            return 1.0, "heuristic:expect_empty"
        return 0.0, f"heuristic:false_positive n={n}"
    if n == 0:
        return 0.0, "heuristic:miss_empty"
    if normalize(expect_kw) in normalize(actual):
        return 1.0, "heuristic:substring"
    return 0.0, "heuristic:miss_kw"


def run_extract(
    base: str,
    samples: list[dict[str, Any]],
    namespace: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for i, s in enumerate(samples):
        session_id = s["id"]
        reports: list[dict[str, Any]] = []
        errors: list[str] = []
        for turn_i, query in enumerate(s["input"]):
            answer = s["answers"][turn_i] if turn_i < len(s["answers"]) else ""
            try:
                report = reconcile_turn(base, query, answer, namespace, session_id)
                reports.append(
                    {
                        "memories_extracted": report.get("memories_extracted"),
                        "memories_kept": report.get("memories_kept"),
                        "written_ids": report.get("written_ids"),
                        "skipped": report.get("skipped"),
                        "warnings_n": len(report.get("warnings") or []),
                    }
                )
            except Exception as e:  # noqa: BLE001
                errors.append(f"turn{turn_i}: {e}")
                reports.append({"error": str(e)})

        memories: list[dict[str, Any]] = []
        list_error = None
        try:
            memories = list_memories_for_session(base, namespace, session_id)
        except Exception as e:  # noqa: BLE001
            list_error = str(e)
            errors.append(f"list: {e}")

        actual_res = format_memories(memories)
        if list_error and not memories:
            score, reason = 0.0, f"list_error: {list_error}"
            judge = "error"
        else:
            score, reason = score_extract(s["expect_kw"], memories)
            judge = "heuristic"

        results.append(
            {
                "id": s["id"],
                "dataset": s["dataset"],
                "input": s["input"],
                "answers": s["answers"],
                "recall": s["recall"],
                "expect_kw": s["expect_kw"],
                "actual_res": actual_res,
                "memory_count": len(memories),
                "memories": [
                    {
                        "id": m.get("id"),
                        "content": m.get("content"),
                        "summary": m.get("summary"),
                        "kind": m.get("kind"),
                        "tag": m.get("tag"),
                    }
                    for m in memories
                ],
                "reconcile": reports,
                "score": score,
                "reason": reason,
                "judge": judge,
                "errors": errors,
            }
        )
        if (i + 1) % 5 == 0 or (i + 1) == len(samples):
            print(
                f"  extract {i + 1}/{len(samples)} "
                f"last={s['id']} score={score} n_mem={len(memories)}",
                flush=True,
            )
    return results


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(r["score"]) for r in rows if r.get("score") is not None]
    by_ds: dict[str, list[float]] = {}
    by_bucket = {"0": 0, "0.5": 0, "1": 0}
    neg = [r for r in rows if not r.get("expect_kw")]
    pos = [r for r in rows if r.get("expect_kw")]
    neg_ok = sum(1 for r in neg if float(r.get("score") or 0) >= 1.0)
    pos_ok = sum(1 for r in pos if float(r.get("score") or 0) >= 1.0)
    false_pos = [
        r for r in neg if float(r.get("score") or 0) < 1.0 and int(r.get("memory_count") or 0) > 0
    ]
    by_cat: dict[str, list[float]] = {}
    for r in rows:
        meta = str(r.get("recall") or "")
        parts = meta.split("|")
        cat = parts[1] if len(parts) >= 2 else "unknown"
        by_cat.setdefault(cat, []).append(float(r.get("score") or 0.0))
        sc = float(r.get("score") or 0.0)
        by_ds.setdefault(r["dataset"], []).append(sc)
        key = "1" if sc >= 1.0 else ("0.5" if sc >= 0.5 else "0")
        by_bucket[key] += 1
    overall = sum(scores) / len(scores) if scores else 0.0
    return {
        "n": len(rows),
        "avg_score": round(overall, 4),
        "sum_score": round(sum(scores), 4),
        "max_score": len(rows),
        "pct": round(100.0 * overall, 2),
        "buckets": by_bucket,
        "negative": {
            "n": len(neg),
            "ok": neg_ok,
            "false_positive": len(false_pos),
            "precision_empty": round(neg_ok / len(neg), 4) if neg else None,
        },
        "positive": {
            "n": len(pos),
            "ok": pos_ok,
            "hit_rate": round(pos_ok / len(pos), 4) if pos else None,
        },
        "per_category": {
            k: {
                "n": len(v),
                "avg": round(sum(v) / len(v), 4) if v else 0.0,
            }
            for k, v in sorted(by_cat.items())
        },
        "per_dataset": {
            k: {
                "n": len(v),
                "avg": round(sum(v) / len(v), 4) if v else 0.0,
                "sum": round(sum(v), 4),
            }
            for k, v in sorted(by_ds.items())
        },
    }


def write_report_md(
    path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    neg = summary["negative"]
    pos = summary["positive"]
    lines = [
        "# Memora Extract Eval Report",
        "",
        f"- samples: **{summary['n']}**",
        f"- avg score: **{summary['avg_score']}** / 1.0 ({summary['pct']}%)",
        f"- sum: {summary['sum_score']} / {summary['max_score']}",
        f"- buckets: {summary['buckets']}",
        f"- negative (expect empty): n={neg['n']} ok={neg['ok']} "
        f"false_positive={neg['false_positive']} "
        f"precision={neg['precision_empty']}",
        f"- positive (expect kw): n={pos['n']} ok={pos['ok']} hit_rate={pos['hit_rate']}",
        "",
        "## Per category",
        "",
        "| category | n | avg |",
        "|---|---:|---:|",
    ]
    for name, st in summary["per_category"].items():
        lines.append(f"| {name} | {st['n']} | {st['avg']} |")
    lines.extend(["", "## Failures (score=0, first 40)", ""])
    fails = [r for r in rows if float(r.get("score") or 0) <= 0.0]
    for r in fails[:40]:
        lines.append(
            f"- `{r['id']}` meta=`{r.get('recall')}` expect=`{r['expect_kw']}` "
            f"n_mem={r.get('memory_count')} reason={r.get('reason')} "
            f"actual={str(r.get('actual_res') or '')[:120]!r}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Memora memory extraction eval")
    p.add_argument(
        "--base-url",
        default=os.environ.get("MEMORA_EVAL_URL", "http://127.0.0.1:8791"),
    )
    p.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="dataset file names (default: cockpit-extract-dialogue.json)",
    )
    p.add_argument("--limit", type=int, default=None, help="max samples")
    p.add_argument("--namespace", default="eval-extract")
    p.add_argument(
        "--score-only",
        action="store_true",
        help="only re-score / rewrite report from existing result.json",
    )
    p.add_argument("--wait-health", action="store_true")
    return p.parse_args()


def main() -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    report_json = out_dir / "report.json"
    report_md = out_dir / "report.md"

    if args.wait_health or not args.score_only:
        print(f"waiting health {args.base_url} ...", flush=True)
        wait_health(args.base_url)

    if args.score_only:
        rows = json.loads(result_path.read_text(encoding="utf-8"))
        for row in rows:
            mems = row.get("memories") or []
            score, reason = score_extract(str(row.get("expect_kw") or ""), mems)
            row["score"] = score
            row["reason"] = reason
            row["judge"] = "heuristic"
    else:
        files = list_dataset_files(args.eval_dir, args.datasets)
        samples = load_samples(files, args.limit)
        print(
            f"loaded samples={len(samples)} datasets={len(files)} "
            f"inputs={sum(len(s['input']) for s in samples)} "
            f"positive={sum(1 for s in samples if s['expect_kw'])}",
            flush=True,
        )
        print("phase: reconcile + list", flush=True)
        rows = run_extract(args.base_url, samples, args.namespace)
        result_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {result_path}", flush=True)

    summary = summarize(rows)
    result_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "summary": summary,
        "config": {
            "base_url": args.base_url,
            "namespace": args.namespace,
            "limit": args.limit,
            "datasets": args.datasets or [DEFAULT_DATASET],
            "mode": "extract",
        },
    }
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report_md(report_md, summary, rows)
    print(
        f"DONE avg={summary['avg_score']} ({summary['pct']}%) "
        f"n={summary['n']} "
        f"neg_fp={summary['negative']['false_positive']} "
        f"pos_hit={summary['positive']['hit_rate']} "
        f"report={report_md}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
