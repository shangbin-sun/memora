#!/usr/bin/env python3
"""Memora 记忆评测：写入 → 召回 → 批量 LLM 打分。"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from datetime import datetime
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = ROOT / "tests" / "eval"
DEFAULT_OUT_DIR = ROOT / "tests" / "eval_out"


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
    timeout: float = 120.0,
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
            samples.append(
                {
                    "id": f"{fp.stem}#{i}",
                    "dataset": fp.name,
                    "input": list(row.get("input") or []),
                    "recall": str(row.get("recall") or "").strip(),
                    "expect_kw": str(row.get("expect_kw") or "").strip(),
                }
            )
            if limit is not None and len(samples) >= limit:
                return samples
    return samples


def format_actual(result: dict[str, Any], top_k: int) -> str:
    hits = result.get("hits") or []
    lines: list[str] = []
    for h in hits[:top_k]:
        mem = h.get("memory") or {}
        text = (mem.get("summary") or mem.get("content") or "").strip()
        scores = h.get("scores") or {}
        score = scores.get("final_score")
        prefix = f"[{score:.3f}] " if isinstance(score, (int, float)) else ""
        if text:
            lines.append(prefix + text)
    if not lines:
        for mem in result.get("stm_hits") or []:
            text = (mem.get("summary") or mem.get("content") or "").strip()
            if text:
                lines.append("[stm] " + text)
            if len(lines) >= top_k:
                break
    return "\n".join(lines)


def normalize(s: str) -> str:
    return (
        s.lower()
        .replace(" ", "")
        .replace("　", "")
        .replace("℃", "度")
        .replace("°c", "度")
    )


def parse_choice_map(recall: str) -> dict[str, str]:
    """从题干解析 A. xxx / B. xxx 选项。"""
    mapping: dict[str, str] = {}
    for line in recall.splitlines():
        line = line.strip()
        m = re.match(r"^([A-Da-d])[\.．、\)]\s*(.+)$", line)
        if m:
            mapping[m.group(1).upper()] = m.group(2).strip()
    return mapping


def heuristic_score(expect_kw: str, actual_res: str, recall: str = "") -> float | None:
    """可确定的启发式：命中=1，空召回=0；其余交给 LLM。"""
    if not actual_res.strip():
        return 0.0
    e = normalize(expect_kw)
    a = normalize(actual_res)
    if not e:
        return None

    # ScriptMem 字母答案：映射到选项正文再匹配
    if e.isascii() and e.isalpha() and 1 <= len(e) <= 4:
        choice_map = parse_choice_map(recall)
        if choice_map:
            # 单选或多选字母串，如 A / ABC
            letters = list(expect_kw.upper())
            if all(ch in choice_map for ch in letters):
                hit = 0
                for ch in letters:
                    opt = normalize(choice_map[ch])
                    if opt and opt in a:
                        hit += 1
                if hit == len(letters):
                    return 1.0
                if hit > 0:
                    return 0.5
                return 0.0
        return None

    if e in a:
        return 1.0
    return None


JUDGE_SYSTEM = """你是记忆检索评测裁判。根据召回结果 actual_res，判断是否足以支撑 expect_kw 回答 recall。
打分只能是 0 / 0.5 / 1：
- 1：actual_res 明确包含 expect_kw，或语义等价且可直接据此作答
- 0.5：有相关线索但不完整、含糊，或被干扰信息淹没导致不完全确定
- 0：未提及、明显矛盾、或内容无关
选择题（expect_kw 为 A/B/C/ABC 等）：若 actual_res 含正确选项对应事实则 1，否则 0 或 0.5。
只输出 JSON 数组，不要 markdown，不要解释外的文字。格式：
[{"idx":0,"score":1,"reason":"简短原因"}]"""


def chat_complete(messages: list[dict[str, str]], max_tokens: int = 2048) -> str:
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("missing LLM_API_KEY for scoring")
    url = os.environ.get(
        "LLM_CHAT_URL", "https://qianfan.baidubce.com/v2/chat/completions"
    )
    model = os.environ.get("LLM_CHAT_MODEL", "qwen3.5-27b")
    body = {
        "messages": messages,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "top_p": 0.7,
        "reasoning_split": True,
    }
    data = http_json(
        "POST",
        url,
        body=body,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=180,
    )
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"empty chat choices: {data}")
    content = (choices[0].get("message") or {}).get("content") or ""
    if not str(content).strip():
        raise RuntimeError(f"empty chat content: {data}")
    return str(content).strip()


def extract_json_array(text: str) -> list[Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < start:
        raise ValueError(f"no json array in LLM response: {text[:400]}")
    return json.loads(text[start : end + 1])


def score_batch_llm(batch: list[dict[str, Any]]) -> dict[int, tuple[float, str]]:
    payload = []
    for i, item in enumerate(batch):
        payload.append(
            {
                "idx": i,
                "recall": item["recall"],
                "expect_kw": item["expect_kw"],
                "actual_res": item["actual_res"][:1200],
            }
        )
    user = "请对以下样本打分，返回与 idx 对应的 JSON 数组：\n" + json.dumps(
        payload, ensure_ascii=False
    )
    raw = chat_complete(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ]
    )
    arr = extract_json_array(raw)
    out: dict[int, tuple[float, str]] = {}
    for row in arr:
        idx = int(row["idx"])
        score = float(row["score"])
        if score not in (0.0, 0.5, 1.0):
            # clamp to nearest bucket
            score = min([0.0, 0.5, 1.0], key=lambda x: abs(x - score))
        reason = str(row.get("reason") or "").strip()
        out[idx] = (score, reason)
    return out


def write_memories(
    base: str,
    samples: list[dict[str, Any]],
    namespace: str,
) -> int:
    n = 0
    total_inputs = sum(len(s["input"]) for s in samples)
    done = 0
    for s in samples:
        for content in s["input"]:
            content = str(content).strip()
            if not content:
                continue
            http_json(
                "POST",
                f"{base}/api/memories",
                body={
                    "content": content,
                    "namespace": namespace,
                    "session_id": s["id"],
                    "actor": "eval",
                    "importance": 0.7,
                    "confidence": 0.9,
                    "skip_consolidation": True,
                },
                timeout=180,
            )
            n += 1
            done += 1
            if done % 20 == 0 or done == total_inputs:
                print(f"  write {done}/{total_inputs}", flush=True)
    return n


def run_recalls(
    base: str,
    samples: list[dict[str, Any]],
    namespace: str,
    top_k: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for i, s in enumerate(samples):
        recall = s["recall"]
        if not recall:
            actual = ""
        else:
            raw = http_json(
                "POST",
                f"{base}/api/recall",
                body={
                    "text": recall,
                    "namespace": namespace,
                    "top_k": top_k,
                    "kinds": [],
                },
                timeout=180,
            )
            actual = format_actual(raw or {}, top_k)
        results.append(
            {
                "id": s["id"],
                "dataset": s["dataset"],
                "input": s["input"],
                "recall": recall,
                "expect_kw": s["expect_kw"],
                "actual_res": actual,
            }
        )
        if (i + 1) % 20 == 0 or (i + 1) == len(samples):
            print(f"  recall {i + 1}/{len(samples)}", flush=True)
    return results


def score_results(
    rows: list[dict[str, Any]],
    batch_size: int,
    use_llm: bool,
) -> list[dict[str, Any]]:
    pending: list[tuple[int, dict[str, Any]]] = []
    for i, row in enumerate(rows):
        hs = heuristic_score(
            row["expect_kw"], row["actual_res"], row.get("recall") or ""
        )
        if hs is not None:
            row["score"] = hs
            exp = row["expect_kw"]
            if exp.isascii() and exp.isalpha() and 1 <= len(exp) <= 4:
                row["reason"] = f"heuristic:choice:{hs}"
            elif hs == 0.0 and not str(row.get("actual_res") or "").strip():
                row["reason"] = "heuristic:empty"
            elif hs == 1.0:
                row["reason"] = "heuristic:substring"
            else:
                row["reason"] = f"heuristic:{hs}"
            row["judge"] = "heuristic"
        else:
            row["score"] = None
            row["reason"] = None
            row["judge"] = "pending"
            pending.append((i, row))

    if not use_llm:
        for _, row in pending:
            row["score"] = 0.0
            row["reason"] = "llm disabled; unmarked"
            row["judge"] = "skipped"
        return rows

    print(
        f"scoring: heuristic={len(rows) - len(pending)} llm_pending={len(pending)} "
        f"batch_size={batch_size}",
        flush=True,
    )
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        batch_rows = [r for _, r in chunk]
        judged: dict[int, tuple[float, str]] = {}
        try:
            judged = score_batch_llm(batch_rows)
        except Exception as e:  # noqa: BLE001
            print(f"  llm batch failed: {e}; retry once", flush=True)
            time.sleep(2)
            try:
                judged = score_batch_llm(batch_rows)
            except Exception as e2:  # noqa: BLE001
                print(f"  llm batch failed again: {e2}; fallback score=0", flush=True)
                judged = {
                    local_i: (0.0, f"llm_error: {e2}")
                    for local_i, _ in enumerate(batch_rows)
                }
        for local_i, (global_i, row) in enumerate(chunk):
            if local_i in judged:
                score, reason = judged[local_i]
                row["score"] = score
                row["reason"] = reason
                row["judge"] = (
                    "llm_error" if str(reason).startswith("llm_error") else "llm"
                )
            else:
                row["score"] = 0.0
                row["reason"] = "llm missing idx"
                row["judge"] = "llm_error"
            rows[global_i] = row
        print(
            f"  scored llm {min(start + batch_size, len(pending))}/{len(pending)}",
            flush=True,
        )
        time.sleep(0.3)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(r["score"]) for r in rows if r.get("score") is not None]
    by_ds: dict[str, list[float]] = {}
    by_bucket = {"0": 0, "0.5": 0, "1": 0}
    for r in rows:
        sc = float(r.get("score") or 0.0)
        by_ds.setdefault(r["dataset"], []).append(sc)
        key = "1" if sc >= 1.0 else ("0.5" if sc >= 0.5 else "0")
        by_bucket[key] += 1
    overall = sum(scores) / len(scores) if scores else 0.0
    per_dataset = {
        k: {
            "n": len(v),
            "avg": round(sum(v) / len(v), 4) if v else 0.0,
            "sum": round(sum(v), 4),
        }
        for k, v in sorted(by_ds.items())
    }
    return {
        "n": len(rows),
        "avg_score": round(overall, 4),
        "sum_score": round(sum(scores), 4),
        "max_score": len(rows),
        "pct": round(100.0 * overall, 2),
        "buckets": by_bucket,
        "per_dataset": per_dataset,
        "judge_counts": {
            "heuristic": sum(1 for r in rows if r.get("judge") == "heuristic"),
            "llm": sum(1 for r in rows if r.get("judge") == "llm"),
            "other": sum(1 for r in rows if r.get("judge") not in ("heuristic", "llm")),
        },
    }


def write_report_md(
    path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    lines = [
        "# Memora Eval Report",
        "",
        f"- samples: **{summary['n']}**",
        f"- avg score: **{summary['avg_score']}** / 1.0 ({summary['pct']}%)",
        f"- sum: {summary['sum_score']} / {summary['max_score']}",
        f"- buckets: {summary['buckets']}",
        f"- judges: {summary['judge_counts']}",
        "",
        "## Per dataset",
        "",
        "| dataset | n | avg |",
        "|---|---:|---:|",
    ]
    for name, st in summary["per_dataset"].items():
        lines.append(f"| {name} | {st['n']} | {st['avg']} |")
    lines.extend(["", "## Failures (score=0, first 30)", ""])
    fails = [r for r in rows if float(r.get("score") or 0) <= 0.0]
    for r in fails[:30]:
        lines.append(
            f"- `{r['id']}` expect=`{r['expect_kw']}` "
            f"reason={r.get('reason')} actual={str(r.get('actual_res') or '')[:120]!r}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Memora memory eval")
    p.add_argument(
        "--base-url", default=os.environ.get("MEMORA_EVAL_URL", "http://127.0.0.1:8790")
    )
    p.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--datasets", nargs="*", default=None, help="dataset file names")
    p.add_argument("--limit", type=int, default=None, help="max samples")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--namespace", default="eval")
    p.add_argument("--skip-write", action="store_true")
    p.add_argument("--skip-recall", action="store_true")
    p.add_argument(
        "--score-only", action="store_true", help="only score existing result.json"
    )
    p.add_argument("--no-llm", action="store_true", help="skip LLM judge")
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
    else:
        files = list_dataset_files(args.eval_dir, args.datasets)
        samples = load_samples(files, args.limit)
        print(
            f"loaded samples={len(samples)} datasets={len(files)} "
            f"inputs={sum(len(s['input']) for s in samples)}",
            flush=True,
        )
        if not args.skip_write:
            print("phase: write", flush=True)
            n = write_memories(args.base_url, samples, args.namespace)
            print(f"wrote memories={n}", flush=True)
        if args.skip_recall and result_path.is_file():
            rows = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            print("phase: recall", flush=True)
            rows = run_recalls(args.base_url, samples, args.namespace, args.top_k)
            result_path.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"wrote {result_path}", flush=True)

    print("phase: score", flush=True)
    rows = score_results(rows, args.batch_size, use_llm=not args.no_llm)
    result_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = summarize(rows)
    report = {
        "summary": summary,
        "config": {
            "base_url": args.base_url,
            "namespace": args.namespace,
            "top_k": args.top_k,
            "batch_size": args.batch_size,
            "limit": args.limit,
            "datasets": args.datasets,
            "llm_model": os.environ.get("LLM_CHAT_MODEL"),
        },
    }
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report_md(report_md, summary, rows)
    print(
        f"DONE avg={summary['avg_score']} ({summary['pct']}%) "
        f"n={summary['n']} report={report_md}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
