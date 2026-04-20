"""Run eval cases concurrently, print a pass/fail table, exit non-zero on any miss.
Also mirrors stdout to a timestamped file under evals/runs/.

Usage:
    uv run python -m evals.eval              # run all cases
    uv run python -m evals.eval Q1 H3        # run only cases whose id starts with Q1 or H3
    uv run python -m evals.eval --show       # print full answers for failing cases
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from evals.cases import CASES
from src.agent import build_agent

load_dotenv()

MAX_WORKERS = 7
RUNS_DIR = Path(__file__).parent / "runs"


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


@dataclass
class ToolCall:
    name: str
    args: dict


@dataclass
class Result:
    id: str
    passed: bool
    duration_s: float
    missing: list[str] = field(default_factory=list)
    leaked: list[str] = field(default_factory=list)
    answer: str = ""
    error: str = ""
    node_times: dict[str, float] = field(default_factory=dict)
    tool_calls: list[ToolCall] = field(default_factory=list)


def _fmt_args(args: dict, max_len: int = 60) -> str:
    parts = []
    for k, v in args.items():
        if v in ("", None):
            continue
        s = str(v).replace("\n", " ")
        if len(s) > 40:
            s = s[:37] + "…"
        parts.append(f"{k}={s!r}" if isinstance(v, str) else f"{k}={s}")
    out = ", ".join(parts)
    return out if len(out) <= max_len else out[: max_len - 1] + "…"


def _normalize(s: str) -> str:
    # Fold unicode dashes so "7–10" / "7—10" match "7-10".
    return s.lower().replace("\u2013", "-").replace("\u2014", "-").replace("\u2012", "-")


def score(answer: str, case: dict) -> tuple[list[str], list[str]]:
    """must_include entries may be str (required) or tuple/list (any-of)."""
    low = _normalize(answer)
    missing = []
    for kw in case.get("must_include", []):
        if isinstance(kw, (list, tuple)):
            if not any(_normalize(k) in low for k in kw):
                missing.append(" | ".join(kw))
        elif _normalize(kw) not in low:
            missing.append(kw)
    leaked = [kw for kw in case.get("must_not_include", []) if _normalize(kw) in low]
    return missing, leaked


def run_case(case: dict) -> Result:
    start = time.time()
    node_times: dict[str, float] = {}
    tool_calls: list[ToolCall] = []
    answer = ""
    try:
        agent = build_agent()
        last_t = time.time()
        for step in agent.stream(
            {"messages": [{"role": "user", "content": case["question"]}]},
            config={"configurable": {"thread_id": case["id"]}},
            stream_mode="updates",
        ):
            now = time.time()
            for node, update in step.items():
                node_times[node] = node_times.get(node, 0.0) + (now - last_t)
                msgs = (update or {}).get("messages") or []
                for m in msgs:
                    for tc in getattr(m, "tool_calls", None) or []:
                        tool_calls.append(ToolCall(name=tc["name"], args=tc.get("args", {}) or {}))
                    if node == "answer":
                        answer = getattr(m, "content", "") or ""
            last_t = now

        missing, leaked = score(answer, case)
        return Result(
            id=case["id"],
            passed=not missing and not leaked,
            duration_s=time.time() - start,
            missing=missing,
            leaked=leaked,
            answer=answer,
            node_times=node_times,
            tool_calls=tool_calls,
        )
    except Exception as e:
        return Result(
            id=case["id"],
            passed=False,
            duration_s=time.time() - start,
            error=f"{type(e).__name__}: {e}",
            node_times=node_times,
            tool_calls=tool_calls,
        )


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    show_full = "--show" in sys.argv[1:]
    cases = [c for c in CASES if not args or any(c["id"].startswith(a) for a in args)]
    if not cases:
        print(f"no cases match {args}")
        sys.exit(2)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUNS_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    log_file = open(log_path, "w")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    print(f"running {len(cases)} case(s) with {MAX_WORKERS} workers (log: {log_path.name})\n")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_case, c): c for c in cases}
        results = []
        for fut in futures:
            r = fut.result()
            status = "PASS" if r.passed else "FAIL"
            print(f"  {status}  {r.id}  ({r.duration_s:.1f}s, {len(r.tool_calls)} tool calls)")
            results.append(r)

    order = {c["id"]: i for i, c in enumerate(cases)}
    results.sort(key=lambda r: order[r.id])

    print("\n" + "=" * 78)
    width = max(len(r.id) for r in results)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        timings = " ".join(f"{n}={t:.1f}s" for n, t in r.node_times.items())
        print(f"{r.id:<{width}}  {status}  {r.duration_s:>5.1f}s  [{timings}]")
        if r.error:
            print(f"  ERROR: {r.error}")
        if r.missing:
            print(f"  missing: {', '.join(r.missing)}")
        if r.leaked:
            print(f"  leaked:  {', '.join(r.leaked)}")
        if r.tool_calls:
            print(f"  tools ({len(r.tool_calls)}):")
            for tc in r.tool_calls:
                print(f"    - {tc.name}({_fmt_args(tc.args)})")
        if show_full and not r.passed and r.answer:
            print(f"  answer:  {r.answer[:800]}{'…' if len(r.answer) > 800 else ''}")
    print("=" * 78)

    passed = sum(1 for r in results if r.passed)
    total_tools = sum(len(r.tool_calls) for r in results)
    tool_name_counts: dict[str, int] = {}
    for r in results:
        for tc in r.tool_calls:
            tool_name_counts[tc.name] = tool_name_counts.get(tc.name, 0) + 1
    node_totals: dict[str, float] = {}
    for r in results:
        for n, t in r.node_times.items():
            node_totals[n] = node_totals.get(n, 0.0) + t
    avg_dur = sum(r.duration_s for r in results) / len(results)
    print(
        f"{passed}/{len(results)} passed · wall {time.time() - t0:.1f}s · "
        f"avg/case {avg_dur:.1f}s · {total_tools} tool calls "
        f"({', '.join(f'{n}={c}' for n, c in sorted(tool_name_counts.items()))})"
    )
    if node_totals:
        print("node time totals: " + ", ".join(f"{n}={t:.1f}s" for n, t in node_totals.items()))
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
