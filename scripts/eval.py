"""Run eval cases concurrently, print a pass/fail table, exit non-zero on any miss.

Usage:
    uv run python -m scripts.eval              # run all cases
    uv run python -m scripts.eval Q1 H3        # run only cases whose id starts with Q1 or H3
    uv run python -m scripts.eval --show       # print full answers for failing cases
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from dotenv import load_dotenv

from evals.cases import CASES
from src.agent import build_agent

load_dotenv()

MAX_WORKERS = 7
AGENT = build_agent()


@dataclass
class Result:
    id: str
    passed: bool
    duration_s: float
    missing: list[str] = field(default_factory=list)
    leaked: list[str] = field(default_factory=list)
    answer: str = ""
    error: str = ""


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
    try:
        out = AGENT.invoke(
            {"messages": [{"role": "user", "content": case["question"]}]},
            config={"configurable": {"thread_id": case["id"]}},
        )
        answer = out["messages"][-1].content
        missing, leaked = score(answer, case)
        return Result(
            id=case["id"],
            passed=not missing and not leaked,
            duration_s=time.time() - start,
            missing=missing,
            leaked=leaked,
            answer=answer,
        )
    except Exception as e:
        return Result(
            id=case["id"],
            passed=False,
            duration_s=time.time() - start,
            error=f"{type(e).__name__}: {e}",
        )


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    show_full = "--show" in sys.argv[1:]
    cases = [c for c in CASES if not args or any(c["id"].startswith(a) for a in args)]
    if not cases:
        print(f"no cases match {args}")
        sys.exit(2)

    print(f"running {len(cases)} case(s) with {MAX_WORKERS} workers\n")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_case, c): c for c in cases}
        results = []
        for fut in futures:
            r = fut.result()
            status = "PASS" if r.passed else "FAIL"
            print(f"  {status}  {r.id}  ({r.duration_s:.1f}s)")
            results.append(r)

    # sort back into case order
    order = {c["id"]: i for i, c in enumerate(cases)}
    results.sort(key=lambda r: order[r.id])

    print("\n" + "=" * 72)
    width = max(len(r.id) for r in results)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"{r.id:<{width}}  {status}  {r.duration_s:>5.1f}s")
        if r.error:
            print(f"  ERROR: {r.error}")
        if r.missing:
            print(f"  missing: {', '.join(r.missing)}")
        if r.leaked:
            print(f"  leaked:  {', '.join(r.leaked)}")
        if show_full and not r.passed and r.answer:
            print(f"  answer:  {r.answer[:800]}{'…' if len(r.answer) > 800 else ''}")
    print("=" * 72)
    passed = sum(1 for r in results if r.passed)
    print(f"{passed}/{len(results)} passed · wall time {time.time() - t0:.1f}s")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
