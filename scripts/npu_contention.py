"""Contention test: how much an agent-style model workload slows your own work, NPU vs CPU.

usage: contention.py ENDPOINT MODEL --mode npu|cpu|none [--seconds 120] [--fork-dir DIR]
                     [--job "CMD"] [--out FILE] [--label TEXT]

For --seconds, concurrently:
  * agent loop (unless --mode none): sequential /api/chat requests with a long system prompt
    (tool definitions), coding tasks in rotation, up to 200 new tokens each, a 0.3 s pause
    between turns (a tool "running"); --mode cpu sends num_gpu 0
  * interactive probe: a fixed single-thread work unit every 0.5 s; median and p95 duration
    stand for typing/UI latency
  * batch probe: a fixed all-cores work unit (one process per core), back to back; median
    duration stands for a build or a test run
  * optional --job: an external command run back to back, wall time per run
Reports the model's decode and prompt speed, requests completed, and the CPU cores the fork's
processes used (TotalProcessorTime of processes under --fork-dir, via PowerShell).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import statistics
import subprocess
import threading
import time
import urllib.request

TASKS = [
    "Write a Python function that parses an ISO 8601 duration string into seconds, with tests.",
    "Explain why this Go code races and fix it: a map written from two goroutines without a mutex.",
    "Refactor a 300-line PowerShell script that builds a project into functions; outline the plan first.",
    "Given a CSV of orders with columns id,date,amount, write SQL for monthly totals and the top customer.",
    "Review this snippet for bugs: for i in range(len(xs)-1): if xs[i] > xs[i+1]: xs[i], xs[i+1] = xs[i+1], xs[i]",
    "Write a bash one-liner that finds files over 100 MB changed in the last day, then explain each flag.",
    "Design the data model for a to-do app with projects, tags and recurring tasks; list the tables.",
    "Convert this JavaScript callback code to async/await and handle errors: fs.readFile(p, cb).",
    "What does git rebase --onto do? Give an example with three branches.",
    "Write a regex that matches semantic versions with optional pre-release tags, and test cases.",
]
TOOLS = [
    ("read_file", "Read a file from the workspace.", '{"path": string}'),
    ("write_file", "Create or overwrite a file.", '{"path": string, "content": string}'),
    ("list_dir", "List a directory.", '{"path": string, "recursive": boolean}'),
    ("search", "Search the workspace with a regular expression.", '{"pattern": string, "glob": string}'),
    ("run", "Run a shell command and return stdout, stderr and the exit code.", '{"command": string, "timeout_s": integer}'),
    ("git", "Run a git subcommand in the repository.", '{"args": [string]}'),
    ("http_get", "Fetch a URL.", '{"url": string}'),
    ("ask_user", "Ask the developer a question and wait for the answer.", '{"question": string}'),
]
SYSTEM = (
    "You are a coding agent working inside a developer's editor. Think step by step, be concise, "
    "and use tools when needed. Available tools (call with a JSON object on its own line):\n"
    + "\n".join(f"- {name}: {desc} Parameters: {params}" for name, desc, params in TOOLS)
    + "\nRules: never delete files without asking; keep diffs minimal; run the tests after a change; "
    "report what you changed and why. Workspace: a mid-sized Python and Go monorepo with CI on GitHub "
    "Actions, a Makefile, ruff and mypy configured, and a docs folder. The developer prefers short "
    "answers with code blocks, and explanations only where the reasoning is not obvious from the code."
) * 2  # about 700 tokens


def work1() -> float:
    """One fixed single-thread unit (hashing plus a Python loop); returns seconds."""
    data = b"x" * (4 << 20)
    t = time.perf_counter()
    h = hashlib.sha256()
    for _ in range(24):
        h.update(data)
    _ = sum(i * i for i in range(300_000))
    return time.perf_counter() - t


def _worker(_: int) -> float:
    return sum(work1() for _ in range(2))


def cpu_seconds(fork_dir: str) -> float:
    quoted = fork_dir.replace("'", "''")
    script = (
        "$s = 0.0; Get-Process | Where-Object { $_.Path -and $_.Path.StartsWith('" + quoted
        + "', [System.StringComparison]::OrdinalIgnoreCase) } | ForEach-Object { $s += $_.TotalProcessorTime.TotalSeconds }; "
        "[string]::Format([System.Globalization.CultureInfo]::InvariantCulture, '{0:F3}', $s)"
    )
    out = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True, timeout=60)
    return float(out.stdout.strip() or 0)


def chat(endpoint: str, model: str, mode: str, task: str) -> dict:
    options: dict = {"num_predict": 200, "temperature": 0.0}
    if mode == "cpu":
        options["num_gpu"] = 0
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}],
        "stream": False,
        "options": options,
        "keep_alive": "10m",
    }
    req = urllib.request.Request(endpoint + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("endpoint")
    ap.add_argument("model")
    ap.add_argument("--mode", choices=["npu", "cpu", "none"], required=True)
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--fork-dir", default="")
    ap.add_argument("--job", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    stop = time.monotonic() + a.seconds
    lat: list[float] = []
    batch: list[float] = []
    jobs: list[float] = []
    reqs: list[dict] = []
    errors: list[str] = []

    def agent() -> None:
        i = 0
        while time.monotonic() < stop:
            t = time.perf_counter()
            try:
                r = chat(a.endpoint, a.model, a.mode, TASKS[i % len(TASKS)])
                reqs.append({"wall": time.perf_counter() - t, "eval": r.get("eval_count", 0),
                             "eval_ns": r.get("eval_duration", 0), "prompt": r.get("prompt_eval_count", 0),
                             "prompt_ns": r.get("prompt_eval_duration", 0)})
            except Exception as e:  # noqa: BLE001
                errors.append(f"{type(e).__name__}: {e}"[:200])
                time.sleep(2)
            i += 1
            time.sleep(0.3)

    def interactive() -> None:
        while time.monotonic() < stop:
            lat.append(work1())
            time.sleep(0.5)

    def batches() -> None:
        n = os.cpu_count() or 4
        with mp.Pool(n) as pool:
            while time.monotonic() < stop:
                t = time.perf_counter()
                pool.map(_worker, range(n))
                batch.append(time.perf_counter() - t)
                time.sleep(1.0)

    def job() -> None:
        while time.monotonic() < stop:
            t = time.perf_counter()
            subprocess.run(a.job, shell=True, capture_output=True)
            jobs.append(time.perf_counter() - t)

    if a.mode != "none":
        chat(a.endpoint, a.model, a.mode, "Say ready.")  # load the model before timing starts
    cpu0 = cpu_seconds(a.fork_dir) if a.fork_dir else 0.0
    t0 = time.perf_counter()
    threads = [threading.Thread(target=interactive, daemon=True), threading.Thread(target=batches, daemon=True)]
    if a.mode != "none":
        threads.append(threading.Thread(target=agent, daemon=True))
    if a.job:
        threads.append(threading.Thread(target=job, daemon=True))
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    elapsed = time.perf_counter() - t0
    cpu1 = cpu_seconds(a.fork_dir) if a.fork_dir else 0.0

    evals = sum(r["eval"] for r in reqs)
    eval_s = sum(r["eval_ns"] for r in reqs) / 1e9
    prompts = sum(r["prompt"] for r in reqs)
    prompt_s = sum(r["prompt_ns"] for r in reqs) / 1e9
    result = {
        "label": a.label or a.mode, "mode": a.mode, "model": a.model, "seconds": round(elapsed, 1),
        "interactive_ms": {"median": round(1000 * statistics.median(lat), 1) if lat else None,
                           "p95": round(1000 * pct(lat, 0.95), 1), "n": len(lat)},
        "batch_s": {"median": round(statistics.median(batch), 2) if batch else None,
                    "p95": round(pct(batch, 0.95), 2), "n": len(batch)},
        "job_s": {"median": round(statistics.median(jobs), 1) if jobs else None, "n": len(jobs)},
        "model_requests": len(reqs), "errors": errors[:3],
        "decode_tok_s": round(evals / eval_s, 1) if eval_s else None,
        "prompt_tok_s": round(prompts / prompt_s, 1) if prompt_s else None,
        "tokens_generated": evals,
        "fork_cores": round((cpu1 - cpu0) / elapsed, 2) if a.fork_dir else None,
    }
    line = json.dumps(result)
    print(line)
    if a.out:
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    mp.freeze_support()
    main()
