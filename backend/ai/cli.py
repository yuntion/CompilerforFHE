"""CLI for sealed tasks, deterministic baselines and model proposals."""
from __future__ import annotations

from pathlib import Path

from .driver import baseline_candidate, run_candidates, write_json
from .protocol import read_json
from .provider import OfflineProvider, OpenAICompatibleProvider
from .task import prepare


def add_parser(commands):
    parser = commands.add_parser("ai", help="prepare and validate restricted AI backend schedules")
    actions = parser.add_subparsers(dest="ai_action", required=True)
    preparer = actions.add_parser("prepare", help="export a public, hash-sealed scheduling task")
    preparer.add_argument("artifact", type=Path)
    preparer.add_argument("--output", type=Path, required=True)
    preparer.add_argument("--documentation-root", type=Path,
                          help="optional external FHE-Coder summaries root; default: repository backend/api mappings")
    preparer.add_argument("--atol", type=float, default=0.005)
    preparer.add_argument("--rtol", type=float, default=0.005)
    preparer.add_argument("--input-bound", type=float, default=0.5, help="public absolute input bound")
    preparer.add_argument("--comparison-scale", type=float, default=1024.0)
    preparer.add_argument("--insecure-demo", action="store_true", help="explicit OpenFHE TOY development checks; never policy passed")
    preparer.add_argument("--seed", type=int, default=17)
    preparer.add_argument("--max-attempts", type=int, default=10, help="total generations, including the first, at most 10")
    preparer.add_argument("--repetitions", type=int, default=3)
    preparer.add_argument("--warmups", type=int, default=1)
    baseline = actions.add_parser("baseline", help="make a deterministic candidate, without calling a model")
    baseline.add_argument("task", type=Path)
    baseline.add_argument("--strategy", choices=("reference", "depth-first"), default="depth-first")
    baseline.add_argument("--output", type=Path, required=True)
    runner = actions.add_parser("run", help="generate, compile and independently validate candidates")
    runner.add_argument("task", type=Path)
    runner.add_argument("--artifact", type=Path, required=True)
    runner.add_argument("--output", type=Path, required=True)
    provider = runner.add_mutually_exclusive_group(required=True)
    provider.add_argument("--candidate", type=Path, help="one schedule JSON proposal")
    provider.add_argument("--responses", type=Path, help="offline JSON response queue, including repair responses")
    provider.add_argument("--model", help="explicitly opt in to a configured chat/completions model")
    runner.add_argument("--api-base", help="OpenAI-compatible base URL, e.g. https://api.openai.com/v1")
    runner.add_argument("--api-key-env", default="OPENAI_API_KEY", help="name of the API credential environment variable")
    runner.add_argument("--timeout", type=float, default=60, help="model HTTP timeout in seconds")
    return parser


def run(args, config):
    if args.ai_action == "prepare":
        task = prepare(args.artifact, args.output, documentation_root=args.documentation_root,
            atol=args.atol, rtol=args.rtol, input_bound=args.input_bound, seed=args.seed,
            comparison_scale=args.comparison_scale, insecure_demo=args.insecure_demo,
            max_attempts=args.max_attempts, repetitions=args.repetitions, warmups=args.warmups)
        return {"status": task["status"], "task_hash": task["task_hash"], "output": str(args.output.resolve())}, 0
    if args.ai_action == "baseline":
        if args.output.exists():
            raise ValueError("candidate output already exists; choose a new file")
        candidate = baseline_candidate(args.task, args.strategy)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, candidate)
        return {"status": "generated_unvalidated", "output": str(args.output.resolve()), "model_called": False}, 0
    if args.model:
        if not args.api_base:
            raise ValueError("--model requires explicit --api-base")
        provider = OpenAICompatibleProvider(model=args.model, base_url=args.api_base,
                                           api_key_env=args.api_key_env, timeout=args.timeout)
    elif args.candidate:
        provider = OfflineProvider([read_json(args.candidate)])
    else:
        provider = OfflineProvider(args.responses)
    report = run_candidates(args.task, args.artifact, args.output, provider, config)
    return report, 0 if report["status"] in ("validated", "functional_only", "no_candidate") else 1
