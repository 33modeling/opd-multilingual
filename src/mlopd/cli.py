"""Command line: python -m mlopd <command>.

  data build            download the pinned datasets and write data/processed/{processed,manifest}.json
  queue dry-run         list every task of the run grid with its state and estimated hours
  queue run             run tasks one at a time (finished tasks are skipped; safe to restart)
  gates                 print the stage-0 decision files that exist
  analyze               write runs/analysis/verdicts.json from finished runs (original verdicts, block C
                        confirmation verdicts with Holm over the C family, block X record)
  prompts               print every prompt text (for the paper appendix)
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mlopd")
    parser.add_argument("--config", default=None, help="config JSON merged over configs/default.json")
    parser.add_argument("--root", default=".", help="repository root (paths in the config are relative to it)")
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("data")
    data.add_argument("action", choices=["build"])
    data.add_argument("--cache", default=None)
    data.add_argument("--out", default="data/processed")
    queue = sub.add_parser("queue")
    queue.add_argument("action", choices=["dry-run", "run", "worker", "status", "retry-failed"],
                       help="run: one process, stops at the first error; worker: leased tasks shared with other "
                            "workers (one per GPU), failures recorded and retried; status; retry-failed")
    queue.add_argument("--wait", action="store_true", help="worker: keep polling while other workers run tasks")
    queue.add_argument("--max-failures", type=int, default=2)
    queue.add_argument("--max-tasks", type=int, default=None)
    queue.add_argument("--stop-file", default="runs/STOP")
    queue.add_argument("--all", action="store_true", help="dry run: print every task line")
    sub.add_parser("gates")
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--out", default=None)
    sub.add_parser("prompts")
    args = parser.parse_args(argv)

    from .experiment import Context, load_config
    cfg = load_config(args.config)
    if args.command == "prompts":
        from .prompts import all_prompt_texts
        print(json.dumps(all_prompt_texts(), indent=1, ensure_ascii=False))
        return
    if args.command == "data":
        from pathlib import Path
        from .data import build
        root = Path(args.root)
        _, manifest = build(root / (args.cache or cfg["paths"]["cache"]), root / args.out,
                            pool_size=cfg["splits"]["pool_size"])
        print(json.dumps({k: manifest[k] for k in ("parallel", "pool", "eval_counts", "processed_sha256")}, indent=1))
        return
    ctx = Context(cfg, root=args.root)
    if args.command == "queue":
        from .experiment import dry_run, run_queue
        if args.action == "dry-run":
            result = dry_run(ctx)
            lines = result["lines"]
            shown = lines if args.all else [l for l in lines if not l.startswith("done")][:60]
            print("\n".join(shown))
            if not args.all and len(lines) > len(shown):
                print(f"... {len(lines) - len(shown)} more lines (use --all)")
            print("\nblock   tasks   hours")
            for block, (count, hours) in result["blocks"].items():
                print(f"{block:7s} {count:5d} {hours:7.1f}")
            for name, what, hours in result["pending_blocks"]:
                print(f"{name:7s} pending gates: {what}, about {hours:.1f}h")
            print(f"\nlisted {len(result['tasks'])} tasks, {result['total_hours']:.1f}h, "
                  f"remaining {result['remaining_hours']:.1f}h (estimates from design §8; calendar 1.5-2x)")
        elif args.action == "run":
            run_queue(ctx, max_tasks=args.max_tasks, stop_file=args.stop_file,
                      log=lambda m: print(m, flush=True))
        else:
            import time
            from .experiment import status
            from .lease import Leases
            leases = Leases(ctx.runs, max_failures=args.max_failures)
            if args.action == "worker":
                import signal

                def interrupt(signum, frame):  # SIGTERM -> KeyboardInterrupt, so the lease is released
                    raise KeyboardInterrupt(f"signal {signum}")
                signal.signal(signal.SIGTERM, interrupt)
                tag = leases.worker
                run_queue(ctx, max_tasks=args.max_tasks, stop_file=args.stop_file, leases=leases, wait=args.wait,
                          log=lambda m: print(f"{time.strftime('%H:%M:%S')} [{tag}] {m}", flush=True))
            elif args.action == "retry-failed":
                print(f"cleared {leases.clear_failures()} failure record(s)")
            else:
                from .experiment import format_status
                print(format_status(status(ctx, leases)))
        return
    if args.command == "gates":
        from pathlib import Path
        for path in sorted(Path(ctx.runs).glob("*/gates/*/COMPLETE.json")):
            print(path)
            print(json.dumps(json.loads(path.read_text()), indent=1, ensure_ascii=False)[:4000])
        return
    if args.command == "analyze":
        from .analysis import write_verdicts
        path, result = write_verdicts(ctx, args.out)
        print(path)
        for key, value in result.items():
            if isinstance(value, dict) and "status" in value:
                p = value.get("test", {}).get("p") if isinstance(value.get("test"), dict) else None
                holm = value.get("p_holm_confirm")
                print(f"{key:8s} {value['status']}" + (f"  p={p:.4g}" if p is not None else "")
                      + (f"  Holm(C)={holm:.4g}" if holm is not None else ""))
        for l, v in (result.get("X", {}).get("per_language") or {}).items():
            print(f"X {l:5s}  D {v['D_mean_pp']:+.2f}pp  G0 OPD {v['G0_pp']['opd']:+.2f} RL {v['G0_pp']['rl']:+.2f}  "
                  f"start {100 * v['start_acc']:.1f}  strict start/OPD/RL {100 * v['strict']['start']:.1f}/"
                  f"{100 * v['strict']['opd']:.1f}/{100 * v['strict']['rl']:.1f}  collapses OPD {v['collapses']['opd']} "
                  f"RL {v['collapses']['rl']}")
        return


if __name__ == "__main__":
    main(sys.argv[1:])
