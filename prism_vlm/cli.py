"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

CLI entrypoint: ``prism-vlm {build, run, score}``.

``build`` samples the source items and synthesises the per-item artifacts for
the five judge-built axes (h, mq, c, vis, sc). ``run`` evaluates a model on the
resulting benchmark and grades each axis. q needs no artifact, and if samples
its constraints deterministically at run time. ``score`` aggregates the graded
items into per-axis means and PScore.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

AXES = ("q", "IF", "h", "mq", "sc", "c", "vis")
# Mirrors scoring.aggregate.AGGREGATORS; pinned by a test so the two cannot drift.
AGGREGATOR_NAMES = ("harmonic", "geometric", "arithmetic", "min")


def _add_subset_args(parser: argparse.ArgumentParser) -> None:
    """Debug-run filters. Pass the same ones to `build` and `run`."""
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Only these benchmark ids, e.g. --benchmarks mmstar okvqa.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Only these domain names, e.g. --domains general_vqa.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the item pool at N (lowest item ids, so build and run agree).",
    )


def _add_terms_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--accept-dataset-terms",
        action="store_true",
        help="Accept the source benchmarks' terms (see NOTICE) without the prompt.",
    )


def _confirm_dataset_terms(cfg, domains_cfg, accepted: bool) -> bool:  # noqa: ANN001
    """Show the source terms once per cache before anything is downloaded."""
    from .data.terms import acceptance_marker, terms_text

    marker = acceptance_marker(cfg.cache_root)
    if marker.exists():
        return True
    print(terms_text([b.id for b in domains_cfg.all_benchmarks]), flush=True)
    if not accepted:
        if not sys.stdin.isatty():
            return False
        if input("Accept these terms and continue? [y/N] ").strip().lower() not in ("y", "yes"):
            return False
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prism-vlm",
        description="Build the PRISM-VLM benchmark, evaluate models, and score them.",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    build = subs.add_parser("build", help="Synthesise the per-item variant bundle.")
    build.add_argument("--config", required=True, type=Path, help="Path to a run YAML")
    _add_subset_args(build)
    _add_terms_arg(build)

    run = subs.add_parser("run", help="Evaluate models and grade every axis.")
    run.add_argument("--config", required=True, type=Path, help="Path to a run YAML")
    run.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model ids from models.yaml (default: all).",
    )
    run.add_argument(
        "--axes",
        nargs="+",
        choices=AXES,
        default=None,
        help="Restrict inference to these axes (default: all seven).",
    )
    _add_subset_args(run)
    _add_terms_arg(run)

    score = subs.add_parser("score", help="Aggregate graded items into per-axis means and PScore.")
    score.add_argument("--config", required=True, type=Path, help="Path to a run YAML")
    score.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model ids to report (default: every model found in the run).",
    )
    score.add_argument(
        "--aggregator",
        choices=AGGREGATOR_NAMES,
        default="harmonic",
        help="How to combine the seven axis means (default: harmonic, the paper's PScore).",
    )
    score.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write scores.json and scores.csv here (default: output_dir from the config).",
    )
    return parser


def _configure_cache_early(config_path: Path) -> None:
    """Load .env and set the HF cache vars before any HF / SDK import."""
    from .config.loader import load_run_config
    from .utils.dotenv import find_project_dotenv, load_dotenv
    from .utils.io import configure_cache_env

    project_root = Path(__file__).resolve().parent.parent
    config_dir = config_path.resolve().parent
    loaded = load_dotenv(find_project_dotenv(config_dir) + find_project_dotenv(project_root))
    for path in loaded:
        # The search reaches three directories above the repository, so a .env
        # belonging to some enclosing project would otherwise apply in silence.
        print(f"[env] loaded {path}", flush=True)

    configure_cache_env(load_run_config(config_path).cache_root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # MUST run before any module imports ``datasets`` / ``transformers``.
    _configure_cache_early(args.config)

    from .config.loader import load_domains, load_run_config

    cfg = load_run_config(args.config)
    domains_cfg = load_domains(cfg.domains_path)

    if args.command in ("build", "run"):
        # Both stages call the judge, but only after sampling the pool,
        # which downloads the source benchmarks. Build the client now --
        # it does no I/O -- so a missing key costs a second, not a fetch.
        from .evaluation.judge import JudgeClient

        try:
            JudgeClient(cfg.judge)
        except Exception as e:  # noqa: BLE001
            parser.error(
                f"judge {cfg.judge.model!r} is not usable: {e}. "
                f"Copy .env.example to .env and fill in the key it names."
            )

        # After the judge check, so nobody accepts the terms only to fail on a key.
        if not _confirm_dataset_terms(cfg, domains_cfg, args.accept_dataset_terms):
            parser.error("the source benchmarks' terms were not accepted; "
                         "pass --accept-dataset-terms to accept them non-interactively")

    if args.command == "build":
        from .runner.build_pipeline import build_variant_bundle

        asyncio.run(build_variant_bundle(
            cfg,
            domains_cfg,
            only_benchmarks=args.benchmarks,
            only_domains=args.domains,
            limit=args.limit,
        ))
        return 0

    if args.command == "run":
        from .config.loader import load_models
        from .runner.run_pipeline import run_model_evaluation

        if cfg.models_path is None:
            parser.error(f"{args.config} declares no `models:` file, which `run` needs")
        models_cfg = load_models(cfg.models_path)
        asyncio.run(
            run_model_evaluation(
                cfg,
                domains_cfg,
                models_cfg,
                only_models=args.models,
                only_axes=args.axes,
                only_benchmarks=args.benchmarks,
                only_domains=args.domains,
                limit=args.limit,
            )
        )
        return 0

    if args.command == "score":
        from .config.loader import plain_name
        from .scoring import discover_models, format_table, score_models, write_report

        scores_dir = cfg.cache_run_dir / "scores"
        # `score` takes model ids from the command line rather than the
        # registry, so they reach the cache path unvalidated otherwise.
        for model_id in args.models or ():
            try:
                plain_name(model_id)
            except ValueError as e:
                parser.error(f"--models {model_id!r}: {e}")
        model_ids = args.models or discover_models(scores_dir)
        report = score_models(scores_dir, model_ids, aggregator=args.aggregator)
        print(format_table(report, args.aggregator))

        out_dir = args.output or cfg.output_dir
        if out_dir and report:
            for path in write_report(report, out_dir, args.aggregator):
                print(f"[score] wrote {path}")
        return 0

    parser.error(f"unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
