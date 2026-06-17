"""Command-line interface for the Data Factory pipeline.

Provides commands for running the pipeline, listing templates, and
inspecting results from the terminal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_factory import DataFactory
from data_factory.config import FactorySettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data-factory",
        description="AI Training Dataset Generation Pipeline",
        epilog="Example: data-factory run --sources doc.pdf https://example.com --tasks qa summarization",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── run ──────────────────────────────────────────────────────────
    run_parser = subparsers.add_parser("run", help="Run the full pipeline")
    run_parser.add_argument(
        "--sources",
        "-s",
        nargs="+",
        required=True,
        help="Sources to load (files, PDFs, URLs)",
    )
    run_parser.add_argument(
        "--tasks",
        "-t",
        nargs="+",
        default=None,
        help="Tasks to generate (qa, summarization, classification)",
    )
    run_parser.add_argument(
        "--name",
        "-n",
        default=None,
        help="Run / dataset name",
    )
    run_parser.add_argument(
        "--export",
        "-e",
        default=None,
        help="Export path for the dataset (e.g., output/training.jsonl)",
    )
    run_parser.add_argument(
        "--no-quality",
        action="store_true",
        help="Skip quality evaluation",
    )
    run_parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Token chunk size (default: 512)",
    )
    run_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for logs and exports",
    )
    run_parser.add_argument(
        "--api-key",
        default=None,
        help="LLM API key",
    )
    run_parser.add_argument(
        "--model",
        default=None,
        help="LLM model name (e.g., gpt-4o-mini)",
    )
    run_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose/debug logging",
    )

    # ── list-templates ───────────────────────────────────────────────
    subparsers.add_parser(
        "list-templates", help="List available task templates"
    )

    # ── config ───────────────────────────────────────────────────────
    config_parser = subparsers.add_parser(
        "config", help="Show current configuration"
    )
    config_parser.add_argument(
        "--show", action="store_true", help="Show resolved configuration"
    )

    # ── version ──────────────────────────────────────────────────────
    subparsers.add_parser("version", help="Show version information")

    return parser


def cmd_run(args: argparse.Namespace) -> None:
    """Execute the ``run`` subcommand."""
    settings_kwargs: Dict[str, Any] = {}

    if args.chunk_size:
        settings_kwargs["chunk_size"] = args.chunk_size
    if args.output_dir:
        settings_kwargs["output_dir"] = args.output_dir
    if args.api_key:
        settings_kwargs["llm_api_key"] = args.api_key
    if args.model:
        settings_kwargs["llm_model"] = args.model
    if args.verbose:
        settings_kwargs["log_level"] = "DEBUG"

    settings = FactorySettings(**settings_kwargs)
    factory = DataFactory(settings)

    dataset = factory.run(
        sources=args.sources,
        tasks=args.tasks,
        run_name=args.name,
        export_path=args.export,
        skip_quality=args.no_quality,
    )

    summary = factory.summary()
    summary_json = json.dumps(summary, indent=2, default=str)
    print(f"\n{summary_json}")


def cmd_list_templates() -> None:
    """Execute the ``list-templates`` subcommand."""
    from data_factory.tasks.templates import TaskTemplateLibrary

    library = TaskTemplateLibrary()
    print(f"{'Name':<20} {'Task Type':<18} {'Description'}")
    print("-" * 80)
    for tmpl in library.list_templates():
        print(
            f"{tmpl['name']:<20} {tmpl['task_type']:<18} {tmpl['description']}"
        )
    print(f"\nTotal templates: {library.count}")


def cmd_config() -> None:
    """Execute the ``config`` subcommand."""
    settings = FactorySettings()
    data = settings.model_dump()
    print(json.dumps(data, indent=2, default=str))


def cmd_version() -> None:
    """Execute the ``version`` subcommand."""
    from data_factory import __version__, __author__

    print(f"Data Factory v{__version__}")
    print(f"Author: {__author__}")


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "list-templates":
        cmd_list_templates()
    elif args.command == "config":
        cmd_config()
    elif args.command == "version":
        cmd_version()
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
