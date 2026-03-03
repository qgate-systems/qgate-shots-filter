"""
cli.py — Command-line interface for qgate.

Usage::

    qgate run config.json --adapter mock --seed 42
    qgate validate config.json
    qgate version
    qgate adapters
    qgate schema

Powered by `typer <https://typer.tiangolo.com>`_.

Patent reference: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer

if TYPE_CHECKING:
    from qgate.adapters.base import BaseAdapter
    from qgate.config import GateConfig

app = typer.Typer(
    name="qgate",
    help="Quantum Trajectory Filter — runtime post-selection for quantum circuits.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Verbosity callback (applied globally)
# ---------------------------------------------------------------------------


def _configure_logging(verbose: bool, quiet: bool) -> None:
    """Set up stdlib logging based on CLI flags."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        format="%(name)s | %(levelname)s | %(message)s",
        level=level,
        force=True,
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def adapters() -> None:
    """List all registered qgate adapters (entry-point discovery)."""
    from qgate.adapters.registry import list_adapters

    available = list_adapters()
    if not available:
        typer.echo("No adapters discovered.")
        return
    typer.echo("Registered adapters:")
    for name, target in sorted(available.items()):
        typer.echo(f"  {name:15s}  →  {target}")


@app.command()
def schema() -> None:
    """Print the GateConfig JSON Schema."""
    import json as _json

    from qgate.config import GateConfig

    typer.echo(_json.dumps(GateConfig.model_json_schema(), indent=2))


@app.command()
def version() -> None:
    """Print the qgate version and exit."""
    from qgate import __version__

    typer.echo(f"qgate {__version__}")


@app.command()
def validate(
    config_path: Path = typer.Argument(
        ..., exists=True, readable=True, help="Path to JSON config file."
    ),
) -> None:
    """Validate a GateConfig JSON file."""
    from qgate.config import GateConfig

    try:
        text = config_path.read_text()
        cfg = GateConfig.model_validate_json(text)
        typer.echo(
            f"✅  Config valid — variant={cfg.variant.value}, "
            f"n_subsystems={cfg.n_subsystems}, shots={cfg.shots}"
        )
    except Exception as exc:
        typer.echo(f"❌  Validation error: {exc}", err=True)
        raise typer.Exit(code=1) from None


@app.command()
def run(
    config_path: Path = typer.Argument(
        ..., exists=True, readable=True, help="Path to JSON config file."
    ),
    adapter: str = typer.Option(
        "mock", "--adapter", "-a", help="Adapter: mock | qiskit | cirq | pennylane"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Log output file (.jsonl, .csv, .parquet)"
    ),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed (mock adapter)"),
    error_rate: Optional[float] = typer.Option(
        None,
        "--error-rate",
        "-e",
        help="Mock adapter per-subsystem per-cycle error rate (default 0.05)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="DEBUG-level logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress info messages"),
) -> None:
    """Run a trajectory filter with the given config."""
    _configure_logging(verbose, quiet)

    from qgate.config import GateConfig
    from qgate.filter import TrajectoryFilter
    from qgate.run_logging import RunLogger

    # Load config
    text = config_path.read_text()
    cfg = GateConfig.model_validate_json(text)

    # Build adapter
    backend_adapter = _make_adapter(
        adapter,
        seed=seed,
        error_rate=error_rate,
        config=cfg,
    )

    # Optional logger (use context manager to flush Parquet on exit)
    run_logger = RunLogger(output) if output else None

    try:
        tf = TrajectoryFilter(cfg, backend_adapter, logger=run_logger)
        result = tf.run()
    finally:
        if run_logger is not None:
            run_logger.close()

    typer.echo(
        f"run_id={result.run_id}  "
        f"variant={result.variant}  "
        f"shots={result.total_shots}  "
        f"accepted={result.accepted_shots}  "
        f"P_acc={result.acceptance_probability:.4f}  "
        f"TTS={result.tts:.2f}"
    )
    if result.mean_combined_score is not None:
        typer.echo(
            f"mean_score={result.mean_combined_score:.4f}  threshold={result.threshold_used:.4f}"
        )

    if output:
        typer.echo(f"📝  Logged to {output}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_adapter(
    name: str,
    seed: Optional[int] = None,
    error_rate: Optional[float] = None,
    config: Optional[GateConfig] = None,
) -> BaseAdapter:
    from qgate.adapters.registry import list_adapters, load_adapter

    # Special-case: mock adapter accepts seed/error_rate kwargs
    if name == "mock":
        from qgate.adapters.base import MockAdapter

        return MockAdapter(error_rate=error_rate or 0.05, seed=seed)

    available = list_adapters()
    if name not in available:
        typer.echo(
            f"❌  Unknown adapter: {name}. Available: {', '.join(sorted(available))}",
            err=True,
        )
        raise typer.Exit(code=1)

    cls = load_adapter(name)

    # Pass adapter_options from config if available
    if config is not None and config.adapter_options:
        return cls(**config.adapter_options)  # type: ignore[no-any-return]
    return cls()  # type: ignore[no-any-return]


if __name__ == "__main__":
    app()
