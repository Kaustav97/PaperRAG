"""Local CLI; JSON results on stdout, human progress/errors on stderr."""
from __future__ import annotations

import functools
import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from .config import load_settings
from .models import StageEvent

app = typer.Typer(no_args_is_help=True, help="Page-grounded local research PDFs")
console = Console(stderr=True)
DataDir = Annotated[Path | None, typer.Option("--data-dir", help="Override storage only")]
DocIds = Annotated[list[str] | None, typer.Option("--document-id", help="Restrict every retrieval round to these indexed IDs")]


def guarded(fn):
    @functools.wraps(fn)
    def run(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except typer.Exit:
            raise
        except Exception as exc:
            console.print(f"Error: {exc}", style="red", markup=False)
            raise typer.Exit(1) from None
    return run


def settings_for(data_dir):
    settings = load_settings()
    if data_dir is not None:
        settings.data_dir = data_dir.expanduser().resolve()
    return settings


def progress(event: StageEvent):
    console.print(f"{event.stage}: {event.message}", markup=False)


def emit(value, json_output=False):
    data = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    if json_output:
        typer.echo(json.dumps(data, ensure_ascii=False))
    else:
        console.print_json(data=data)


@app.command("download-samples")
@guarded
def download_samples(destination: Path | None = None):
    """Download the three official arXiv PDFs, preserving provenance."""
    from .samples import download_samples as download
    result = download(destination or load_settings().pdf_dir)
    emit(result)
    if result["failures"]:
        raise typer.Exit(1)


@app.command()
@guarded
def ingest(paths: Annotated[list[Path], typer.Argument()], data_dir: DataDir = None, force_ocr: bool = False):
    """Recursively index PDFs without changing originals."""
    from .ingest import Ingestor
    report = Ingestor(settings_for(data_dir), on_progress=progress).ingest_paths(paths, force_ocr=force_ocr)
    emit(report)
    if report.failed:
        raise typer.Exit(1)


@app.command("index")
@guarded
def maintain_index(data_dir: DataDir = None):
    """Reconcile pending commits and maintain native FTS/scalar/ANN indexes."""
    from .storage import Store
    with Store(settings_for(data_dir)) as store, store.writer():
        emit(store.maintain_indexes())


@app.command()
@guarded
def stats(data_dir: DataDir = None, json_output: Annotated[bool, typer.Option("--json")] = False):
    """Show committed corpus counts and index readiness."""
    from .storage import Store
    with Store(settings_for(data_dir)) as store:
        emit(store.stats(), json_output)


@app.command()
@guarded
def search(question: str, data_dir: DataDir = None, top_k: int = 12, document_id: DocIds = None,
           json_output: Annotated[bool, typer.Option("--json")] = False):
    """Search local evidence without credentials or answer generation."""
    from .retrieval import RetrievalSession
    with RetrievalSession(settings_for(data_dir), document_ids=document_id) as session:
        sources = session.retrieve(question, top_k=top_k)
    emit({"status": "ok" if sources else "insufficient_evidence", "question": question,
          "message": "Nearest neighbors are not proof of relevance." if sources else "No evidence found. Ingest papers with: paperrag ingest PATH",
          "sources": [s.model_copy(update={"label": f"S{i}"}).model_dump() for i, s in enumerate(sources, 1)]}, json_output)


@app.command()
@guarded
def ask(question: str, data_dir: DataDir = None, document_id: DocIds = None,
        json_output: Annotated[bool, typer.Option("--json")] = False):
    """Run deconstruction, bounded research, synthesis and independent checking."""
    from .pipeline import Pipeline
    result = Pipeline(settings_for(data_dir)).answer(question, document_ids=document_id, on_progress=progress)
    if json_output:
        emit(result, True)
    else:
        console.print(result.answer_markdown, markup=False)
        console.print("Checked against retrieved paper excerpts — not independent scientific truth.")
        emit(result)


@app.command()
@guarded
def ui(host: str = "127.0.0.1", port: int = 8501):
    """Launch the browser app from the checkout, using this environment's Python."""
    project_dir = Path(__file__).resolve().parents[2]
    # Editable installs keep configuration and relative paths anchored to the checkout.
    working_dir = project_dir if (project_dir / "pyproject.toml").is_file() else Path.cwd()
    result = subprocess.run([sys.executable, "-m", "streamlit", "run", str(Path(__file__).with_name("web.py")),
                             "--server.address", host, "--server.port", str(port),
                             "--server.headless", "true", "--browser.gatherUsageStats", "false"], cwd=working_dir)
    raise typer.Exit(result.returncode)
