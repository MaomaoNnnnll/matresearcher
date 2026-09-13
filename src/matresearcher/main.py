"""MatResearcher CLI entry point.

Usage:
    matresearcher survey "What is the current status of LLZO electrolyte research?"
    matresearcher survey --question "LLZO ionic conductivity" --output report.md
    matresearcher serve --port 8000
    matresearcher evaluate --baseline keyword
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

# Load project .env + user secrets.env BEFORE any config/module that reads os.getenv()
from .env_loader import load_env_files

load_env_files()

from .workflow.engine import MatResearcherWorkflow
from .state import WorkflowState

app = typer.Typer(
    name="matresearcher",
    help="MatResearcher: Multi-agent literature survey for solid-state battery materials",
)
console = Console()


@app.command()
def survey(
    question: Optional[str] = typer.Argument(None, help="Research question to survey"),
    q: Optional[str] = typer.Option(None, "--question", "-q", help="Research question"),
    config: str = typer.Option(
        None, "--config", "-c", help="Path to workflow YAML config"
    ),
    output: str = typer.Option(
        "report.md", "--output", "-o", help="Output file path for the report"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    report_only: bool = typer.Option(
        False, "--report-only", help="Output only the report text (no progress)"
    ),
    resume: Optional[str] = typer.Option(
        None,
        "--resume",
        help="Resume from a previous run by run_id (e.g. 2026-08-12_100048). "
             "Completed nodes and extracted papers are reused from cache.",
    ),
):
    """Run a literature survey for a research question.

    Examples:
        matresearcher survey "LLZO solid electrolyte conductivity comparison"
        matresearcher survey -q "What is the optimal sintering temperature for LLZO?" -o llzo_report.md
        matresearcher survey -q "..." --resume 2026-08-12_100048
    """
    question = question or q
    if not question:
        console.print("[red]Error: Please provide a research question.[/red]")
        console.print("Usage: matresearcher survey <question>")
        console.print("       matresearcher survey --question <question>")
        raise typer.Exit(1)

    if not report_only:
        console.print(f"\n[bold cyan]MatResearcher Literature Survey[/bold cyan]")
        console.print(f"[dim]Question: {question}[/dim]")

    async def _run():
        workflow = MatResearcherWorkflow(config, resume_run_id=resume)
        result = await workflow.run(question)

        if result.get("error"):
            console.print(f"\n[red]Error: {result['error']}[/red]")
            raise typer.Exit(1)

        # submission 模式下优先取 submission_report（参赛方案文档）；
        # survey 模式该行返回空串，自动退回 final_report / draft_report（survey 报告）。
        report = (
            result.get("submission_report")
            or result.get("final_report")
            or result.get("draft_report", "")
        )
        generate_time = datetime.now().strftime("%Y%m%d%H%M%S")

        # Write report to output file — same run folder as logs
        run_dir = workflow.run_output_dir
        if run_dir:
            output_path = Path(run_dir) / output
        else:
            output_path = Path().cwd() / "outputs" / generate_time / output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")

        if not report_only:
            console.print(f"\n[green]Report saved to: {output_path.absolute()}[/green]")

            # Print summary
            _print_summary(result)

        return report

    report = asyncio.run(_run())

    if report_only:
        print(report)


@app.command()
def serve(
    port: int = typer.Option(8000, "--port", "-p", help="Port to serve on"),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
):
    """Start a simple HTTP API server for running surveys.

    Endpoints:
        POST /survey  - Run a survey (body: {"question": "..."})
        GET  /health  - Health check
    """
    try:
        from fastapi import FastAPI
        import uvicorn
    except ImportError:
        console.print("[red]Error: fastapi and uvicorn required for serve mode.[/red]")
        console.print("Install: pip install fastapi uvicorn")
        raise typer.Exit(1)

    api = FastAPI(title="MatResearcher API")

    @api.get("/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    @api.post("/survey")
    async def run_survey(body: dict):
        question = body.get("question", "")
        if not question:
            return {"error": "question is required"}, 400

        workflow = MatResearcherWorkflow()
        result = await workflow.run(question)
        return {
            "report": (
                result.get("submission_report")
                or result.get("final_report")
                or result.get("draft_report", "")
            ),
            "step_log": result.get("step_log", []),
        }

    console.print(f"[green]Starting MatResearcher API on http://{host}:{port}[/green]")
    uvicorn.run(api, host=host, port=port)


@app.command()
def evaluate(
    question: Optional[str] = typer.Argument(None, help="Research question"),
    baseline: str = typer.Option(
        "all",
        "--baseline",
        "-b",
        help="Baseline methods: keyword, semantic, hybrid, rag, all",
    ),
    output: str = typer.Option(
        "evaluation_results.json", "--output", "-o", help="Output path"
    ),
):
    """Evaluate MatResearcher against baseline methods.

    Baselines:
        keyword   - Simple keyword search + LLM summary
        semantic  - Semantic search + LLM summary
        hybrid    - Hybrid search + LLM summary
        rag       - Single-agent RAG pipeline
        all       - Run all baselines
    """
    # NOTE: this used to be `from ...scripts.evaluate import run_evaluation`,
    # a 3-level relative import beyond the top-level package that raised
    # `ImportError: attempted relative import beyond top-level package` at
    # runtime. The harness now lives inside the package (see evaluation/).
    from .evaluation import run_evaluation, DEFAULT_QUESTION

    if not question:
        question = DEFAULT_QUESTION

    console.print(f"[cyan]Running evaluation for: {question}[/cyan]")
    console.print(f"[dim]Baseline: {baseline}[/dim]")

    result = asyncio.run(run_evaluation(question, baseline, output))
    console.print(f"[green]Results saved to: {output}[/green]")


@app.command()
def version():
    """Print MatResearcher version."""
    from matresearcher import __version__
    console.print(f"MatResearcher v{__version__}")


def _print_summary(result: dict):
    """Print a workflow execution summary table."""
    table = Table(title="Workflow Summary")
    table.add_column("Step", style="cyan", no_wrap=True)
    table.add_column("Description", style="green")

    log = result.get("step_log", [])
    for entry in log:
        table.add_row(str(entry.get("step", "")), entry.get("message", ""))

    console.print(table)


if __name__ == "__main__":
    app()
