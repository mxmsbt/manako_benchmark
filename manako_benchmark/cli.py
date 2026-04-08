"""CLI entry point for the Manako Vision AI Benchmark."""
import json
from pathlib import Path

import click

from .data.dataset import BenchmarkDataset
from .evaluation.runner import run_benchmark
from .evaluation.temporal import TemporalTracker
from .reporting.dashboard import generate_report


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Manako Vision AI Detection Benchmark.

    Compare SN44 (subnet-trained) vs SAM3 (ceiling) vs Roboflow (best open-source)
    on object detection using mAP@50.
    """


@main.command()
@click.option("--frames", required=True, type=click.Path(exists=True), help="Directory containing evaluation frames")
@click.option("--annotations", required=True, type=click.Path(exists=True), help="COCO JSON or YOLO labels directory")
@click.option("--sn44-weights", default=None, help="Path to SN44 ONNX weights (auto-downloads from HF if omitted)")
@click.option("--sn44-repo", default="alfred8995/kane001", help="HuggingFace repo for SN44 weights")
@click.option("--sam3-endpoint", default=None, help="SAM3 API endpoint URL")
@click.option("--sam3-key", default=None, help="SAM3 API key")
@click.option("--roboflow-key", default=None, help="Roboflow API key")
@click.option("--roboflow-model", default="vehicles-q0x2v/1", help="Roboflow model ID")
@click.option("--roboflow-local", is_flag=True, help="Use local Roboflow inference SDK")
@click.option("--checkpoint-tag", default="", help="Tag for temporal tracking (e.g. epoch_10)")
@click.option("--output-dir", default="results", help="Output directory for results")
@click.option("--report-dir", default="reports", help="Output directory for HTML report")
@click.option("--device", default="cpu", type=click.Choice(["cpu", "cuda"]), help="Inference device")
@click.option("--conf-threshold", default=0.45, type=float, help="Confidence threshold for SN44")
@click.option("--models", default="sn44,roboflow,sam3", help="Comma-separated list of models to evaluate")
def run(
    frames, annotations, sn44_weights, sn44_repo,
    sam3_endpoint, sam3_key,
    roboflow_key, roboflow_model, roboflow_local,
    checkpoint_tag, output_dir, report_dir, device, conf_threshold, models,
):
    """Run the full benchmark evaluation."""
    model_list = [m.strip() for m in models.split(",")]

    # Load dataset
    click.echo(f"Loading dataset from {frames}...")
    dataset = BenchmarkDataset(frames, annotations)
    click.echo(f"  {len(dataset)} images, {len(dataset.annotations)} annotations, {len(dataset.categories)} classes")

    # Initialize models
    adapters = []

    if "sn44" in model_list:
        click.echo("Initializing SN44 model...")
        from .models.sn44 import SN44Adapter
        adapters.append(SN44Adapter(
            model_path=sn44_weights,
            hf_repo=sn44_repo,
            device=device,
            conf_threshold=conf_threshold,
        ))

    if "roboflow" in model_list:
        if not roboflow_key:
            click.echo("WARNING: --roboflow-key not provided, skipping Roboflow model")
        else:
            click.echo("Initializing Roboflow model...")
            from .models.roboflow import RoboflowAdapter
            adapters.append(RoboflowAdapter(
                api_key=roboflow_key,
                model_id=roboflow_model,
                use_local=roboflow_local,
            ))

    if "sam3" in model_list:
        if not sam3_endpoint:
            click.echo("WARNING: --sam3-endpoint not provided, skipping SAM3 model")
        else:
            click.echo("Initializing SAM3 model...")
            from .models.sam3 import SAM3Adapter
            adapters.append(SAM3Adapter(
                endpoint_url=sam3_endpoint,
                api_key=sam3_key,
            ))

    if not adapters:
        click.echo("ERROR: No models configured. Provide at least one model's credentials.")
        raise SystemExit(1)

    # Run benchmark
    click.echo(f"\nRunning benchmark with {len(adapters)} model(s)...")
    results = run_benchmark(dataset, adapters, checkpoint_tag=checkpoint_tag)

    # Save results
    output_path = Path(output_dir)
    results_file = output_path / f"results_{results.timestamp[:10]}.json"
    results.save(results_file)
    click.echo(f"\nResults saved to: {results_file}")

    # Update temporal tracker
    tracker = TemporalTracker(output_path / "history.json")
    tracker.add_result(results)

    # Generate report
    report_path = Path(report_dir) / f"report_{results.timestamp[:10]}.html"
    generate_report(results, report_path, tracker=tracker)
    click.echo(f"Report saved to: {report_path}")

    # Print summary
    click.echo("\n" + "=" * 60)
    click.echo("BENCHMARK SUMMARY")
    click.echo("=" * 60)
    for i, mr in enumerate(sorted(results.model_results, key=lambda m: m.mAP50, reverse=True)):
        rank = i + 1
        click.echo(f"  #{rank} {mr.model_name.upper():12s} mAP@50={mr.mAP50:.4f}  ({mr.avg_inference_ms:.0f}ms avg)")
    click.echo("=" * 60)


@main.command()
@click.option("--results-dir", default="results", type=click.Path(exists=True), help="Directory with history.json")
@click.option("--output", default="reports/report_latest.html", help="Output HTML path")
def report(results_dir, output):
    """Regenerate the HTML report from saved results."""
    history_path = Path(results_dir) / "history.json"
    if not history_path.exists():
        click.echo(f"No history found at {history_path}")
        raise SystemExit(1)

    tracker = TemporalTracker(history_path)
    if not tracker.history:
        click.echo("History is empty.")
        raise SystemExit(1)

    # Load latest results
    latest = tracker.history[-1]
    from .evaluation.runner import BenchmarkResults, ModelResult
    results = BenchmarkResults(
        timestamp=latest["timestamp"],
        dataset_info=latest.get("dataset_info", {}),
    )
    for model_name, data in latest["models"].items():
        results.model_results.append(ModelResult(
            model_name=model_name,
            mAP50=data["mAP50"],
            per_class_ap={int(k): v for k, v in data.get("per_class_ap", {}).items()},
            avg_inference_ms=data.get("avg_inference_ms", 0),
        ))

    generate_report(results, output, tracker=tracker)


@main.command()
@click.option("--results-dir", default="results", type=click.Path(exists=True))
def history(results_dir):
    """Show benchmark history and improvement over time."""
    tracker = TemporalTracker(Path(results_dir) / "history.json")

    if not tracker.history:
        click.echo("No benchmark history found.")
        return

    click.echo(f"Total runs: {len(tracker.history)}\n")

    # Latest comparison
    click.echo("Latest comparison:")
    for row in tracker.get_comparison_table():
        click.echo(f"  {row['model'].upper():12s} mAP@50={row['mAP50']:.4f}")

    # Improvement for SN44
    improvement = tracker.get_improvement_summary("sn44")
    if improvement:
        click.echo(f"\nSN44 improvement over {improvement['num_checkpoints']} checkpoints:")
        click.echo(f"  {improvement['initial_mAP50']*100:.1f}% -> {improvement['final_mAP50']*100:.1f}% (+{improvement['absolute_improvement']*100:.1f}%)")


@main.command()
@click.option("--frames", required=True, type=click.Path(exists=True), help="Directory containing evaluation frames")
@click.option("--annotations", required=True, type=click.Path(exists=True), help="Annotations path")
def validate(frames, annotations):
    """Validate dataset format and integrity."""
    try:
        dataset = BenchmarkDataset(frames, annotations)
        click.echo(f"Dataset valid!")
        click.echo(f"  Images:      {len(dataset)}")
        click.echo(f"  Annotations: {len(dataset.annotations)}")
        click.echo(f"  Categories:  {len(dataset.categories)}")
        click.echo(f"  Classes:     {dataset.get_class_names()}")

        # Check annotation distribution
        from collections import Counter
        class_counts = Counter(a["category_id"] for a in dataset.annotations)
        click.echo("\n  Annotation distribution:")
        for cls_id, count in sorted(class_counts.items()):
            name = dataset.get_class_names().get(cls_id, str(cls_id))
            click.echo(f"    {name}: {count}")
    except Exception as e:
        click.echo(f"Validation failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
