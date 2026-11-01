"""Execute the notebook using only its original Excel input and compare outputs."""

import asyncio
import base64
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


async def main():
    notebook_path = ROOT / "V93_Excel_to_CSV.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    protected = {name: sha(ROOT / name) for name in (
        "v93_model.py", "admissions_calendar.py", "forecast_2024_v93.csv",
        "forecast_2024_v84.csv", "output/v93/selected_config.json",
    )}
    score_record = json.loads((ROOT / "output/v93/leaderboard_result.json").read_text())
    assert protected["forecast_2024_v93.csv"] == score_record["sha256"]

    with tempfile.TemporaryDirectory(prefix="v93_notebook_excel_") as temporary:
        work = Path(temporary)
        (work / "abitur_stage1").mkdir()
        name = "Зарегистрированы (2014-2023).xlsx"
        (work / "abitur_stage1" / name).symlink_to(ROOT / "abitur_stage1" / name)

        kernels = work / "kernels"
        kernel_dir = kernels / "v93-check"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "V93 verification", "language": "python",
        }))
        manager = AsyncKernelManager(
            kernel_name="v93-check",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]),
        )
        client = NotebookClient(
            notebook, km=manager, timeout=180,
            resources={"metadata": {"path": str(work)}},
        )
        try:
            await client.async_execute()
        finally:
            if manager.has_kernel:
                await manager.shutdown_kernel(now=True)

        folder = work / "output/v93_notebook"
        expected = {
            "forecast_2024_v93.csv": ROOT / "forecast_2024_v93.csv",
            "forecast_2024_v93_bachelor.csv": ROOT / "forecast_2024_v93_bachelor.csv",
            "forecast_2024_v93_master.csv": ROOT / "forecast_2024_v93_master.csv",
            "forecast_2024_v93_components.csv": ROOT / "output/v93/degree_forecasts.csv",
        }
        for filename, original in expected.items():
            if (folder / filename).read_bytes() != original.read_bytes():
                raise AssertionError(f"Notebook forecast differs from verified V93: {filename}")

        destination = ROOT / "output/v93_notebook"
        destination.mkdir(exist_ok=True)
        for path in folder.glob("*.csv"):
            shutil.copy2(path, destination / path.name)

        image_outputs = [output for cell in notebook.cells for output in cell.get("outputs", [])
                         if "image/png" in output.get("data", {})]
        assert image_outputs, "Forecast plot missing"
        (destination / "forecast.png").write_bytes(base64.b64decode(image_outputs[-1]["data"]["image/png"]))

    for filename, digest in protected.items():
        assert sha(ROOT / filename) == digest, filename
    nbformat.write(notebook, notebook_path)
    report = {
        "executed_in_fresh_kernel": True,
        "executed_in_isolated_directory": True,
        "input": "original Excel registrations only; calendar and configuration embedded",
        "old_python_modules_or_forecast_inputs_required": False,
        "matched_byte_for_byte": list(expected),
        "original_public_score_user_reported": score_record["public_rmsle"],
        "combined_csv_sha256": protected["forecast_2024_v93.csv"],
        "protected_originals_unchanged": True,
        "cells": len(notebook.cells),
        "executed_code_cells": sum(cell.cell_type == "code" for cell in notebook.cells),
    }
    (ROOT / "output/v93_notebook/verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
