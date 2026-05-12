"""Contract tests for Docker support files (T038).

Tests verify structural/content contracts for Dockerfile, requirements-docker.txt,
and .dockerignore without requiring a Docker daemon.
"""
from pathlib import Path
import shutil
import subprocess

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestDockerfile:
    DOCKERFILE = PROJECT_ROOT / "Dockerfile"

    def test_dockerfile_exists(self) -> None:
        assert self.DOCKERFILE.exists(), "Dockerfile not found at project root"

    def test_base_image_is_slim(self) -> None:
        text = self.DOCKERFILE.read_text()
        assert "FROM python:3.11-slim" in text, (
            "Dockerfile must use python:3.11-slim as base image"
        )

    def test_entrypoint_is_src_train(self) -> None:
        text = self.DOCKERFILE.read_text()
        assert 'ENTRYPOINT ["python", "-m", "src.train"]' in text, (
            'Dockerfile ENTRYPOINT must be ["python", "-m", "src.train"]'
        )

    def test_data_volume_declared(self) -> None:
        text = self.DOCKERFILE.read_text()
        assert '"/app/data"' in text, (
            "Dockerfile must declare /app/data as a VOLUME"
        )

    def test_results_volume_declared(self) -> None:
        text = self.DOCKERFILE.read_text()
        assert '"/app/results"' in text, (
            "Dockerfile must declare /app/results as a VOLUME"
        )

    def test_cpu_only_wheel_index(self) -> None:
        text = self.DOCKERFILE.read_text()
        assert "download.pytorch.org/whl/cpu" in text, (
            "Dockerfile must install torch from the CPU-only wheel index"
        )

    def test_no_xpu_wheel_index(self) -> None:
        text = self.DOCKERFILE.read_text()
        assert "nightly/xpu" not in text, (
            "Dockerfile must not reference the XPU nightly wheel index"
        )

    def test_unbuffered_env(self) -> None:
        text = self.DOCKERFILE.read_text()
        assert "PYTHONUNBUFFERED" in text, (
            "Dockerfile must set PYTHONUNBUFFERED for reliable log output"
        )

    def test_requirements_docker_used(self) -> None:
        text = self.DOCKERFILE.read_text()
        assert "requirements-docker.txt" in text, (
            "Dockerfile must COPY and install requirements-docker.txt"
        )

    def test_src_copied(self) -> None:
        text = self.DOCKERFILE.read_text()
        assert "COPY src/" in text, "Dockerfile must COPY src/ into the image"

    def test_tests_copied(self) -> None:
        text = self.DOCKERFILE.read_text()
        assert "COPY tests/" in text, "Dockerfile must COPY tests/ into the image"


class TestRequirementsDocker:
    REQS = PROJECT_ROOT / "requirements-docker.txt"

    def test_file_exists(self) -> None:
        assert self.REQS.exists(), "requirements-docker.txt not found at project root"

    def test_contains_matplotlib(self) -> None:
        text = self.REQS.read_text()
        assert "matplotlib" in text, (
            "requirements-docker.txt must include matplotlib"
        )

    def test_contains_pytest(self) -> None:
        text = self.REQS.read_text()
        assert "pytest" in text, (
            "requirements-docker.txt must include pytest"
        )

    def test_no_torch_entry(self) -> None:
        text = self.REQS.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            assert not stripped.startswith("torch"), (
                "requirements-docker.txt must not list torch or torchvision "
                "(they are installed inline in the Dockerfile from the CPU wheel index)"
            )

    def test_no_xpu_index_url(self) -> None:
        text = self.REQS.read_text()
        assert "nightly/xpu" not in text, (
            "requirements-docker.txt must not reference the XPU nightly index"
        )


class TestDockerignore:
    DOCKERIGNORE = PROJECT_ROOT / ".dockerignore"

    def test_file_exists(self) -> None:
        assert self.DOCKERIGNORE.exists(), ".dockerignore not found at project root"

    def test_excludes_data_dir(self) -> None:
        lines = {line.strip() for line in self.DOCKERIGNORE.read_text().splitlines()}
        assert "data/" in lines, ".dockerignore must exclude data/"

    def test_excludes_results_dir(self) -> None:
        lines = {line.strip() for line in self.DOCKERIGNORE.read_text().splitlines()}
        assert "results/" in lines, ".dockerignore must exclude results/"

    def test_excludes_git_dir(self) -> None:
        lines = {line.strip() for line in self.DOCKERIGNORE.read_text().splitlines()}
        assert ".git/" in lines, ".dockerignore must exclude .git/"

    def test_excludes_pycache(self) -> None:
        text = self.DOCKERIGNORE.read_text()
        assert "__pycache__/" in text, ".dockerignore must exclude __pycache__/"

    def test_excludes_venv(self) -> None:
        text = self.DOCKERIGNORE.read_text()
        assert ".venv/" in text or "venv/" in text, (
            ".dockerignore must exclude virtual environment directories"
        )


def test_full_container_workflow_execution(tmp_path) -> None:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("docker CLI not available")

    info = subprocess.run(
        [docker, "info"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if info.returncode != 0:
        pytest.skip("docker daemon is unavailable or permission denied")

    image_tag = "mnist-classifier:test"
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    data_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)

    try:
        subprocess.run(
            [docker, "build", "-t", image_tag, str(PROJECT_ROOT)],
            check=True,
            cwd=PROJECT_ROOT,
        )

        subprocess.run(
            [
                docker,
                "run",
                "--rm",
                "-v",
                f"{data_dir}:/app/data",
                "-v",
                f"{results_dir}:/app/results",
                image_tag,
                "--epochs",
                "1",
                "--batch",
                "32",
                "--lr",
                "0.001",
            ],
            check=True,
            cwd=PROJECT_ROOT,
        )

        subprocess.run(
            [
                docker,
                "run",
                "--rm",
                "-v",
                f"{results_dir}:/app/results",
                "--entrypoint",
                "python",
                image_tag,
                "-m",
                "src.analyze",
                "--results",
                "/app/results",
            ],
            check=True,
            cwd=PROJECT_ROOT,
        )
    except subprocess.CalledProcessError as exc:
        pytest.skip(f"docker runtime unavailable for integration execution: {exc}")

    expected = [
        "metrics_train.csv",
        "metrics_validation.csv",
        "metrics_test.csv",
        "run_summary.csv",
        "predictions.csv",
        "model.pt",
        "learning_curves_loss.png",
        "learning_curves_accuracy.png",
        "confusion_matrix.png",
        "classification_report.png",
    ]
    for filename in expected:
        assert (results_dir / filename).exists(), f"Missing artifact: {filename}"
