import itertools
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class DefenceSweepTest(unittest.TestCase):
    def test_regular_runner_coefficients_and_checkpoint_reuse(self):
        root = Path(__file__).resolve().parents[1]
        script = (root / "scripts/tamia/run_defence_attack_eval.sbatch").read_text()
        script = script.replace("uv sync --frozen", ":").replace(
            'source "$UV_PROJECT_ENVIRONMENT/bin/activate"', ":"
        )
        python_stub = """
python() {
    if [[ $1 == -c ]]; then printf 'sp\\n'; return; fi
    printf '%s\\0' "$@"
    exit 0
}
"""
        for reuse in [False, True]:
            with self.subTest(reuse=reuse), tempfile.TemporaryDirectory() as directory:
                model = "artifacts/tamia/models/model--def-sp--alpha0.1--beta0.0001"
                if reuse:
                    checkpoint = Path(directory) / model
                    checkpoint.mkdir(parents=True)
                    (checkpoint / ".trained").touch()
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        python_stub + script,
                        "runner",
                        "model",
                        " ",
                        "0.1",
                        "0.0001",
                    ],
                    cwd=directory,
                    env=os.environ
                    | {
                        "SLURM_JOB_ID": "123",
                        "SLURM_TMPDIR": directory,
                        "SCRATCH": directory,
                    },
                    capture_output=True,
                    check=True,
                )
                args = result.stdout.decode().split("\0")
                if reuse:
                    self.assertEqual(
                        args[:2], ["-m", "representation_noising.evaluation"]
                    )
                    self.assertEqual(args[args.index("--model") + 1], model)
                else:
                    self.assertEqual(args[:2], ["-m", "representation_noising.train"])
                    self.assertEqual(args[args.index("--alpha") + 1], "0.1")
                    self.assertEqual(args[args.index("--beta") + 1], "0.0001")
                    self.assertEqual(args[args.index("--output-dir") + 1], model)

    def test_array_selects_and_passes_each_coefficient_pair(self):
        root = Path(__file__).resolve().parents[1]
        script = (root / "scripts/tamia/run_defence_sweep.sbatch").read_text()
        self.assertIn("#SBATCH --array=0-8", script)
        # Stop at the defense invocation; no environment setup or training is run.
        script = script.replace("uv sync --frozen", ":").replace(
            'source "$UV_PROJECT_ENVIRONMENT/bin/activate"', ":"
        )
        python_stub = """
python() {
    if [[ $1 == -c ]]; then printf 'nl\\n'; return; fi
    printf '%s\\0' "$@"
    exit 0
}
"""
        pairs = itertools.product(["0.1", "1.0", "10.0"], ["0.0001", "0.001", "0.01"])
        for index, (alpha, beta) in enumerate(pairs):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run(
                    ["bash", "-c", python_stub + script, "sweep", "model", "\n"],
                    cwd=directory,
                    env=os.environ
                    | {
                        "SLURM_ARRAY_TASK_ID": str(index),
                        "SLURM_JOB_ID": "123",
                        "SLURM_TMPDIR": directory,
                        "SCRATCH": directory,
                    },
                    capture_output=True,
                    check=True,
                )
                args = result.stdout.decode().split("\0")
                self.assertEqual(args[args.index("--alpha") + 1], alpha)
                self.assertEqual(args[args.index("--beta") + 1], beta)
                self.assertEqual(
                    args[args.index("--harmful-response-separator") + 1], "\n"
                )
                self.assertEqual(
                    args[args.index("--output-dir") + 1],
                    f"artifacts/tamia/models/model--def-nl--alpha{alpha}--beta{beta}",
                )
