"""General-purpose Slurm job submission with tag-based tracking and cancel-resubmit."""

import logging
import os
import subprocess
import tempfile

log = logging.getLogger(__name__)


class SlurmJobManager:
    """Submit and track Slurm jobs by tag, with cancel-and-resubmit logic.

    Usage::

        mgr = SlurmJobManager()
        mgr.submit("python train.py", tag="eval", partition="standard,preemptible")
        # Later — cancels the old job if still active, submits new one:
        mgr.submit("python train.py --step=2000", tag="eval", cancel_pending=True)
    """

    def __init__(self):
        self._jobs: dict[str, int] = {}  # tag -> job_id

    def submit(
        self,
        command: str,
        tag: str | None = None,
        partition: str = "standard,preemptible",
        gres: str = "gpu:1",
        mem: str = "64G",
        time: str = "01:00:00",
        job_name: str | None = None,
        cancel_pending: bool = True,
        env: dict[str, str] | None = None,
        work_dir: str | None = None,
    ) -> int | None:
        """Submit *command* via ``sbatch``.

        If *cancel_pending* and a job with the same *tag* is still active
        (PENDING or RUNNING), cancel it first so we always evaluate the
        freshest checkpoint.

        Returns the Slurm job ID, or ``None`` on failure.
        """
        if tag and cancel_pending and tag in self._jobs:
            self.cancel(tag)

        job_name = job_name or tag or "slurm_job"
        work_dir = work_dir or os.getcwd()

        script = self._build_script(
            command, partition=partition, gres=gres, mem=mem,
            time=time, job_name=job_name, env=env, work_dir=work_dir,
        )

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".sbatch", delete=False, dir=work_dir,
            ) as f:
                f.write(script)
                script_path = f.name

            result = subprocess.run(
                ["sbatch", script_path],
                capture_output=True, text=True, timeout=30,
            )

            # Clean up temp script
            try:
                os.unlink(script_path)
            except OSError:
                pass

            if result.returncode != 0:
                log.warning(f"sbatch failed: {result.stderr.strip()}")
                return None

            # Parse "Submitted batch job 12345"
            job_id = int(result.stdout.strip().split()[-1])
            if tag:
                self._jobs[tag] = job_id
            log.info(f"Submitted Slurm job {job_id} (tag={tag})")
            return job_id

        except Exception as e:
            log.warning(f"Failed to submit Slurm job: {e}")
            return None

    def cancel(self, tag: str) -> bool:
        """Cancel the active job for *tag*. Returns True if cancelled."""
        job_id = self._jobs.get(tag)
        if job_id is None:
            return False

        if not self._job_is_active(job_id):
            del self._jobs[tag]
            return False

        try:
            subprocess.run(
                ["scancel", str(job_id)],
                capture_output=True, timeout=10,
            )
            log.info(f"Cancelled Slurm job {job_id} (tag={tag})")
        except Exception as e:
            log.warning(f"Failed to cancel job {job_id}: {e}")

        del self._jobs[tag]
        return True

    def _job_is_active(self, job_id: int) -> bool:
        """Check via ``squeue`` whether *job_id* is PENDING or RUNNING."""
        try:
            result = subprocess.run(
                ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
                capture_output=True, text=True, timeout=10,
            )
            state = result.stdout.strip()
            return state in ("PENDING", "RUNNING")
        except Exception:
            return False

    @staticmethod
    def _build_script(
        command: str,
        partition: str,
        gres: str,
        mem: str,
        time: str,
        job_name: str,
        env: dict[str, str] | None,
        work_dir: str,
    ) -> str:
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={partition}",
            f"#SBATCH --gres={gres}",
            f"#SBATCH --mem={mem}",
            f"#SBATCH --time={time}",
            f"#SBATCH --output={work_dir}/slurm_logs/%x_%j.out",
            f"#SBATCH --error={work_dir}/slurm_logs/%x_%j.err",
            "",
            f"cd {work_dir}",
            "",
        ]
        if env:
            for k, v in env.items():
                lines.append(f"export {k}={v}")
            lines.append("")
        lines.append(command)
        lines.append("")
        return "\n".join(lines)
