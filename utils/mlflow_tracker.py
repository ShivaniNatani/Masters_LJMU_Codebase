import os
import subprocess
import mlflow
from typing import Dict, Any
from utils.logger import setup_logger

logger = setup_logger("mlflow_tracker")


def get_git_commit_hash() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        return commit
    except Exception:
        return "UNKNOWN_GIT_COMMIT"


class MLflowTracker:
    """
    MLflow Experiment Tracker managing runs, metrics, parameters, and artifact logging.
    Uses SQLite database backend to maintain full compatibility.
    """

    def __init__(
        self,
        experiment_name: str = "Baseline_BioMedCLIP_FLAN_T5",
        db_path: str = "logs/mlflow.db",
    ):
        self.experiment_name = experiment_name
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

        db_uri = f"sqlite:///{os.path.abspath(db_path)}"
        mlflow.set_tracking_uri(db_uri)
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str, params: Dict[str, Any]):
        mlflow.start_run(run_name=run_name)
        mlflow.log_param("git_commit_hash", get_git_commit_hash())
        for k, v in params.items():
            mlflow.log_param(k, str(v))
        logger.info(f"MLflow Run Started: {run_name} (Experiment: {self.experiment_name})")

    def log_metrics(self, metrics: Dict[str, float], step: int):
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v), step=step)

    def log_artifact(self, local_path: str):
        if os.path.exists(local_path):
            mlflow.log_artifact(local_path)

    def end_run(self):
        mlflow.end_run()
        logger.info("MLflow Run Ended.")
