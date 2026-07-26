import importlib
import sys
import torch
import yaml
from utils.logger import setup_logger

logger = setup_logger("env_check")


def verify_environment() -> dict:
    """
    Verifies all required libraries, CUDA / GPU status, and system hardware.
    Returns status report dict.
    """
    report = {}

    # Hardware & PyTorch
    report["python_version"] = sys.version
    report["torch_version"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()

    if torch.cuda.is_available():
        report["cuda_version"] = torch.version.cuda
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["gpu_count"] = torch.cuda.device_count()
        report["compute_backend"] = "CUDA"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        report["gpu_name"] = "Apple Silicon MPS"
        report["gpu_count"] = 1
        report["compute_backend"] = "MPS"
    else:
        report["gpu_name"] = "None"
        report["gpu_count"] = 0
        report["compute_backend"] = "CPU"

    # Core Dependencies Verification
    required_packages = [
        "transformers",
        "faiss",
        "cv2",
        "pandas",
        "sklearn",
        "wandb",
        "mlflow",
        "yaml",
        "PIL",
    ]

    package_statuses = {}
    for pkg in required_packages:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "Available")
            package_statuses[pkg] = {"installed": True, "version": ver}
        except ImportError as e:
            package_statuses[pkg] = {"installed": False, "error": str(e)}

    report["packages"] = package_statuses

    logger.info("=== ENVIRONMENT VERIFICATION REPORT ===")
    logger.info(f"Compute Backend: {report['compute_backend']} ({report.get('gpu_name')})")
    logger.info(f"PyTorch Version: {report['torch_version']}")
    for pkg, info in package_statuses.items():
        status = f"v{info['version']}" if info["installed"] else f"MISSING ({info['error']})"
        logger.info(f" - {pkg}: {status}")

    return report


if __name__ == "__main__":
    verify_environment()
