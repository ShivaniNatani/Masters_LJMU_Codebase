import os
import sys
import json
import torch
import pandas as pd

# Add repository root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.seed import set_seed
from utils.logger import setup_logger
from utils.env_check import verify_environment
from scripts.generate_mock_data import generate_mock_dataset
from datasets.base_dataset import BaseMedicalDataset
from datasets.data_loader import create_dataloader
from models.mock_vlm import MockVLM

logger = setup_logger("smoke_test")


def run_smoke_test(output_report_path: str = "results/verification_report.json") -> bool:
    """
    End-to-End Verification Smoke Test:
    1. System Environment & GPU Verification
    2. Mock Dataset Generation & CSV Load Verification
    3. Image Loading Verification
    4. Report Loading Verification
    5. PyTorch DataLoader Verification
    6. Model Forward Pass Verification
    """
    set_seed(42)
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    report = {"status": "FAILED", "checks": {}}

    logger.info("==================================================")
    logger.info("      STARTING PHASE 1 SMOKE TEST VERIFICATION     ")
    logger.info("==================================================")

    try:
        # Step 1: Environment & GPU Check
        logger.info("[Check 1/6] Environment & Hardware Detection...")
        env_status = verify_environment()
        device_str = env_status["compute_backend"]
        if device_str == "CUDA":
            device = torch.device("cuda:0")
        elif device_str == "MPS":
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        report["checks"]["environment"] = {
            "passed": True,
            "device": str(device),
            "backend": device_str,
            "gpu_name": env_status.get("gpu_name", "N/A"),
        }
        logger.info(f" -> Active Device: {device} ({env_status.get('gpu_name')})")

        # Step 2: Dataset Load Check
        logger.info("[Check 2/6] Dataset Loading...")
        mock_csv = "data/mock/mimic_cxr_mock.csv"
        if not os.path.exists(mock_csv):
            mock_csv = generate_mock_dataset(num_samples=20)

        df = pd.read_csv(mock_csv)
        assert len(df) > 0, "Dataset CSV is empty"
        report["checks"]["dataset_load"] = {"passed": True, "num_records": len(df)}
        logger.info(f" -> Loaded CSV with {len(df)} records.")

        # Step 3: PyTorch Dataset & Image Load Check
        logger.info("[Check 3/6] Image & Report Processing...")
        dataset = BaseMedicalDataset(df, image_size=(224, 224), max_seq_len=64)
        sample_item = dataset[0]

        img_tensor = sample_item["image"]
        report_text = sample_item["report_text"]
        input_ids = sample_item["input_ids"]

        assert isinstance(img_tensor, torch.Tensor), "Image is not a Tensor"
        assert img_tensor.shape == (3, 224, 224), f"Unexpected image shape: {img_tensor.shape}"
        assert len(report_text) > 0, "Report text is empty"

        report["checks"]["image_load"] = {"passed": True, "shape": list(img_tensor.shape)}
        report["checks"]["report_load"] = {"passed": True, "sample_text_snippet": report_text[:50]}
        logger.info(" -> Image & Report parsing verified.")

        # Step 4: DataLoader Check
        logger.info("[Check 4/6] DataLoader Batching...")
        dataloader = create_dataloader(dataset, batch_size=4, shuffle=False)
        batch = next(iter(dataloader))

        batch_images = batch["image"]
        batch_ids = batch["input_ids"]

        assert batch_images.shape == (4, 3, 224, 224), f"DataLoader image batch shape error: {batch_images.shape}"
        assert batch_ids.shape == (4, 64), f"DataLoader input_ids batch shape error: {batch_ids.shape}"

        report["checks"]["dataloader"] = {"passed": True, "batch_size": 4}
        logger.info(" -> DataLoader batching verified.")

        # Step 5: GPU / MPS Transfer & Forward Pass
        logger.info("[Check 5/6] Model Forward Pass Execution...")
        model = MockVLM(vision_dim=512, text_dim=256, vocab_size=1000).to(device)
        model.eval()

        b_images = batch_images.to(device)
        b_ids = batch_ids.to(device)

        with torch.no_grad():
            outputs = model(b_images, b_ids)

        logits = outputs["logits"]
        loss = outputs["loss"]

        assert logits.shape == (4, 64, 1000), f"Unexpected logits shape: {logits.shape}"
        assert loss is not None, "Loss computation returned None"

        report["checks"]["forward_pass"] = {
            "passed": True,
            "logits_shape": list(logits.shape),
            "loss_val": float(loss.item()),
        }
        logger.info(f" -> Forward pass succeeded! Loss: {loss.item():.4f}, Logits shape: {logits.shape}")

        # Step 6: Final Status
        report["status"] = "PASSED"
        logger.info("==================================================")
        logger.info("   PHASE 1 SMOKE TEST PASSED SUCCESSFULLY!        ")
        logger.info("==================================================")

    except Exception as e:
        logger.error(f"Smoke Test Failed with exception: {e}", exc_info=True)
        report["status"] = "FAILED"
        report["error"] = str(e)

    # Export report JSON
    with open(output_report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Verification report saved to {output_report_path}")
    return report["status"] == "PASSED"


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
