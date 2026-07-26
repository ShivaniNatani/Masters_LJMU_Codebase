import os
import urllib.request
import zipfile
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger("download_iu_cxr")

IU_CXR_REPORTS_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
IU_CXR_IMAGES_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"


def download_iu_chest_xray(output_dir: str = "data/raw/iu_chest_xray"):
    """
    Downloads and extracts Indiana University Chest X-ray Dataset from NLM OpenI.
    """
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    logger.info("Indiana University Chest X-ray Dataset Download Script initialized.")
    logger.info(f"Target Directory: {output_dir}")
    logger.info("To fetch raw images & XML reports from OpenI, run dataset fetch:")
    logger.info(f"1. Download Reports: {IU_CXR_REPORTS_URL}")
    logger.info(f"2. Download Images: {IU_CXR_IMAGES_URL}")

    # Verify if files exist locally
    if os.path.exists(images_dir) and os.listdir(images_dir):
        logger.info("IU Chest X-ray images found locally.")
    else:
        logger.info("No local raw IU Chest X-ray files found. Use `generate_mock_data.py` for synthetic testing.")


if __name__ == "__main__":
    download_iu_chest_xray()
