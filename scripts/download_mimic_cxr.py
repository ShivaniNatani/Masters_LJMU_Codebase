import os
import stat
import sys
import getpass
import subprocess
import tempfile
from urllib.parse import urlparse
from utils.logger import setup_logger

logger = setup_logger("download_mimic_cxr")

PHYSIONET_URL = "https://physionet.org/files/mimic-cxr-jpg/2.0.0/"


def download_mimic_cxr(
    username: str = None,
    password: str = None,
    output_dir: str = "data/raw/mimic_cxr",
):
    """
    Downloads MIMIC-CXR-JPG (v2.0.0) from PhysioNet.
    Requests credentials interactively if not supplied.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("=== PhysioNet Credential Manager & MIMIC-CXR-JPG Downloader ===")
    logger.info("MIMIC-CXR-JPG is a credentialed dataset hosted on PhysioNet.")
    logger.info("URL: https://physionet.org/content/mimic-cxr-jpg/2.0.0/")

    if not username:
        username = input("Enter your PhysioNet username: ").strip()
    if not password:
        password = getpass.getpass("Enter your PhysioNet password: ").strip()

    if not username or not password:
        logger.error("PhysioNet credentials required for MIMIC-CXR access.")
        sys.exit(1)

    # Write credentials to a temporary, owner-only-readable .netrc file rather
    # than passing --password on the command line, which would otherwise be
    # visible to any other user on the host via `ps`/`/proc` and end up in
    # shell history if typed directly.
    host = urlparse(PHYSIONET_URL).netloc
    netrc_fd, netrc_path = tempfile.mkstemp(prefix="mimic_cxr_netrc_")
    try:
        os.chmod(netrc_path, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(netrc_fd, "w") as f:
            f.write(f"machine {host}\nlogin {username}\npassword {password}\n")

        cmd = [
            "wget",
            "-r",
            "-N",
            "-c",
            "-np",
            "--netrc-file",
            netrc_path,
            "-P",
            output_dir,
            PHYSIONET_URL,
        ]

        logger.info(f"Executing PhysioNet download command for user: {username}...")
        try:
            subprocess.run(cmd, check=True)
            logger.info("MIMIC-CXR-JPG dataset download complete.")
        except Exception as e:
            logger.error(f"Download command failed or was interrupted: {e}")
            logger.info("Please verify your PhysioNet credentialing status for MIMIC-CXR-JPG.")
    finally:
        os.remove(netrc_path)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        download_mimic_cxr(sys.argv[1], sys.argv[2])
    else:
        # Prompt interactively if run directly
        logger.info("Run with arguments: python download_mimic_cxr.py <physionet_username> <physionet_password>")
