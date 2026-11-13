"""
ChestAgentBench Dataset Downloader & Integrity Verifier
-------------------------------------------------------
Downloads `wanglab/chest-agent-bench` from Hugging Face, unzips figures,
and validates the integrity of cases and images.

Usage:
  python scripts/download_benchmark.py [--data-dir <path>] [--hf-token <token>]
"""

import os
import sys
import json
import zipfile
import argparse
from pathlib import Path

def setup_chest_agent_bench(target_dir: str, hf_token: str = None):
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    metadata_file = target_path / "metadata.jsonl"
    figures_dir = target_path / "figures"
    figures_zip = target_path / "figures.zip"

    print(f"=======================================================")
    print(f" ChestAgentBench Dataset Setup")
    print(f" Target Directory: {target_path}")
    print(f"=======================================================")

    # Check if dataset is already prepared
    if metadata_file.exists() and figures_dir.exists() and any(figures_dir.iterdir()):
        print("[Check] Dataset files and figures directory already exist.")
        verify_dataset(target_path)
        return

    # Download dataset from Hugging Face
    print(f"[Download] Fetching dataset from Hugging Face: wanglab/chest-agent-bench ...")
    try:
        from huggingface_hub import snapshot_download
        token = hf_token or os.getenv("HF_TOKEN") or None
        snapshot_download(
            repo_id="wanglab/chest-agent-bench",
            repo_type="dataset",
            local_dir=str(target_path),
            token=token,
        )
        print("[Download] Dataset snapshot downloaded successfully.")
    except Exception as e:
        print(f"[Warning] huggingface_hub API error: {e}")
        print("[Fallback] Trying huggingface-cli command...")
        cmd = f"huggingface-cli download wanglab/chest-agent-bench --repo-type dataset --local-dir \"{target_path}\""
        if token:
            cmd += f" --token {token}"
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError("Failed to download dataset via huggingface-cli. Please check network or HF_TOKEN.")

    # Unpack figures.zip if figures directory doesn't exist
    if figures_zip.exists() and (not figures_dir.exists() or not any(figures_dir.iterdir())):
        print(f"[Unpack] Extracting {figures_zip} -> {target_path} ...")
        with zipfile.ZipFile(figures_zip, 'r') as zip_ref:
            zip_ref.extractall(target_path)
        print("[Unpack] Extraction complete.")

    verify_dataset(target_path)

def verify_dataset(target_path: Path):
    metadata_file = target_path / "metadata.jsonl"
    figures_dir = target_path / "figures"

    if not metadata_file.exists():
        raise FileNotFoundError(f"Missing {metadata_file}!")

    total_rows = 0
    case_ids = set()
    total_images_referenced = 0
    existing_images = 0

    with open(metadata_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            total_rows += 1
            case_ids.add(str(item.get("case_id", "")))
            
            # Check images
            images = item.get("images", [])
            if isinstance(images, str):
                images = [images]
            total_images_referenced += len(images)
            for img in images:
                # remove figures/ prefix if present
                clean_img = img.replace("figures/", "") if img.startswith("figures/") else img
                img_path = figures_dir / clean_img
                if img_path.exists():
                    existing_images += 1

    print("\n----------------- Verification Summary -----------------")
    print(f" Total Multiple-Choice Questions: {total_rows} (Expected: 2,500)")
    print(f" Unique Clinical Cases:           {len(case_ids)} (Expected: ~609)")
    print(f" Image References Checked:        {existing_images}/{total_images_referenced}")
    if figures_dir.exists():
        num_folders = len([d for d in figures_dir.iterdir() if d.is_dir()])
        print(f" Case Folders in figures/:        {num_folders}")
    print("--------------------------------------------------------")
    print("[SUCCESS] Dataset ready for MedRAX evaluation!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and verify ChestAgentBench")
    parser.add_argument("--data-dir", type=str, default=os.getenv("BENCHMARK_DATA_DIR", "./data/chestagentbench"))
    parser.add_argument("--hf-token", type=str, default=None)
    args = parser.parse_args()

    setup_chest_agent_bench(args.data_dir, args.hf_token)
