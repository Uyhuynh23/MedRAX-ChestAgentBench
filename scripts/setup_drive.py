"""
Google Drive Setup & Persistent Storage Initializer for MedRAX on Colab
------------------------------------------------------------------------
This script configures Google Drive persistent directories for:
  1. Pretrained model checkpoints (Maira-2, CheXagent, LLaVA-Med, etc.)
  2. Hugging Face hub cache (HF_HOME)
  3. ChestAgentBench dataset (metadata.jsonl + unzipped figures/)
  4. Reproduction evaluation outputs, logs, and ACE playbooks

Usage in Google Colab:
  python scripts/setup_drive.py --drive-folder "MedRAX_Thesis"
"""

import os
import sys
import argparse
from pathlib import Path

def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

def mount_drive():
    if is_colab():
        print("[Drive Setup] Running in Google Colab environment.")
        from google.colab import drive
        mount_point = "/content/drive"
        if not os.path.exists(mount_point) or len(os.listdir(mount_point)) == 0:
            print("[Drive Setup] Mounting Google Drive to /content/drive ...")
            drive.mount(mount_point)
            print("[Drive Setup] Google Drive mounted successfully.")
        else:
            print("[Drive Setup] Google Drive is already mounted.")
        return mount_point
    else:
        print("[Drive Setup] Not running in Google Colab. Using local persistent directory.")
        return str(Path.home())

def init_drive_directories(base_drive_dir: str, folder_name: str = "MedRAX_Thesis"):
    """
    Creates persistent directory structure in Google Drive and symlinks them to current working dir.
    """
    if is_colab():
        drive_project_root = Path("/content/drive/MyDrive") / folder_name
    else:
        drive_project_root = Path(base_drive_dir) / folder_name

    subdirs = {
        "checkpoints": drive_project_root / "checkpoints",
        "hf_cache": drive_project_root / "cache" / "huggingface",
        "torch_cache": drive_project_root / "cache" / "torch",
        "data": drive_project_root / "data" / "chestagentbench",
        "results": drive_project_root / "results",
        "logs": drive_project_root / "logs",
    }

    print(f"\n=======================================================")
    print(f" Initializing Persistent Google Drive Architecture")
    print(f" Root: {drive_project_root}")
    print(f"=======================================================")

    for name, path in subdirs.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] Directory ready: {path}")

    # Set Environment Variables
    os.environ["HF_HOME"] = str(subdirs["hf_cache"])
    os.environ["TRANSFORMERS_CACHE"] = str(subdirs["hf_cache"])
    os.environ["TORCH_HOME"] = str(subdirs["torch_cache"])
    os.environ["MEDRAX_MODEL_DIR"] = str(subdirs["checkpoints"])
    os.environ["BENCHMARK_DATA_DIR"] = str(subdirs["data"])
    os.environ["MEDRAX_RESULTS_DIR"] = str(subdirs["results"])

    # Setup Symlinks into current workspace for transparent access
    cwd = Path.cwd()
    link_mappings = {
        cwd / "checkpoints": subdirs["checkpoints"],
        cwd / "cache": drive_project_root / "cache",
        cwd / "data" / "chestagentbench": subdirs["data"],
        cwd / "results": subdirs["results"],
        cwd / "logs": subdirs["logs"],
    }

    print(f"\nSetting up local workspace symlinks:")
    for local_link, target_dir in link_mappings.items():
        local_link.parent.mkdir(parents=True, exist_ok=True)
        if not local_link.exists():
            try:
                local_link.symlink_to(target_dir, target_is_directory=True)
                print(f"  [Symlinked] {local_link} -> {target_dir}")
            except Exception as e:
                print(f"  [Notice] Could not create symlink ({e}). Scripts will use direct Drive paths.")
        else:
            print(f"  [Exists] {local_link}")

    print(f"\nPersistent Environment Variables Configured:")
    print(f"  export HF_HOME='{subdirs['hf_cache']}'")
    print(f"  export MEDRAX_MODEL_DIR='{subdirs['checkpoints']}'")
    print(f"  export BENCHMARK_DATA_DIR='{subdirs['data']}'")
    print(f"  export MEDRAX_RESULTS_DIR='{subdirs['results']}'")
    print(f"=======================================================\n")

    return subdirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize Google Drive structure for MedRAX")
    parser.add_argument("--drive-folder", type=str, default="MedRAX_Thesis", help="Target folder in Google Drive")
    args = parser.parse_args()

    mount_point = mount_drive()
    paths = init_drive_directories(mount_point, folder_name=args.drive_folder)
