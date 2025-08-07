# toolgrad/utils/drive_utils.py

import os
import sys
import zipfile
from pathlib import Path

import gdown


def get_cache_dir() -> Path:
  """Return a Path pointing to the cache directory."""
  cache_env = os.getenv("TOOLGRAD_CACHE")
  if cache_env:
    cache_path = Path(cache_env).expanduser()
  else:
    cache_path = Path.home() / ".toolgrad_cache"
  cache_path.mkdir(parents=True, exist_ok=True)
  return cache_path


def download_and_extract_from_drive(
    drive_file_id: str,
    extract: bool = True,
) -> None:
  """Download a file from Google Drive.
  
  If `archive_name` ends with '.zip' (or extract=True), unzips it.
  Skips work if dest_dir already exists and is non‐empty.

  Args:
    drive_file_id: The Google Drive file ID (e.g. '1AbCDeFGhIJKLMnopQRstuVWXYz123456').
    extract: If True, assumes the downloaded file is a zip and unpacks it into dest_dir.
  """
  dest_dir = get_cache_dir() / drive_file_id

  dest_dir.mkdir(parents=True, exist_ok=True)
  parent = dest_dir.parent

  archive_name = f"{drive_file_id}.zip"
  if not archive_name.endswith(".zip") and extract:
    archive_name = archive_name + ".zip"

  archive_path = parent / archive_name

  url = f"https://drive.google.com/uc?id={drive_file_id}"
  print(f"[drive_utils] Downloading → {archive_path}", file=sys.stderr)
  gdown.download(url, str(archive_path), quiet=False)

  if extract:
    print(f"[drive_utils] Extracting {archive_path} → {dest_dir}",
          file=sys.stderr)
    try:
      with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(path=str(dest_dir))
    except zipfile.BadZipFile:
      raise RuntimeError(
          f"Failed to unzip {archive_path}; file may be corrupt.")
