# %%writefile /content/Video-Dubbing/scripts/hf_mirror.py
import os
import time
import requests
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
  from huggingface_hub import snapshot_download
except Exception as e:
  print(e)

def download_file(url, path, redownload=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path) and not redownload:
        if os.path.getsize(path) > 0:
            return f"✔️ Skipped: {os.path.basename(path)}"

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))

            with open(path, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=os.path.basename(path),
                leave=False,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return f"⬇️ Downloaded: {os.path.basename(path)}"

    except Exception as e:
        return f"❌ Failed: {url} ({e})"


def download_model(
    repo_id,
    download_folder="./",
    redownload=False,
    workers=6,
    use_snapshot=True,
):
    start_time = time.time()

    # download_dir = os.path.abspath(
    #     f"{download_folder.rstrip('/')}/{repo_id.split('/')[-1]}"
    # )
    download_dir=download_folder
    os.makedirs(download_dir, exist_ok=True)
    download_dir = os.path.abspath(download_dir)
    print(f"📂 Download directory: {download_dir}")

    # ---------- SNAPSHOT DOWNLOAD ----------
    if use_snapshot:
        try:
            print("🚀 Trying snapshot_download...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=download_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )

            print("✅ Snapshot download successful")
            print(f"\n⏱ Total time: {time.time()-start_time:.2f} sec")
            return download_dir

        except Exception as e:
            print("⚠️ Snapshot failed → fallback to parallel download")
            print("Reason:", e)

    # ---------- FALLBACK PARALLEL DOWNLOAD ----------
    print("🚀 Starting parallel download...")

    api_url = f"https://huggingface.co/api/models/{repo_id}"
    response = requests.get(api_url)
    response.raise_for_status()

    files = [f["rfilename"] for f in response.json().get("siblings", [])]

    print(f"📦 {len(files)} files | Workers: {workers}\n")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []

        for file in files:
            url = f"https://huggingface.co/{repo_id}/resolve/main/{file}"
            path = os.path.join(download_dir, file)

            futures.append(
                executor.submit(download_file, url, path, redownload)
            )

        for future in tqdm(as_completed(futures), total=len(futures), desc="Overall"):
            print(future.result())

    print(f"\n⏱ Total time: {time.time()-start_time:.2f} sec")

    return download_dir
  
# Example usage
# pip install huggingface-hub
# from hf_mirror import download_model
# download_model(
#     "ACE-Step/Ace-Step1.5",
#     download_folder="./model",
#     redownload=True,
#     workers=6,
#     use_snapshot=True,  
# )
