import requests

url = "https://huggingface.co/datasets/chehablab/NIHChestXR/resolve/main/data/train-00000-of-00006.parquet"
path = "outputs/xray_labeled_evaluation/source/train-00000-of-00006.parquet"
response = requests.get(url, stream=True, timeout=120)
response.raise_for_status()
total = 0
with open(path, "wb") as handle:
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        if chunk:
            handle.write(chunk)
            total += len(chunk)
print("downloaded", total, "bytes")
