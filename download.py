import requests
import os

base_url = "https://huggingface.co/datasets/TommyChien/UltraDomain/resolve/main/"

files = [
    "agriculture.jsonl",
    "cs.jsonl",
    "legal.jsonl",
    "mix.jsonl",
]

os.makedirs("raw_data", exist_ok=True)

for file in files:
    url = base_url + file
    save_path = os.path.join("raw_data", file)

    print(f"Downloading {file}...")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print(f"✅ Saved: {save_path}")
