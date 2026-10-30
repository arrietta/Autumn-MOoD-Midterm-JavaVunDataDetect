import hashlib
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW

BASE = "https://raw.githubusercontent.com/rmusab/vul-llm-finetune/main/Datasets"
ARCHIVES = {
    "with_p3": f"{BASE}/with_p3/java_k_1_strict_2023_06_30.tar.gz",
    "without_p3": f"{BASE}/without_p3/java_k_1_strict_2023_07_03.tar.gz",
}
SHA256 = {
    "with_p3/test.jsonl": "76258098f9f0036daf00184b073502d3d161b322865220b4914e134e940b7895",
    "with_p3/train.jsonl": "9420850dbb55d850fed9d367fde7bf90d604c0877a25abedecd078e47641bbf7",
    "with_p3/valid.jsonl": "1f209e0bd7c82e9b6a006055b04ea3f7ee7f2ca074b46090e21410d465e09417",
    "without_p3/test.jsonl": "9000c317f9ebf9c32c606b24e7c34dbcb67fa38049a46fc9e96e0c5892a1f8fe",
    "without_p3/train.jsonl": "5e7ee29f3014468b0ffbaf3e3186c7e150b04ed7e40a2dce01cd58bb296b354c",
    "without_p3/valid.jsonl": "2fb274ee086cff88633d2b787c3b6deabdd2f96f0a79ceb566be4b0229c7668a",
}


def main():
    for version, url in ARCHIVES.items():
        out = RAW / version
        out.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        data = urllib.request.urlopen(url, timeout=600).read()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
                (out / name).write_bytes(tar.extractfile(name).read())
    for rel, expected in SHA256.items():
        got = hashlib.sha256((RAW / rel).read_bytes()).hexdigest()
        print(f"{'OK  ' if got == expected else 'DIFF'} {rel}")


if __name__ == "__main__":
    main()
