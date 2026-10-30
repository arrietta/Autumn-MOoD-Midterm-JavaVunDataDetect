from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
WAREHOUSE = ROOT / "data" / "warehouse.db"
ML_READY = ROOT / "data" / "ml_ready"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
LOGS = RESULTS / "logs"
MODELS = ROOT / "models"

VERSIONS = ["with_p3", "without_p3"]
SPLITS = {"train": "train", "valid": "val", "test": "test"}
LABEL_OF_STATUS = {"VULNERABLE": 1, "FIXED": 0, "NOT_VULNERABLE": 0}
SEED = 42


def hf(name):
    local = MODELS / "hf" / name.split("/")[-1]
    return str(local) if (local / "config.json").exists() else name


MODELS_CFG = {
    "codebert_lora": {
        "hf_name": hf("microsoft/codebert-base"),
        "lora": {"r": 16, "alpha": 32, "dropout": 0.1, "target_modules": ["query", "value"]},
        "lr": 3e-4, "epochs": 3, "batch_size": 8, "max_len": 512,
    },
    "distilbert_head": {
        "hf_name": hf("distilbert/distilbert-base-uncased"),
        "lora": None,
        "lr": 3e-5, "epochs": 3, "batch_size": 8, "max_len": 512,
    },
}
