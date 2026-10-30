#!/bin/zsh
set -e
cd "$(dirname "$0")"
PY=${PY:-python}

[ -f data/raw/with_p3/train.jsonl ] || $PY src/download_data.py
$PY src/etl.py
$PY src/data_quality.py
$PY src/diagrams.py

$PY src/train.py --model codebert_lora   --data with_p3
$PY src/train.py --model distilbert_head --data with_p3    --max-len 256 --epochs 2
$PY src/train.py --model codebert_lora   --data without_p3 --epochs 3
$PY src/train.py --model distilbert_head --data without_p3 --max-len 256 --epochs 3

for d in with_p3 without_p3; do $PY src/baseline.py --data $d; done
$PY src/evaluate.py
echo ALL_DONE
