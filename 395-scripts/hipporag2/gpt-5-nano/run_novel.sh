#!/bin/bash
#ONLY RUN ON LINUX
set -e

python -m Examples.run_hipporag2 \
  --subset novel \
  --mode API \
  --base_dir 395-scripts/hipporag2/gpt-5-nano/novel \
  --model_name gpt-5-nano \
  --embed_model_path facebook/contriever \
  --sample -1 \
  --llm_base_url https://api.openai.com/v1
