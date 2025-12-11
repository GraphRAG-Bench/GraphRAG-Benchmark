#!/bin/bash
set -e

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export PYTHONPATH="$(pwd)"

python -m Examples.run_hipporag2 \
  --subset novel \
  --mode API \
  --base_dir 395-scripts/hipporag2/gpt-4o-mini/novel \
  --model_name gpt-4o-mini \
  --embed_model_path facebook/contriever \
  --sample -1 \
  --llm_base_url https://api.openai.com/v1
