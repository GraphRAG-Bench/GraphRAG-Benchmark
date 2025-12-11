#!/bin/bash
set -e

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export PYTHONPATH="$(pwd)"

python -m Examples.run_hipporag2 \
  --subset medical \
  --mode API \
  --base_dir 395-scripts/hipporag2/gpt-4.1-nano/medical \
  --model_name gpt-4.1-nano \
  --embed_model_path facebook/contriever \
  --sample -1 \
  --llm_base_url https://api.openai.com/v1
