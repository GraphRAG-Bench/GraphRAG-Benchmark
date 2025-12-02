#!/bin/bash
set -e

python 395-scripts/fastgraphrag/gpt-4.1-nano/fast-graphrag.py \
  --subset novel \
  --model_name gpt-4.1-nano \
  --embed_model_path BAAI/bge-large-en-v1.5 \
  --llm_base_url https://api.openai.com/v1 \
  --llm_api_key $OPENAI_API_KEY \
  --base_dir 395-scripts/fastgraphrag/gpt-4.1-nano/workspace


