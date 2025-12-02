python 395-scripts/fastgraphrag/qwen3_8b/fast-graphrag_ollama.py \
    --subset medical \
    --model_name qwen3:8b \
    --sample 3 \
    --embed_model_path BAAI/bge-large-en-v1.5 \
    --llm_base_url http://127.0.0.1:11434/v1 \
    --llm_api_key ollama \
    --base_dir ./Examples/graphrag_qwen3_8b_medical
