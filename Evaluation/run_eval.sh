export PYTHONPATH=$(pwd)

python Evaluation/generation_eval.py \
    --mode API \
    --model gpt-4o-mini \
    --base_url https://api.openai.com/v1 \
    --embedding_model BAAI/bge-large-en-v1.5 \
    --data_file 395-scripts/lightrag/gpt-4.1-nano/results/medical_results_41_nano_light.json \
    --output_file 395-scripts/lightrag/gpt-4.1-nano/results/medical_eval.json
