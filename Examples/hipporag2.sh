set -e
[ -z "$OPENAI_API_KEY" ] && echo "set OPENAI_API_KEY first" && exit 1

pyenv local 3.11.10
source .venv/bin/activate

python -m pip install -U pip
pip install -r requirements.txt
pip install "torch>=2.2,<3"
pip install "transformers>=4.40,<5" tokenizers accelerate huggingface_hub

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

python Examples/run_hipporag2.py \
  --subset medical \
  --mode API \
  --base_dir ./Examples/hipporag2_workspace \
  --model_name gpt-5-nano \
  --embed_model_path facebook/contriever \
  --sample 2 \
  --llm_base_url https://api.openai.com/v1
