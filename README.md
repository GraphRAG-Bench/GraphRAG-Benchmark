<div align="center">

# GraphRAG-Bench : A Comprehensive Benchmark for Evaluating Graph Retrieval-Augmented Generation Models

[![Static Badge](https://img.shields.io/badge/arxiv-2506.05690-ff0000?style=for-the-badge&labelColor=000)](https://arxiv.org/abs/2506.05690)  [![Static Badge](https://img.shields.io/badge/huggingface-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench)  [![Static Badge](https://img.shields.io/badge/leaderboard-steelblue?style=for-the-badge&logo=googlechrome&logoColor=ffffff)](https://graphrag-bench.github.io/)  [![Static Badge](https://img.shields.io/badge/license-mit-teal?style=for-the-badge&labelColor=000)](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/blob/main/LICENSE)

<p>
    <a href="#news" style="text-decoration: none; font-weight: bold;">🎉News</a> •
    <a href="#about" style="text-decoration: none; font-weight: bold;">📖About</a> •
    <a href="#leaderboards" style="text-decoration: none; font-weight: bold;">🏆Leaderboards</a> •
    <a href="#task-examples" style="text-decoration: none; font-weight: bold;">🧩Task Examples</a>

</p>
  <p>
  <a href="#getting-started" style="text-decoration: none; font-weight: bold;">🔧Getting Started</a> •
    <a href="#contribution--contact" style="text-decoration: none; font-weight: bold;">📬Contact</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">📝Citation</a> •
    <a href="#stars" style="text-decoration: none; font-weight: bold;">✨Stars History</a>
  </p>
</div>

This repository is for the GraphRAG-Bench project, a comprehensive benchmark for evaluating Graph Retrieval-Augmented Generation models.
![pipeline](./pipeline.jpg)

<h2 id="news">🎉 News</h2>

- **[2025-05-25]** We release [GraphRAG-Bench](https://graphrag-bench.github.io), the benchmark for evaluating GraphRAG models.
- **[2025-05-14]** We release the [GraphRAG-Bench dataset](https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench).
- **[2025-01-21]** We release the [GraphRAG survey](https://github.com/DEEP-PolyU/Awesome-GraphRAG).

<h2 id="about">📖 About</h2>

- Introduces Graph Retrieval-Augmented Generation (GraphRAG) concept
- Compares traditional RAG vs GraphRAG approach
- Explains research objective: Identify scenarios where GraphRAG outperforms traditional RAG
- Visual comparison diagram of RAG vs GraphRAG

![overview](./RAGvsGraphRAG.jpg)

<details>
<summary>
  More Details
</summary>
Graph retrieval-augmented generation (GraphRAG) has emerged as a powerful paradigm for enhancing large language models (LLMs) with external knowledge. It leverages graphs to model the hierarchical structure between specific concepts, enabling more coherent and effective knowledge retrieval for accurate reasoning. Despite its conceptual promise, recent studies report that GraphRAG frequently underperforms vanilla RAG on many real-world tasks. This raises a critical question: Is GraphRAG really effective, and in which scenarios do graph structures provide measurable benefits for RAG systems? To address this, we propose GraphRAG-Bench, a comprehensive benchmark designed to evaluate GraphRAG models on both hierarchical knowledge retrieval and deep contextual reasoning. GraphRAG-Bench features a comprehensive dataset with tasks of increasing difficulty, covering fact retrieval, complex reasoning, contextual summarization, and creative generation, and a systematic evaluation across the entire pipeline, from graph construction and knowledge retrieval to final generation. Leveraging this novel benchmark, we systematically investigate the conditions when GraphRAG surpasses traditional RAG and the underlying reasons for its success, offering guidelines for its practical application.
</details>

<h2 id="leaderboards">🏆 Leaderboards</h2>

Two domain-specific leaderboards with comprehensive metrics:

**1. GraphRAG-Bench (Novel)**

- Evaluates models on literary/fictional content

**2. GraphRAG-Bench (Medical)**

- Evaluates models on medical/healthcare content

**Evaluation Dimensions:**

- Fact Retrieval (Accuracy, ROUGE-L)
- Complex Reasoning (Accuracy, ROUGE-L)
- Contextual Summarization (Accuracy, Coverage)
- Creative Generation (Accuracy, Factual Score, Coverage)

<h2 id="task-examples">🧩 Task Examples</h2>
Four difficulty levels with representative examples:

**Level 1: Fact Retrieval**
*Example: "Which region of France is Mont St. Michel located?"*

**Level 2: Complex Reasoning**
*Example: "How did Hinze's agreement with Felicia relate to the perception of England's rulers?"*

**Level 3: Contextual Summarization**
*Example: "What role does John Curgenven play as a Cornish boatman for visitors exploring this region?"*

**Level 4: Creative Generation**
*Example: "Retell King Arthur's comparison to John Curgenven as a newspaper article."*

<h2 id="getting-started">🔧 Getting Started(GraphRAG-Bench Examples)</h2>

First, install the necessary dependencies for GraphRAG-Bench.

```bash
pip install -r requirements.txt
```

## 🛠 Installation Guide

**To prevent dependency conflicts, we strongly recommend using separate Conda environments for each framework:**

We use the installation of LightRAG as an example. For other frameworks, please refer to their respective installation instructions.

```bash
# Create and activate environment (example for LightRAG)
conda create -n lightrag python=3.10 -y
conda activate lightrag

# Install LightRAG
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG
pip install -e .

```

## 🚀 Running Example
Next, we provide detailed instructions on how to use GraphRAG-Bench to evaluate each framework. Specifically, we introduce how to perform index construction and batch inference for each framework. Note that the evaluation code is standardized across all frameworks to ensure fair comparison.
### 1. Indexing and inference
#### a. LightRAG
**We use LightRAG version v1.2.5.**

Before running the above script, you need to modify the source code(LightRAG) to enable extraction of the corresponding context used during generation. Please make the following changes:
1. In `lightrag/operate.py`, update the kg_query method to return the context along with the response:

```python
# Original Code
async def kg_query(...) -> str | AsyncIterator[str]:
  return response

# Modified Code
async def kg_query(...) -> tuple[str, str] | tuple[AsyncIterator[str], str]:
  return response, context
```

2. In `lightrag/lightrag.py`, update the aquery method to receive and return the context when calling kg_query:

```python
# Modified Code
async def aquery(...):
  ...
  if param.mode in ["local", "global", "hybrid"]:
      response, context = await kg_query(...)
  ...
  return response, context

```

Then you can run the following command to indexing and inference

Note: Mode can choose:"API" or "ollama" and llm_base_url is where your ollama running(default: http://localhost:11434):

```shell
export LLM_API_KEY=your_actual_api_key_here

python run_lightrag.py \
  --subset medical \
  --mode API \
  --base_dir ./Examples/lightrag_workspace \
  --model_name bge-large-en-v1.5 \
  --embed_model bge-base-en \
  --retrieve_topk 5 \
#   --sample 100 \
  --llm_base_url https://api.openai.com/v1

```
#### b. fast-graphrag
Since the original fast-LightRAG does not support HuggingFace Embedding, we need to adapt the library accordingly. The detailed adaptation process is as follows:
1. Go to the `fast_graphrag/_llm` directory and create a new file named _hf.py.
The content of this file is as follows. This code mainly adds support for HuggingFace Embedding:

```python
import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from aiolimiter import AsyncLimiter
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from fast_graphrag._utils import logger
from fast_graphrag._llm._base import BaseEmbeddingService, NoopAsyncContextManager

@dataclass
class HuggingFaceEmbeddingService(BaseEmbeddingService):
    """Embedding service using HuggingFace models."""

    embedding_dim: Optional[int] = None  # Can be set dynamically if needed
    max_token_size: int = 512
    max_elements_per_request: int = field(default=32)
    tokenizer: Any = None
    model: Any = None

    def __post_init__(self):
        self.embedding_max_requests_concurrent = (
            asyncio.Semaphore(self.max_requests_concurrent) if self.rate_limit_concurrency else NoopAsyncContextManager()
        )
        self.embedding_per_minute_limiter = (
            AsyncLimiter(self.max_requests_per_minute, 60) if self.rate_limit_per_minute else NoopAsyncContextManager()
        )
        self.embedding_per_second_limiter = (
            AsyncLimiter(self.max_requests_per_second, 1) if self.rate_limit_per_second else NoopAsyncContextManager()
        )
        logger.debug("Initialized HuggingFaceEmbeddingService.")

    async def encode(self, texts: list[str], model: Optional[str] = None) -> np.ndarray:
        try:
            logger.debug(f"Getting embedding for texts: {texts}")

            batched_texts = [
                texts[i * self.max_elements_per_request : (i + 1) * self.max_elements_per_request]
                for i in range((len(texts) + self.max_elements_per_request - 1) // self.max_elements_per_request)
            ]
            responses = await asyncio.gather(*[self._embedding_request(batch) for batch in batched_texts])
            embeddings = np.vstack(responses)
            logger.debug(f"Received embedding response: {len(embeddings)} embeddings")
            return embeddings
        except Exception:
            logger.exception("An error occurred during HuggingFace embedding.", exc_info=True)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RuntimeError, torch.cuda.CudaError)),
    )
    async def _embedding_request(self, input_texts: list[str]) -> np.ndarray:
        async with self.embedding_max_requests_concurrent:
            async with self.embedding_per_minute_limiter:
                async with self.embedding_per_second_limiter:
                    logger.debug(f"Embedding request for batch size: {len(input_texts)}")
                    device = (
                        next(self.model.parameters()).device if torch.cuda.is_available()
                        else torch.device("mps") if torch.backends.mps.is_available()
                        else torch.device("cpu")
                    )
                    self.model = self.model.to(device)

                    encoded = self.tokenizer(
                        input_texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_token_size
                    ).to(device)

                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=encoded["input_ids"],
                            attention_mask=encoded["attention_mask"]
                        )
                        embeddings = outputs.last_hidden_state.mean(dim=1)

                    if embeddings.dtype == torch.bfloat16:
                        return embeddings.detach().to(torch.float32).cpu().numpy()
                    else:
                        return embeddings.detach().cpu().numpy()

```

2. Then, modify `fast_graphrag/_llm/__init__.py` to include the initialization of the newly added classes.

```python
__all__ = [
    ...
    "HuggingFaceEmbeddingService",
]
...
from ._hf import HuggingFaceEmbeddingService

```

Then you can run the following command to indexing and inference:

```shell
export LLM_API_KEY=your_actual_api_key_here

python run_fast-graphrag.py \
  --subset medical \
  --base_dir ./Examples/fast-graphrag_workspace \
  --model_name gpt-4o-mini \
  --embed_model_path bge-large-en-v1.5 \
#   --sample 100 \
  --llm_base_url https://api.openai.com/v1

```

#### c. hipporag2
**We use hipporag2 version v1.0.0**.


```shell
export OPENAI_API_KEY=your_actual_api_key_here

python run_hipporag2.py \
  --subset medical \
  --base_dir ./Examples/hipporag2_workspace \
  --model_name gpt-4o-mini \
  --embed_model_path contriever \
#   --sample 100 \
  --llm_base_url https://api.openai.com/v1
```

We will continue updating other GraphRAG frameworks as much as possible. If you wish to integrate a different framework, you can refer to the structure of our result format. As long as your returned output matches the following fields, the evaluation code will run successfully:

```json
{
  "id": q["id"],
  "question": q["question"],
  "source": corpus_name,
  "context": context,
  "evidence": q["evidence"],
  "question_type": q["question_type"],
  "generated_answer": predicted_answer,
  "gold_answer": q["answer"]
}

```

### 2. Evaluation
#### a. Generation

```shell
cd Evaluation
export OPENAI_API_KEY=your_actual_api_key_here

python -m Evaluation.generation_eval \
  --mode API \
  --model gpt-4-turbo \
  --base_url https://api.openai.com/v1 \
  --bge_model BAAI/bge-large-en-v1.5 \
  --data_file ./results/lightrag.json \
  --output_file ./results/evaluation_results.json
```


#### b. Retrieval

```shell
cd Evaluation
export OPENAI_API_KEY=your_actual_api_key_here

python -m Evaluation.retrieval_eval \
  --mode API \
  --model gpt-4-turbo \
  --base_url https://api.openai.com/v1 \
  --bge_model BAAI/bge-large-en-v1.5 \
  --data_file ./results/lightrag.json \
  --output_file ./results/evaluation_results.json
```

## 📬 Contribution & Contact

Contributions to improve the benchmark website are welcome. Please contact the project team via `<a href="mailto:GraphRAG@hotmail.com">`GraphRAG@hotmail.com`</a>`.

## 📝 Citation

If you find this benchmark helpful, please cite our paper:

```
@article{xiang2025use,
  title={When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation},
  author={Xiang, Zhishang and Wu, Chuanjie and Zhang, Qinggang and Chen, Shengyuan and Hong, Zijin and Huang, Xiao and Su, Jinsong},
  journal={arXiv preprint arXiv:2506.05690},
  year={2025}
}
```

<h2 id="stars">✨ Stars History</h2>

[![Star History Chart](https://api.star-history.com/svg?repos=GraphRAG-Bench/GraphRAG-Benchmark&type=Date)](https://www.star-history.com/#GraphRAG-Bench/GraphRAG-Benchmark&Date)
