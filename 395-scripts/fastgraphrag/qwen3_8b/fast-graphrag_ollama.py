import asyncio
import os
import logging
import argparse
import json
from typing import Dict, List

from dotenv import load_dotenv
from datasets import load_dataset
from fast_graphrag import GraphRAG
from fast_graphrag._llm import OpenAILLMService, OpenAIEmbeddingService
from tqdm import tqdm

load_dotenv()

DOMAIN_CONFIG = {
    "medical": {
        "domain": "Analyze these medical texts and extract key conditions, symptoms, treatments, medications, and relationships.",
        "entity_types": ["Symptom", "Condition", "Treatment", "Medication", "Procedure"],
        "example_queries": [
            "What symptoms are mentioned?",
            "What treatments are described?",
            "What medical conditions does the patient have?",
        ],
    },
    "novel": {
        "domain": "Analyze this story and identify the characters and their relationships.",
        "entity_types": ["Character", "Place", "Event", "Object", "Activity"],
        "example_queries": [
            "Who are the main characters?",
            "What events drive the plot?",
            "How do characters interact with each other?",
        ],
    },
}

SUBSET_PATHS = {
    "medical": {
        "corpus": "./Datasets/Corpus/medical.parquet",
        "questions": "./Datasets/Questions/medical_questions.parquet",
    },
    "novel": {
        "corpus": "./Datasets/Corpus/novel.parquet",
        "questions": "./Datasets/Questions/novel_questions.parquet",
    },
}


def group_questions_by_source(question_list: List[dict]) -> Dict[str, List[dict]]:
    grouped = {}
    for q in question_list:
        grouped.setdefault(q["source"], []).append(q)
    return grouped


def process_corpus(
    corpus_name: str,
    context: str,
    subset: str,
    base_dir: str,
    model_name: str,
    embed_model_path: str,
    llm_base_url: str,
    llm_api_key: str,
    questions: Dict[str, List[dict]],
    sample: int | None,
):
    logging.info(f"Processing corpus: {corpus_name}")

    output_dir = f"./results/fast-graphrag/{subset}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{subset}_predictions.json")

    llm_service = OpenAILLMService(
        model=model_name,
        base_url=llm_base_url,
        api_key=llm_api_key,
    )

    embedding_service = OpenAIEmbeddingService(
        model=embed_model_path,
        base_url=llm_base_url,
        api_key=llm_api_key,
        embedding_dim=768,
    )

    cfg = DOMAIN_CONFIG[subset]
    grag = GraphRAG(
        working_dir=os.path.join(base_dir, subset),
        domain=cfg["domain"],
        example_queries="\n".join(cfg["example_queries"]),
        entity_types=cfg["entity_types"],
        config=GraphRAG.Config(
            llm_service=llm_service,
            embedding_service=embedding_service,
        ),
    )

    grag.insert(context)
    logging.info(f"Indexed corpus ({len(context.split())} words)")

    corpus_questions = questions.get(corpus_name, [])
    if not corpus_questions:
        logging.warning(f"No questions found for corpus: {corpus_name}")
        return

    if sample:
        corpus_questions = corpus_questions[:sample]

    logging.info(f"Processing {len(corpus_questions)} questions")

    results: List[dict] = []

    for q in tqdm(corpus_questions, desc=f"Answering {subset} questions"):
        try:
            response = grag.query(q["question"])
            ctx = response.to_dict().get("context", {})
            chunk_data = ctx.get("chunks", [])
            contexts = [c[0]["content"] for c in chunk_data]

            results.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "source": corpus_name,
                    "question_type": q.get("question_type"),
                    "evidence": q.get("evidence"),
                    "context": contexts,
                    "generated_answer": str(response.response),
                    "ground_truth": q.get("answer"),
                }
            )
        except Exception as e:
            logging.error(f"Error processing question {q['id']}: {e}")
            results.append({"id": q["id"], "error": str(e)})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"Saved predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="FastGraphRAG with Ollama")
    parser.add_argument("--subset", required=True, choices=["medical", "novel"])
    parser.add_argument("--base_dir", default="./Examples/graphrag_qwen3_8b")
    parser.add_argument("--model_name", default=os.getenv("MODEL_NAME", "qwen3:8b"))
    parser.add_argument("--embed_model_path", default=os.getenv("EMBED_MODEL", "nomic-embed-text"))
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument(
        "--llm_base_url",
        default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1"),
    )
    parser.add_argument(
        "--llm_api_key",
        default=os.getenv("OPENAI_API_KEY", "ollama"),
        help="API key for OpenAI-compatible endpoint",
    )

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"graphrag_{args.subset}.log"),
        ],
    )

    logging.info(f"Starting FastGraphRAG for subset: {args.subset}")

    corpus_path = SUBSET_PATHS[args.subset]["corpus"]
    question_path = SUBSET_PATHS[args.subset]["questions"]

    try:
        corpus = load_dataset("parquet", data_files=corpus_path, split="train")
        corpus_data = [
            {"corpus_name": row["corpus_name"], "context": row["context"]}
            for row in corpus
        ]
        logging.info(f"Loaded {len(corpus_data)} documents")
    except Exception as e:
        logging.error(f"Failed to load corpus: {e}")
        return

    try:
        qset = load_dataset("parquet", data_files=question_path, split="train")
        question_list = [
            {
                "id": row["id"],
                "source": row["source"],
                "question": row["question"],
                "answer": row["answer"],
                "question_type": row["question_type"],
                "evidence": row["evidence"],
            }
            for row in qset
        ]
        grouped_questions = group_questions_by_source(question_list)
        logging.info(f"Loaded {len(question_list)} questions")
    except Exception as e:
        logging.error(f"Failed to load questions: {e}")
        return

    async def run_all():
        tasks = [
            asyncio.to_thread(
                process_corpus,
                item["corpus_name"],
                item["context"],
                args.subset,
                args.base_dir,
                args.model_name,
                args.embed_model_path,
                args.llm_base_url,
                args.llm_api_key,
                grouped_questions,
                args.sample,
            )
            for item in corpus_data
        ]
        await asyncio.gather(*tasks)

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
