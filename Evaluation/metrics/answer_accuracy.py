import asyncio
import json
import numpy as np
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional
from langchain_core.language_models import BaseLanguageModel
import re
from langchain_core.embeddings import Embeddings
from langchain_core.callbacks import Callbacks
from Evaluation.metrics.utils import JSONHandler

# Define necessary Pydantic models
class StatementsWithReason(BaseModel):
    statement: str
    reason: str

class ClassificationWithReason(BaseModel):
    TP: List[StatementsWithReason] = []
    FP: List[StatementsWithReason] = []
    FN: List[StatementsWithReason] = []

class QuestionAnswerGroundTruth(BaseModel):
    question: str
    answer: List[str]
    ground_truth: List[str]

# F-beta score calculation
def fbeta_score(tp: int, fp: int, fn: int, beta: float = 1.0) -> float:
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-10)

# Statement generation prompt template
STATEMENT_GENERATOR_PROMPT = """
Given a question and an answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements. Ensure that no pronouns are used in any statement. Format the outputs in JSON.

Example Input: 
Question: Who was Albert Einstein and what is he best known for?
Answer: He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.

Example Output:
["Albert Einstein was a German-born theoretical physicist.", "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.","Albert Einstein was best known for developing the theory of relativity.","Albert Einstein also made important contributions to the development of the theory of quantum mechanics."]

Input Text:
Question:{question}
Answer: {answer}

Generated Statements:
"""

# Correctness classification prompt template
CORRECTNESS_PROMPT_TEMPLATE = """
Given a ground truth and an answer statements, analyze each statement and classify them in one of the following categories: TP (true positive): statements that are present in answer that are also directly supported by the one or more statements in ground truth, FP (false positive): statements present in the answer but not directly supported by any statement in ground truth, FN (false negative): statements found in the ground truth but not present in answer. Each statement can only belong to one of the categories. Provide a reason for each classification.

Examples:
{examples}

Current Analysis:
Question: {question}
Answer Statements: {answer}
Ground Truth Statements: {ground_truth}
"""

# Pre-defined examples for correctness classification
CORRECTNESS_EXAMPLES = [
    {
        "input": {
            "question": "What powers the sun and what is its primary function?",
            "answer": [
                "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.",
                "The primary function of the sun is to provide light to the solar system."
            ],
            "ground_truth": [
                "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.",
                "This fusion process in the sun's core releases a tremendous amount of energy.",
                "The energy from the sun provides heat and light, which are essential for life on Earth.",
                "The sun's light plays a critical role in Earth's climate system.",
                "Sunlight helps to drive the weather and ocean currents."
            ]
        },
        "output": {
            "TP": [
                {
                    "statement": "The primary function of the sun is to provide light to the solar system.",
                    "reason": "This statement is somewhat supported by the ground truth mentioning the sun providing light and its roles, though it focuses more broadly on the sun's energy."
                }
            ],
            "FP": [
                {
                    "statement": "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.",
                    "reason": "This statement is incorrect and contradicts the ground truth which states that the sun is powered by nuclear fusion."
                }
            ],
            "FN": [
                {
                    "statement": "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.",
                    "reason": "This accurate description of the sun’s power source is not included in the answer."
                },
                {
                    "statement": "This fusion process in the sun's core releases a tremendous amount of energy.",
                    "reason": "This process and its significance are not mentioned in the answer."
                },
                {
                    "statement": "The energy from the sun provides heat and light, which are essential for life on Earth.",
                    "reason": "The answer only mentions light, omitting the essential aspects of heat and its necessity for life, which the ground truth covers."
                },
                {
                    "statement": "The sun's light plays a critical role in Earth's climate system.",
                    "reason": "This broader impact of the sun’s light on Earth's climate system is not addressed in the answer."
                },
                {
                    "statement": "Sunlight helps to drive the weather and ocean currents.",
                    "reason": "The effect of sunlight on weather patterns and ocean currents is omitted in the answer."
                }
            ]
        }
    },
    {
        "input": {
            "question": "What is the boiling point of water?",
            "answer": [
                "The boiling point of water is 100 degrees Celsius at sea level"
            ],
            "ground_truth": [
                "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
                "The boiling point of water can change with altitude."
            ]
        },
        "output": {
            "TP": [
                {
                    "statement": "The boiling point of water is 100 degrees Celsius at sea level",
                    "reason": "This statement is directly supported by the ground truth which specifies the boiling point of water as 100 degrees Celsius at sea level."
                }
            ],
            "FP": [],
            "FN": [
                {
                    "statement": "The boiling point of water can change with altitude.",
                    "reason": "This additional information about how the boiling point of water can vary with altitude is not mentioned in the answer."
                }
            ]
        }
    }
]

async def compute_answer_correctness(
    question: str,
    answer: str,
    ground_truth: str,
    llm: BaseLanguageModel,
    embeddings: Embeddings,
    weights: List[float] = [0.75, 0.25],
    beta: float = 1.0,
    callbacks: Callbacks = None
) -> float:
    """Compute answer correctness score combining factuality and semantic similarity"""
    # Generate statements from answer and ground truth
    answer_statements = await generate_statements(llm, question, answer, callbacks)
    gt_statements = await generate_statements(llm, question, ground_truth, callbacks)

    # Calculate factuality score using statement classification
    factuality_score = await calculate_factuality(
        llm, question, answer_statements, gt_statements, callbacks, beta
    ) if weights[0] != 0 else 0.0
    # Calculate semantic similarity
    similarity_score = await calculate_semantic_similarity(
        embeddings, answer, ground_truth
    ) if weights[1] != 0 else 0.0

    # Combine scores using weighted average
    return float(np.average([factuality_score, similarity_score], weights=weights))

async def generate_statements(
    llm: BaseLanguageModel, question: str, answer:str, callbacks: Callbacks
) -> List[str]:
    """Generate concise factual statements from text"""
    handler = JSONHandler()
    prompt = STATEMENT_GENERATOR_PROMPT.format(question=question, answer=answer)
    response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
    parsed = await handler.parse_with_fallbacks(response.content)
    # Ensure we always return List[str]
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    if isinstance(parsed, dict):
        # Prefer common keys if present
        for key in ["statements", "answers", "items", "list", "output", "result"]:
            value = parsed.get(key)
            if isinstance(value, list):
                return [str(x) for x in value]
        # Fallback: flatten dict values
        return [str(v) for v in parsed.values()]
    # Fallback to single string wrapped in list
    return [str(parsed)]

async def calculate_factuality(
    llm: BaseLanguageModel,
    question: str,
    answer_stmts: List[str],
    gt_stmts: List[str],
    callbacks: Callbacks,
    beta: float
) -> float:
    """Classify statements and calculate factuality F-beta score"""
    if not answer_stmts and not gt_stmts:
        return 1.0  # Perfect score if both empty

    # Prepare examples for prompt
    examples = "\n".join(
        f"Input: {json.dumps(ex['input'])}\nOutput: {json.dumps(ex['output'])}"
        for ex in CORRECTNESS_EXAMPLES
    )

    # Generate classification
    prompt = CORRECTNESS_PROMPT_TEMPLATE.format(
        examples=examples,
        question=question,
        answer=json.dumps(answer_stmts),
        ground_truth=json.dumps(gt_stmts)
    )
    response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
    
    try:
        parsed = await JSONHandler().parse_with_fallbacks(response.content)
        classification = ClassificationWithReason(**parsed)
        tp = len(classification.TP)
        fp = len(classification.FP)
        fn = len(classification.FN)
        return fbeta_score(tp, fp, fn, beta)
    except (json.JSONDecodeError, TypeError):
        return 0.0  # Return minimum score on failure

async def calculate_semantic_similarity(
    embeddings: Embeddings, answer: str, ground_truth: str
) -> float:
    """Compute cosine similarity between answer and ground truth embeddings"""
    a_embed, gt_embed = await asyncio.gather(
        embeddings.aembed_query(answer),
        embeddings.aembed_query(ground_truth)
    )
    cosine_sim = np.dot(a_embed, gt_embed) / (
        np.linalg.norm(a_embed) * np.linalg.norm(gt_embed))
    return (cosine_sim + 1) / 2  # Scale to [0, 1]