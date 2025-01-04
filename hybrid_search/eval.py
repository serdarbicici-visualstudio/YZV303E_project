from openai import OpenAI
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

# --- Initialize OpenAI Client with DeepInfra ---
openai = OpenAI(
    api_key="2Q2AU9IG4jKLdqmRrHc2UxaLP8hHB0ii",  # Replace with your actual API key
    base_url="https://api.deepinfra.com/v1/openai",
)

# --- LLM Invocation ---
def invoke_llm(prompt):
    """
    Sends a prompt to the LLM and returns a numeric score between 0 and 1.
    """
    system_prompt = """
    You are an evaluator LLM. Respond with a number between 0 and 1. Just give a number between 0 and 1.
    Example output: 0.3423
    """
    response = openai.chat.completions.create(
        model="Qwen/QwQ-32B-Preview",
        messages=[{'role': 'system', 'content': system_prompt}, {"role": "user", "content": prompt}],
        temperature=0.0,  # Force deterministic output
    )
    output = response.choices[0].message.content.strip()
    try:
        return float(output)
    except ValueError:
        return 0.0  # Default to 0.0 if invalid

# --- Faithfulness Metric ---
def faithfulness_metric(prediction, context):
    """
    Evaluates if the prediction aligns with the provided context.
    """
    prompt = f"""
    Evaluate whether the given answer is supported by the provided context.

    **Instructions**:
    - Give a score between **0 and 1** based on how well the answer matches the information in the context.
    - Score **1.0** if the answer is **fully supported** by the context.
    - Score **0.5** if the answer is **partially supported** but contains some inaccuracies.
    - Score **0.0** if the answer **contradicts** the context or is **unsupported**.

    **Examples**:
    Context: "The Hagia Sophia, located in Istanbul, was originally built as a church in 537 AD. It was later converted into a mosque and now serves as a museum and mosque."
    Answer: "Hagia Sophia was built in the 6th century and has always been a mosque."
    Score: **0.5**

    Context: "Topkapi Palace served as the administrative center of the Ottoman Empire until the 19th century."
    Answer: "Topkapi Palace was the residence of Ottoman sultans for 400 years."
    Score: **1.0**

    **Evaluation**:
    Context: {context}
    Answer: {prediction}
    Score:
    """
    return invoke_llm(prompt)

# --- Relevancy Metric ---
def relevancy_metric(query, prediction):
    """
    Evaluates if the prediction directly addresses the query.
    """
    prompt = f"""
    Evaluate whether the given answer directly addresses the question.

    **Instructions**:
    - Give a score between **0 and 1** based on relevance.
    - Score **1.0** if the answer **fully answers** the question.
    - Score **0.5** if the answer **partially addresses** the question but lacks details.
    - Score **0.0** if the answer is **irrelevant** or **off-topic**.

    **Examples**:
    Question: "What are the main attractions in Sultanahmet?"
    Answer: "Sultanahmet has landmarks like the Blue Mosque, Hagia Sophia, and Topkapi Palace."
    Score: **1.0**

    Question: "What are the main attractions in Sultanahmet?"
    Answer: "Istanbul has many historical places, including the Galata Tower."
    Score: **0.3**

    **Evaluation**:
    Question: {query}
    Answer: {prediction}
    Score:
    """
    return invoke_llm(prompt)
# --- Factual Correctness Metric ---
def factual_correctness_metric(prediction, reference):
    """
    Evaluates if the prediction is factually accurate based on the reference.
    """
    prompt = f"""
    Evaluate whether the given prediction is factually correct based on the provided reference.

    **Instructions**:
    - Give a score between **0 and 1** based on factual accuracy.
    - Score **1.0** if the prediction is **entirely correct**.
    - Score **0.5** if the prediction is **partially correct** but has **minor errors**.
    - Score **0.0** if the prediction contains **factual errors** or **false information**.

    **Examples**:
    Reference: "The Galata Tower offers panoramic views of Istanbul and dates back to the medieval period."
    Prediction: "Galata Tower is a medieval structure with great views of Istanbul."
    Score: **1.0**

    Reference: "Topkapi Palace served as the administrative center of the Ottoman Empire until the 19th century."
    Prediction: "Topkapi Palace is a modern government building in Istanbul."
    Score: **0.0**

    **Evaluation**:
    Reference: {reference}
    Prediction: {prediction}
    Score:
    """
    return invoke_llm(prompt)

# --- BLEU Score Metric ---
def bleu_score_metric(prediction, reference):
    """
    Calculates BLEU score to measure similarity between prediction and reference.
    """
    smooth = SmoothingFunction().method1  # Apply smoothing
    ref_tokens = [reference.split()]
    pred_tokens = prediction.split()
    score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smooth)
    return score

# --- Evaluation Pipeline ---
def evaluate_single_input(query, reference, prediction, context):
    """
    Evaluates a single query, reference, prediction, and context using multiple metrics.
    """
    # Calculate Metrics
    faithfulness = faithfulness_metric(prediction, context)
    relevancy = relevancy_metric(query, prediction)
    factual_correctness = factual_correctness_metric(prediction, reference)
    bleu_score = bleu_score_metric(prediction, reference)

    # Combine metrics with weights
    overall_score = (
        0.3 * faithfulness + 0.3 * relevancy + 0.4 * factual_correctness
    )

    return {
        "Faithfulness": faithfulness,
        "Relevancy": relevancy,
        "Factual Correctness": factual_correctness,
        "BLEU Score": bleu_score,
        "Overall Score": overall_score,
    }