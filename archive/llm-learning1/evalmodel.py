import json
import requests
import streamlit as st
import re

st.set_page_config(page_title="Local LLM Evaluator", layout="centered")
st.title("Local LLM Answer Evaluation")

# ========= INPUTS =========
question = st.text_input(
    "Question (optional):",
    placeholder="e.g. Explain the key concepts of Deep Learning"
)

answer_text = st.text_area(
    "Answer to evaluate (paste here):",
    height=220,
    placeholder="Paste the answer you want to evaluate here..."
)

reference_text = st.text_area(
    "Reference / Ideal Answer (optional):",
    height=180,
    placeholder="Optional: paste an ideal or reference answer"
)

# ========= PROMPT =========
def build_prompt(question, answer, reference):
    prompt = """
You are an independent and strict academic evaluator.

Your task is to evaluate an answer objectively.
"""

    if question.strip():
        prompt += f"""
Question:
\"\"\"
{question}
\"\"\"
"""

    prompt += f"""
Answer to evaluate:
\"\"\"
{answer}
\"\"\"
"""

    if reference.strip():
        prompt += f"""
Reference answer (gold standard):
\"\"\"
{reference}
\"\"\"
Use the reference answer as guidance when scoring.
"""

    prompt += """
Scoring criteria (0–5):
- conceptual_correctness
- coverage
- clarity
- depth

Rules:
- Be critical but fair
- Use integers only
- Return ONLY valid JSON
- Do NOT add any text outside JSON

JSON format:
{
  "conceptual_correctness": number,
  "coverage": number,
  "clarity": number,
  "depth": number,
  "overall_comment": string
}
"""
    return prompt

# ========= OLLAMA CALL =========
def run_evaluation(prompt: str):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0
            }
        },
        timeout=120
    )
    return response.json()["response"]

# ========= JSON SAFETY =========
def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output.")
    return json.loads(match.group())

# ========= RUN =========
if st.button("Evaluate Answer"):
    if not answer_text.strip():
        st.warning("Please paste an answer to evaluate.")
    else:
        prompt = build_prompt(question, answer_text, reference_text)

        with st.spinner("Evaluating with local LLM (llama3)..."):
            raw_output = run_evaluation(prompt)

        st.subheader("Raw Model Output")
        st.code(raw_output)

        try:
            result = extract_json(raw_output)

            st.subheader("Evaluation Result")
            col1, col2 = st.columns(2)

            col1.metric("Conceptual Correctness", result["conceptual_correctness"])
            col1.metric("Coverage", result["coverage"])
            col2.metric("Clarity", result["clarity"])
            col2.metric("Depth", result["depth"])

            st.subheader("Overall Comment")
            st.write(result["overall_comment"])

        except Exception as e:
            st.error("Failed to parse model output as JSON.")
            st.caption(str(e))
