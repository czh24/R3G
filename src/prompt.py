# -*- coding: utf-8 -*-
"""
Centralized prompts for the end-to-end pipeline.

Keep ALL prompt strings here so core scripts stay clean and open-source friendly.
"""

from __future__ import annotations
from typing import List, Optional, Dict


def build_reasoning_before_evidence_prompt(question: str) -> str:
    return f"""You are a reasoning planner for vision-centric question answering.

Your role is NOT to answer the question.
Instead, generate a structured, question-conditioned reasoning plan that specifies what visual evidence is required and how it should be verified BEFORE any retrieved images are used.

Core criteria (used later for evidence verification):
1) Semantic Relatedness: does the needed cue match the semantics implied by the question and query image?
2) Target Correspondence: does the cue precisely correspond to the asked target/viewpoint/state (not generic class info)?
3) Answerability: would observing the cue make the question directly answerable by resolving ambiguity/occlusion/temporal change/fine-grained detail?

Instructions:
- Do NOT mention retrieved images explicitly.
- Do NOT provide the final answer.
- Do NOT include refusal / limitation statements.
- Produce concise single-sentence steps describing concrete visual checks.

Output ONLY JSON:
{{
  "reasoning_steps": [
    "Step 1: ...",
    "Step 2: ..."
  ]
}}

Question: {question}
"""


def build_judge_prompt(question: str, reasoning_steps: Optional[List[str]] = None) -> str:
    steps = "\n".join([f"- {s}" for s in (reasoning_steps or [])])
    if not steps:
        steps = "(none)"

    return f"""You are an MLLM-as-Judge for vision-centric VQA evidence verification.

You will be given:
- A question
- A query image (image 1)
- A candidate evidence image (image 2)
Optionally, a reasoning plan (single-sentence steps) that describes which cues matter.

Score the candidate evidence image using THREE criteria in [0,1]:
1) Semantic relatedness (r): whether the dominant semantics of the candidate match the intent of the question + query image.
2) Target correspondence (t): whether the candidate focuses on the exact target/state/viewpoint asked, not just generic class context.
3) Answerability (a): whether combining the candidate with the query image makes the question decidable by supplying missing cues.

Guidelines:
- Use the question and the query image as the anchor.
- Penalize look-alike but off-topic images (low r).
- Penalize generic exemplars that do not match the specific target/viewpoint/state (low t).
- Penalize images that do not add decisive evidence (low a).
- Be conservative: if uncertain, assign mid/low scores.

Reasoning plan (if provided):
{steps}

Output ONLY one JSON object on a single line:
{{"r":0.xxx,"t":0.xxx,"a":0.xxx}}

Question: {question}
"""


def build_answer_prompt(
    question: str,
    choices: Dict[str, str],
    reasoning_steps: Optional[List[str]] = None,
    use_two_images: bool = True,
) -> str:
    image_tokens = "<image>" + ("<image>" if use_two_images else "")
    guidance = ""
    if reasoning_steps:
        guidance = "Reasoning guide (follow these checks):\n" + "\n".join([f"- {s}" for s in reasoning_steps]) + "\n"

    return (
        f"You will be given {'two images' if use_two_images else 'one image'}. "
        f"The first image is the query image"
        f"{', the second image is an evidence image' if use_two_images else ''}. "
        f"Choose the correct answer option from the given choices. "
        f"Output exactly one letter from [A,B,C,D] with no extra text.\n"
        f"{guidance}"
        f"Question: {question}\n"
        f"Choices:\n"
        f"A: {choices['A']}\n"
        f"B: {choices['B']}\n"
        f"C: {choices['C']}\n"
        f"D: {choices['D']}\n"
        f"{image_tokens}\n"
    )
