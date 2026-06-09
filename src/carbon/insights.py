"""Ollama-powered carbon coaching insights."""

import json
import re

import ollama

INSIGHTS_PROMPT = """You are a carbon footprint coach.

Analyze the provided emission data.

Requirements:
- Keep responses short and practical.
- No more than 20 words per field.
- Use only information supported by the data.
- Name the habit or number behind each point (e.g. air travel, km driven).
- Prioritize transport, flights, vehicle type, heating, diet, and waste.
- No disclaimers.
- No introductions.
- No conclusions.

Return ONLY valid JSON:

{{
  "main_drivers": [
    "...",
    "...",
    "..."
  ],
  "actions": [
    "...",
    "...",
    "..."
  ],
  "reality_check": "..."
}}

Data:
{user_data}"""


def parse_insights_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def get_ai_insights(user_data: dict, model: str = "minimax-m2.5:cloud") -> dict:
    prompt = INSIGHTS_PROMPT.format(
        user_data=json.dumps(user_data, indent=2, ensure_ascii=False)
    )
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    return parse_insights_json(response["message"]["content"])
