#!/usr/bin/env python3
"""
Cross-Lingual Transfer Experiment Runner

Runs 50 benchmark questions across 5 conditions and 7 LLMs.
Each question is asked as a fresh conversation with no prior context.
Results are saved per condition per model as JSON files.

Usage:
    python run_experiments.py                          # Run everything
    python run_experiments.py --model claude_opus_4.5  # Run one model
    python run_experiments.py --condition C1            # Run one condition
    python run_experiments.py --model grok_4.1 --condition C2 C3  # Combine filters
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------

MODELS = {
    "claude_opus_4.5": {
        "provider": "anthropic",
        "model_id": "claude-opus-4-5",
    },
    "gpt_5.2": {
        "provider": "openai",
        "model_id": "gpt-5.2",
    },
    "gemini_3_pro": {
        "provider": "google",
        "model_id": "gemini-3-pro-preview",
    },
    "grok_4.1": {
        "provider": "xai",
        "model_id": "grok-4-1-fast",
    },
    "deepseek_v3.2": {
        "provider": "huggingface",
        "model_id": "deepseek-ai/DeepSeek-V3.2",
    },
    "llama_4_maverick": {
        "provider": "huggingface",
        "model_id": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    },
    "qwen3": {
        "provider": "huggingface",
        "model_id": "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "hf_provider": "together",
    },
    "aya_expanse": {
        "provider": "huggingface",
        "model_id": "CohereLabs/aya-expanse-32b",
    },
}

# ---------------------------------------------------------------------------
# Condition Configuration
# ---------------------------------------------------------------------------

CONDITIONS = {
    "C1": {
        "name": "C1_english",
        "language": "english",
        "question_field": "english",
        "system_prompt": None,
        "description": "English -> English (upper bound baseline)",
    },
    "C2": {
        "name": "C2_kazakh",
        "language": "kazakh",
        "question_field": "kazakh",
        "system_prompt": (
            "Сіз пайдалы көмекші ботсыз. "
            "Барлық жауаптарыңызды тек қазақ тілінде беріңіз."
        ),
        "description": "Kazakh -> Kazakh (direct low-resource performance)",
    },
    "C3": {
        "name": "C3_kazakh_transfer",
        "language": "kazakh",
        "question_field": "kazakh",
        "system_prompt": (
            "Сіз пайдалы көмекші ботсыз. "
            "Пайдаланушы тек қазақ тілін біледі. "
            "Келесі қадамдарды орындаңыз:\n"
            "1. Сұрақты ағылшын тіліне аударыңыз\n"
            "2. Ағылшын тілінде жауап беріңіз\n"
            "3. Жауапты қазақ тіліне аударыңыз\n"
            "Әр қадамды көрсетіңіз."
        ),
        "description": "Kazakh -> English -> English -> Kazakh (cross-lingual transfer)",
    },
    "C4": {
        "name": "C4_mongolian",
        "language": "mongolian",
        "question_field": "mongolian",
        "system_prompt": (
            "Та бол хэрэгтэй туслах бот. "
            "Бүх хариултаа зөвхөн монгол хэлээр бичнэ үү."
        ),
        "description": "Mongolian -> Mongolian (direct low-resource performance)",
    },
    "C5": {
        "name": "C5_mongolian_transfer",
        "language": "mongolian",
        "question_field": "mongolian",
        "system_prompt": (
            "Та бол хэрэгтэй туслах бот. "
            "Хэрэглэгч зөвхөн монгол хэл мэддэг. "
            "Дараах алхмуудыг гүйцэтгэнэ үү:\n"
            "1. Асуултыг англи хэл рүү орчуулна уу\n"
            "2. Англи хэлээр хариулт бичнэ үү\n"
            "3. Хариултыг монгол хэл рүү орчуулна уу\n"
            "Алхам тус бүрийг харуулна уу."
        ),
        "description": "Mongolian -> English -> English -> Mongolian (cross-lingual transfer)",
    },
}

# ---------------------------------------------------------------------------
# API Wrappers
# ---------------------------------------------------------------------------

# Lazy-initialized clients
_clients = {}


def _get_anthropic_client():
    if "anthropic" not in _clients:
        import anthropic
        _clients["anthropic"] = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            timeout=120.0,
        )
    return _clients["anthropic"]


def _get_openai_client():
    if "openai" not in _clients:
        from openai import OpenAI
        _clients["openai"] = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            timeout=120.0,
        )
    return _clients["openai"]


def _get_xai_client():
    if "xai" not in _clients:
        from openai import OpenAI
        _clients["xai"] = OpenAI(
            api_key=os.environ["XAI_API_KEY"],
            base_url="https://api.x.ai/v1",
            timeout=120.0,
        )
    return _clients["xai"]


def _get_google_client():
    if "google" not in _clients:
        from google import genai
        _clients["google"] = genai.Client(
            api_key=os.environ["GOOGLE_API_KEY"],
            http_options={"timeout": 120_000},
        )
    return _clients["google"]


def _get_hf_client():
    if "huggingface" not in _clients:
        from huggingface_hub import InferenceClient
        _clients["huggingface"] = InferenceClient(
            api_key=os.environ["HF_API_KEY"],
            timeout=60,
        )
    return _clients["huggingface"]


def call_anthropic(model_id: str, system_prompt: str | None, user_message: str) -> str:
    client = _get_anthropic_client()
    kwargs = {
        "model": model_id,
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": user_message}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    response = client.messages.create(**kwargs)
    return response.content[0].text


def call_openai(model_id: str, system_prompt: str | None, user_message: str) -> str:
    client = _get_openai_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_completion_tokens=2048,
    )
    return response.choices[0].message.content


def call_xai(model_id: str, system_prompt: str | None, user_message: str) -> str:
    client = _get_xai_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=2048,
    )
    return response.choices[0].message.content


def call_google(model_id: str, system_prompt: str | None, user_message: str) -> str:
    from google.genai import types

    client = _get_google_client()
    config = None
    if system_prompt:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=2048,
        )
    else:
        config = types.GenerateContentConfig(max_output_tokens=2048)
    response = client.models.generate_content(
        model=model_id,
        contents=user_message,
        config=config,
    )
    return response.text


def call_huggingface(model_id: str, system_prompt: str | None, user_message: str) -> str:
    # Look up optional HF provider override (e.g. "together" for Qwen3)
    hf_provider = None
    for cfg in MODELS.values():
        if cfg["model_id"] == model_id:
            hf_provider = cfg.get("hf_provider")
            break
    if hf_provider:
        from huggingface_hub import InferenceClient
        client = InferenceClient(
            provider=hf_provider,
            api_key=os.environ["HF_API_KEY"],
            timeout=60,
        )
    else:
        client = _get_hf_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})
    response = client.chat_completion(
        model=model_id,
        messages=messages,
        max_tokens=2048,
    )
    return response.choices[0].message.content


PROVIDER_DISPATCH = {
    "anthropic": call_anthropic,
    "openai": call_openai,
    "xai": call_xai,
    "google": call_google,
    "huggingface": call_huggingface,
}

# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
QUESTIONS_PATH = BASE_DIR / "questions.json"

MAX_RETRIES = 3
RETRY_BASE_DELAY = 5  # seconds (exponential backoff)
delay_between_calls = 1.0  # seconds; overridden by --delay flag


def load_questions() -> list[dict]:
    with open(QUESTIONS_PATH) as f:
        return json.load(f)


def get_result_path(condition_name: str, model_name: str) -> Path:
    return RESULTS_DIR / condition_name / f"{model_name}.json"


def load_existing_results(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def save_results(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def call_model_with_retry(
    provider: str, model_id: str, system_prompt: str | None, user_message: str
) -> str:
    call_fn = PROVIDER_DISPATCH[provider]
    for attempt in range(MAX_RETRIES):
        try:
            return call_fn(model_id, system_prompt, user_message)
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = any(
                keyword in err_str
                for keyword in ["rate limit", "rate_limit", "429", "too many requests"]
            )
            if is_rate_limit and attempt < MAX_RETRIES - 1:
                wait = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"    Rate limited, waiting {wait}s before retry...")
                time.sleep(wait)
            elif attempt < MAX_RETRIES - 1:
                wait = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"    Error: {e}")
                print(f"    Retrying in {wait}s... (attempt {attempt + 2}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                print(f"    FAILED after {MAX_RETRIES} attempts: {e}")
                return f"[ERROR] {e}"


def run_condition_model(
    condition_key: str,
    model_name: str,
    questions: list[dict],
    force: bool = False,
):
    condition = CONDITIONS[condition_key]
    model_cfg = MODELS[model_name]
    result_path = get_result_path(condition["name"], model_name)

    # Check for existing results (resume support)
    existing = load_existing_results(result_path)
    if existing and not force:
        completed_ids = {r["question_id"] for r in existing.get("responses", [])}
        if len(completed_ids) == len(questions):
            print(f"  [{condition_key}] {model_name}: Already complete (50/50), skipping.")
            return
        print(f"  [{condition_key}] {model_name}: Resuming from {len(completed_ids)}/50")
    else:
        existing = None
        completed_ids = set()

    # Build result structure
    result_data = existing or {
        "model": model_name,
        "model_id": model_cfg["model_id"],
        "condition": condition_key,
        "condition_description": condition["description"],
        "language": condition["language"],
        "system_prompt_used": condition["system_prompt"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "responses": [],
    }

    provider = model_cfg["provider"]
    model_id = model_cfg["model_id"]
    system_prompt = condition["system_prompt"]

    for i, q in enumerate(questions):
        if q["id"] in completed_ids:
            continue

        question_text = q[condition["question_field"]]

        print(f"  [{condition_key}] {model_name}: {i + 1}/50 (Q{q['id']} - {q['category']})")

        start_time = time.time()
        response_text = call_model_with_retry(
            provider, model_id, system_prompt, question_text
        )
        elapsed = round(time.time() - start_time, 2)

        print(f"    -> {elapsed}s")

        result_data["responses"].append({
            "question_id": q["id"],
            "category": q["category"],
            "question_text": question_text,
            "response": response_text,
            "reference_answer": q.get("reference_answer"),
            "response_time_seconds": elapsed,
        })

        # Save incrementally after each response
        result_data["last_updated"] = datetime.now(timezone.utc).isoformat()
        save_results(result_path, result_data)

        time.sleep(delay_between_calls)

    print(f"  [{condition_key}] {model_name}: Done! ({len(result_data['responses'])}/50)")


def check_api_keys(models_to_run: list[str]) -> list[str]:
    """Check which API keys are available and return list of missing ones."""
    provider_env = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "xai": "XAI_API_KEY",
        "huggingface": "HF_API_KEY",
    }
    needed_providers = {MODELS[m]["provider"] for m in models_to_run}
    missing = []
    for provider in needed_providers:
        env_var = provider_env[provider]
        if not os.environ.get(env_var):
            missing.append(f"  {env_var} (for {provider})")
    return missing


def main():
    parser = argparse.ArgumentParser(
        description="Run cross-lingual transfer experiments"
    )
    parser.add_argument(
        "--model", nargs="+",
        help=f"Model(s) to run. Choices: {', '.join(MODELS.keys())}",
    )
    parser.add_argument(
        "--condition", nargs="+",
        help=f"Condition(s) to run. Choices: {', '.join(CONDITIONS.keys())}",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if results already exist",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Delay between API calls in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_config",
        help="List all models and conditions, then exit",
    )
    args = parser.parse_args()

    if args.list_config:
        print("Models:")
        for name, cfg in MODELS.items():
            print(f"  {name:25s} provider={cfg['provider']:15s} model_id={cfg['model_id']}")
        print("\nConditions:")
        for key, cfg in CONDITIONS.items():
            print(f"  {key}: {cfg['description']}")
        print(f"\nTotal data points: {len(MODELS)} models x {len(CONDITIONS)} conditions x 50 questions = {len(MODELS) * len(CONDITIONS) * 50}")
        return

    # Determine what to run
    models_to_run = args.model if args.model else list(MODELS.keys())
    conditions_to_run = args.condition if args.condition else list(CONDITIONS.keys())

    # Validate
    for m in models_to_run:
        if m not in MODELS:
            print(f"Error: Unknown model '{m}'. Valid: {', '.join(MODELS.keys())}")
            sys.exit(1)
    for c in conditions_to_run:
        if c not in CONDITIONS:
            print(f"Error: Unknown condition '{c}'. Valid: {', '.join(CONDITIONS.keys())}")
            sys.exit(1)

    # Check API keys
    missing = check_api_keys(models_to_run)
    if missing:
        print("Missing API keys (set these in .env file):")
        for m in missing:
            print(m)
        print("\nSee .env.example for details.")
        sys.exit(1)

    # Override delay if specified
    global delay_between_calls
    delay_between_calls = args.delay

    # Load questions
    questions = load_questions()
    total_calls = len(models_to_run) * len(conditions_to_run) * len(questions)
    print(f"Running {len(models_to_run)} model(s) x {len(conditions_to_run)} condition(s) x {len(questions)} questions = {total_calls} API calls")
    print(f"Results directory: {RESULTS_DIR}\n")

    # Run experiments: iterate condition by condition, model by model
    for cond_key in conditions_to_run:
        cond = CONDITIONS[cond_key]
        print(f"\n{'='*60}")
        print(f"Condition {cond_key}: {cond['description']}")
        print(f"{'='*60}")
        for model_name in models_to_run:
            run_condition_model(cond_key, model_name, questions, force=args.force)

    print(f"\nAll experiments complete. Results saved to {RESULTS_DIR}/")
    print("Run 'python generate_scoring_csv.py' to create the scoring spreadsheet.")


if __name__ == "__main__":
    main()
