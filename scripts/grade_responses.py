#!/usr/bin/env python3
"""
Auto-grade all 2,000 experiment responses using Claude Sonnet.

Produces 5 CSV files (one per condition), each with 400 rows (50 questions x 8 models).
Each row has: question_id, category, model, accuracy, fluency, completeness, total,
              response_time, response_length, notes

Uses the answer sheet as ground truth and follows the scoring rubric strictly.

Usage:
    python grade_responses.py                    # Grade everything
    python grade_responses.py --condition C1 C2  # Grade specific conditions
    python grade_responses.py --model qwen3      # Grade specific model(s)
    python grade_responses.py --dry-run           # Count what needs grading
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
GRADES_DIR = BASE_DIR / "grades"
ANSWER_SHEET_PATH = BASE_DIR / "Cross_lingual_transfer_benchmark_questions_answer_sheet.txt"

MODELS = [
    "claude_opus_4.5",
    "gpt_5.2",
    "gemini_3_pro",
    "grok_4.1",
    "deepseek_v3.2",
    "llama_4_maverick",
    "qwen3",
    "aya_expanse",
]

CONDITION_META = {
    "C1": {"dir": "C1_english",            "target_lang": "English",   "pipeline": False},
    "C2": {"dir": "C2_kazakh",             "target_lang": "Kazakh",    "pipeline": False},
    "C3": {"dir": "C3_kazakh_transfer",    "target_lang": "Kazakh",    "pipeline": True},
    "C4": {"dir": "C4_mongolian",          "target_lang": "Mongolian", "pipeline": False},
    "C5": {"dir": "C5_mongolian_transfer", "target_lang": "Mongolian", "pipeline": True},
}

CATEGORY_DISPLAY = {
    "neutral_factual":    "Neutral Factual",
    "neutral_reasoning":  "Neutral Reasoning/Logic",
    "neutral_technical":  "Neutral Technical",
    "kazakh_cultural":    "Kazakh-Aligned Cultural",
    "mongolian_cultural": "Mongolian-Aligned Cultural",
}

GRADING_MODEL = "claude-sonnet-4-5-20250929"


# ---------------------------------------------------------------------------
# Step 3 extraction for C3/C5 pipeline responses
# ---------------------------------------------------------------------------

STEP3_PATTERNS = [
    r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:3[\-\.:\s]?\s*(?:қадам|алхам|step|адым)|(?:қадам|алхам|step|адым)\s*3)",
    r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:қазақ тіліне аудар|аудармасы|қазақша жауап)",
    r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:монгол хэл рүү орчуул|орчуулга|монголоор хариул)",
    r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:translation|final answer|translated answer)",
]
STEP3_RE = re.compile("|".join(STEP3_PATTERNS), re.IGNORECASE | re.MULTILINE)


def extract_final_answer(response: str, condition: str) -> str:
    """For C3/C5, extract only the step 3 (final translated answer)."""
    if condition not in ("C3", "C5"):
        return response

    match = STEP3_RE.search(response)
    if match:
        return response[match.end():].strip()

    # Fallback: take the last third of the response (heuristic)
    lines = response.strip().split("\n")
    if len(lines) >= 6:
        third = len(lines) // 3
        return "\n".join(lines[-third:]).strip()

    return response


# ---------------------------------------------------------------------------
# System prompt (cached across calls for efficiency)
# ---------------------------------------------------------------------------

def build_system_prompt() -> list[dict]:
    """Build system prompt with rubric + full answer sheet (API-cached)."""
    answer_sheet = ANSWER_SHEET_PATH.read_text(encoding="utf-8")

    text = f"""\
You are an expert, impartial grader for an academic cross-lingual transfer experiment.
You will grade LLM responses on three metrics, each scored 0-2 (max 6 per question).

SCORING RUBRIC:

ACCURACY (0-2):
  0 = Incorrect: answer is wrong, fabricated, or completely off-topic
  1 = Partially Correct: some correct info but missing key details or has factual errors
  2 = Correct: factually accurate and complete, covers the main points

FLUENCY (0-2):
  0 = Broken: unreadable, wrong language, garbled text, or script errors
  1 = Understandable: meaning is clear but phrasing is awkward or unnatural
  2 = Natural: reads like a fluent native speaker wrote it

COMPLETENESS (0-2):
  0 = Minimal: one sentence or less, no useful detail
  1 = Partial: addresses the question but lacks depth or misses sub-parts
  2 = Thorough: fully addresses all parts of the question with sufficient detail

CRITICAL SCORING RULES:
1. Score each response independently. Do not let previous scores influence the next.
2. For pipeline conditions (C3 and C5), score the FINAL translated output only.
3. If the model refuses to answer or says it cannot help, score 0/0/0.
4. If the model answers in the WRONG language (e.g., responds in English when it
   should respond in Kazakh), score Fluency = 0 but still score Accuracy and
   Completeness based on the content.
5. KAZAKH and KYRGYZ are DIFFERENT languages. If a supposedly Kazakh response is
   actually in Kyrgyz, score Fluency = 0. Key Kyrgyz markers: uses "жооп" instead
   of Kazakh "жауап", "кыргыз" vs "қазақ", and notably LACKS Kazakh-specific
   letters: Ә, Ғ, Қ, Ң, Ө, Ұ, Ү, Һ, І.
6. For Reasoning questions: Accuracy = 2 requires BOTH correct logic AND correct
   final answer. A correct final number with wrong reasoning gets Accuracy = 1.
7. For Cultural questions: Accuracy = 2 requires cultural nuance and significant
   details. Basic facts without culturally important context gets Accuracy = 1.
8. Some questions accept multiple answers (marked in the answer sheet below).
9. For Q10 (NYSE 3rd best day): the English answer sheet says Oct 30, 1929 (+12%,
   Rockefeller statement). The Mongolian answer sheet says April 9, 2025 (+9.5%,
   Trump tariff pause). ACCEPT EITHER as correct for any condition.
10. Keep notes brief. Flag: surprisingly good/bad, hallucinations, wrong language,
    Kyrgyz-instead-of-Kazakh, interesting cultural insight, etc.

COMPLETE ANSWER SHEET (ground truth, all 50 questions in English, Kazakh, Mongolian):
================================================================================
{answer_sheet}
================================================================================

OUTPUT FORMAT:
Return ONLY a single-line JSON object, no markdown fences, no extra text:
{{"accuracy": <0-2>, "fluency": <0-2>, "completeness": <0-2>, "notes": "<brief note or empty string>"}}"""

    return [
        {
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }
    ]


# ---------------------------------------------------------------------------
# Grading via Claude API
# ---------------------------------------------------------------------------

def grade_one(
    client: anthropic.Anthropic,
    system: list[dict],
    *,
    question_id: int,
    category: str,
    question_text: str,
    reference_answer: str,
    response_text: str,
    target_lang: str,
    condition: str,
    is_pipeline: bool,
) -> dict:
    """Grade a single response. Returns {accuracy, fluency, completeness, total, notes}."""

    # For pipeline conditions, extract final answer
    eval_text = response_text
    pipeline_note = ""
    if is_pipeline:
        extracted = extract_final_answer(response_text, condition)
        if extracted != response_text:
            eval_text = extracted
            pipeline_note = (
                "\n\n(Note: Above is the extracted Step 3 / final translated answer "
                "from a pipeline response. Grade this final output only.)"
            )
        else:
            pipeline_note = (
                "\n\n(Note: This is from a pipeline condition (translate->answer->"
                "translate back). The model may have included all steps. Focus on "
                "the FINAL answer portion for grading.)"
            )

    user_msg = (
        f"CONDITION: {condition}\n"
        f"TARGET LANGUAGE for fluency: {target_lang}\n"
        f"CATEGORY: {CATEGORY_DISPLAY.get(category, category)}\n"
        f"QUESTION ID: Q{question_id}\n\n"
        f"QUESTION:\n{question_text}\n\n"
        f"ENGLISH REFERENCE ANSWER:\n{reference_answer}\n\n"
        f"MODEL'S RESPONSE TO EVALUATE:\n{eval_text}{pipeline_note}\n\n"
        f"Grade this response now. Return only the JSON object."
    )

    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=GRADING_MODEL,
                max_tokens=250,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = resp.content[0].text.strip()
            # Clean markdown fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)

            a = int(parsed.get("accuracy", parsed.get("Accuracy", -1)))
            f_ = int(parsed.get("fluency", parsed.get("Fluency", -1)))
            c = int(parsed.get("completeness", parsed.get("Completeness", parsed.get("completion", -1))))
            assert 0 <= a <= 2 and 0 <= f_ <= 2 and 0 <= c <= 2

            return {
                "accuracy": a,
                "fluency": f_,
                "completeness": c,
                "total": a + f_ + c,
                "notes": parsed.get("notes", ""),
            }
        except json.JSONDecodeError:
            if attempt < 2:
                time.sleep(2)
                continue
            return {
                "accuracy": -1, "fluency": -1, "completeness": -1, "total": -3,
                "notes": f"GRADE_PARSE_ERROR: {raw[:120]}",
            }
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
                continue
            return {
                "accuracy": -1, "fluency": -1, "completeness": -1, "total": -3,
                "notes": f"GRADE_API_ERROR: {str(e)[:120]}",
            }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Auto-grade experiment responses")
    ap.add_argument("--condition", nargs="*", help="Only grade these conditions (e.g. C1 C2)")
    ap.add_argument("--model", nargs="*", help="Only grade these models")
    ap.add_argument("--dry-run", action="store_true", help="Show counts without grading")
    args = ap.parse_args()

    conds = args.condition or list(CONDITION_META.keys())
    models = args.model or MODELS

    GRADES_DIR.mkdir(exist_ok=True)

    # Build system prompt once (cached by Anthropic API after first call)
    print("Loading answer sheet and building system prompt...")
    system = build_system_prompt()

    client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        timeout=120.0,
    )

    grand_total = 0
    grand_graded = 0

    for ck in conds:
        meta = CONDITION_META[ck]
        cond_dir = RESULTS_DIR / meta["dir"]
        target_lang = meta["target_lang"]
        is_pipeline = meta["pipeline"]

        # Resume support: load existing grading state
        state_file = GRADES_DIR / f".{meta['dir']}_state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
        else:
            state = {}

        all_rows = []

        print(f"\n{'='*60}")
        print(f"Condition {ck} ({meta['dir']}) — target: {target_lang}")
        print(f"{'='*60}")

        for model_name in models:
            rf = cond_dir / f"{model_name}.json"
            if not rf.exists():
                print(f"  {model_name}: FILE NOT FOUND, skipping")
                continue

            with open(rf) as f:
                data = json.load(f)

            responses = data.get("responses", [])
            grand_total += len(responses)

            already = sum(
                1 for r in responses
                if f"{model_name}_Q{r['question_id']}" in state
            )

            if args.dry_run:
                print(f"  {model_name}: {len(responses)} responses ({already} already graded)")
                continue

            to_grade = len(responses) - already
            print(f"\n  {model_name}: {to_grade} to grade ({already} cached)")

            for i, resp in enumerate(responses):
                qid = resp["question_id"]
                key = f"{model_name}_Q{qid}"

                # Resume support: use cached grade if available
                if key in state:
                    g = state[key]
                    all_rows.append({
                        "question_id": qid,
                        "category": resp["category"],
                        "model": model_name,
                        "accuracy": g["accuracy"],
                        "fluency": g["fluency"],
                        "completeness": g["completeness"],
                        "total": g["total"],
                        "response_time": resp.get("response_time_seconds", ""),
                        "response_length": len(
                            extract_final_answer(resp["response"], ck).split()
                            if is_pipeline else resp["response"].split()
                        ),
                        "notes": g.get("notes", ""),
                    })
                    continue

                # Grade this response
                g = grade_one(
                    client, system,
                    question_id=qid,
                    category=resp["category"],
                    question_text=resp["question_text"],
                    reference_answer=resp.get("reference_answer", ""),
                    response_text=resp["response"],
                    target_lang=target_lang,
                    condition=ck,
                    is_pipeline=is_pipeline,
                )

                state[key] = g
                grand_graded += 1

                # Compute response length (final answer only for C3/C5)
                if is_pipeline:
                    eval_text = extract_final_answer(resp["response"], ck)
                else:
                    eval_text = resp["response"]

                all_rows.append({
                    "question_id": qid,
                    "category": resp["category"],
                    "model": model_name,
                    "accuracy": g["accuracy"],
                    "fluency": g["fluency"],
                    "completeness": g["completeness"],
                    "total": g["total"],
                    "response_time": resp.get("response_time_seconds", ""),
                    "response_length": len(eval_text.split()),
                    "notes": g.get("notes", ""),
                })

                status = (
                    f"A={g['accuracy']} F={g['fluency']} "
                    f"C={g['completeness']} T={g['total']}"
                )
                note_str = f" | {g['notes']}" if g.get("notes") else ""
                print(
                    f"    Q{qid:02d} {resp['category']:20s} {status}{note_str}"
                )

                # Save state every 10 gradings
                if grand_graded % 10 == 0:
                    with open(state_file, "w") as f:
                        json.dump(state, f, ensure_ascii=False)

                time.sleep(0.3)  # Light rate limiting

            # Save state after each model
            with open(state_file, "w") as f:
                json.dump(state, f, ensure_ascii=False)

        if args.dry_run:
            continue

        # Sort by question_id, then model order
        all_rows.sort(
            key=lambda r: (
                r["question_id"],
                MODELS.index(r["model"]) if r["model"] in MODELS else 99,
            )
        )

        # Write CSV
        csv_path = GRADES_DIR / f"{meta['dir']}.csv"
        fields = [
            "question_id", "category", "model",
            "accuracy", "fluency", "completeness", "total",
            "response_time", "response_length", "notes",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)

        print(f"\n  -> Saved {csv_path} ({len(all_rows)} rows)")

        # Print condition summary
        valid = [r for r in all_rows if r["accuracy"] >= 0]
        if valid:
            avg_acc = sum(r["accuracy"] for r in valid) / len(valid)
            avg_flu = sum(r["fluency"] for r in valid) / len(valid)
            avg_com = sum(r["completeness"] for r in valid) / len(valid)
            avg_tot = sum(r["total"] for r in valid) / len(valid)
            print(f"  Averages: Acc={avg_acc:.2f} Flu={avg_flu:.2f} "
                  f"Comp={avg_com:.2f} Total={avg_tot:.2f}")

    if args.dry_run:
        print(f"\nTotal responses to grade: {grand_total}")
    else:
        print(f"\nDone! Graded {grand_graded} new responses.")
        print(f"CSV files saved in: {GRADES_DIR}/")


if __name__ == "__main__":
    main()
