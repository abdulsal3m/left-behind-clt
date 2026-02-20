#!/usr/bin/env python3
"""
Compute automated metrics from experiment results.

Metrics computed:
1. Response Length — character count and estimated token count
2. Language Consistency — ratio of target-language script vs English in response
3. Script Correctness — whether the model uses the correct writing system
4. Code-Switching Detection — identifies mid-response language switches
5. Translation Step Compliance (C3/C5 only) — whether the model showed all 3 steps

Usage:
    python compute_metrics.py                  # Process all results, print summary
    python compute_metrics.py --export         # Also export enriched JSON + CSV
    python compute_metrics.py --condition C2   # Only process specific condition(s)
"""

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Character classification helpers
# ---------------------------------------------------------------------------

# Cyrillic Unicode block: U+0400 to U+04FF (covers Russian, Kazakh, Mongolian)
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
LATIN_RE = re.compile(r"[a-zA-Z]")

# Kazakh-specific Cyrillic characters (not in Russian)
KAZAKH_SPECIFIC = set("ӘәҒғҚқҢңӨөҰұҮүҺһІі")

# Mongolian-specific Cyrillic characters (not in Russian)
MONGOLIAN_SPECIFIC = set("ӨөҮү")

# Common English words to detect code-switching
ENGLISH_MARKERS = re.compile(
    r"\b(the|is|are|was|were|and|or|but|this|that|with|for|from|have|has|"
    r"which|would|could|should|about|because|however|therefore|although|"
    r"answer|question|translation|translate|step)\b",
    re.IGNORECASE,
)


def count_cyrillic(text: str) -> int:
    return len(CYRILLIC_RE.findall(text))


def count_latin(text: str) -> int:
    return len(LATIN_RE.findall(text))


def cyrillic_ratio(text: str) -> float:
    """Ratio of Cyrillic chars to (Cyrillic + Latin). 1.0 = all Cyrillic."""
    cyr = count_cyrillic(text)
    lat = count_latin(text)
    total = cyr + lat
    if total == 0:
        return 0.0
    return cyr / total


def has_kazakh_specific_chars(text: str) -> bool:
    return bool(KAZAKH_SPECIFIC & set(text))


def has_mongolian_specific_chars(text: str) -> bool:
    return bool(MONGOLIAN_SPECIFIC & set(text))


# ---------------------------------------------------------------------------
# Step 3 extraction for C3/C5
# ---------------------------------------------------------------------------

# Patterns models use to mark step 3 (the final translated answer)
STEP3_PATTERNS = [
    # Numbered headers: "## 3-қадам", "## Алхам 3", "3.", "**3.**", "### Step 3"
    r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:3[\-\.:\s]?\s*(?:қадам|алхам|step|адым)|(?:қадам|алхам|step|адым)\s*3)",
    # Kazakh markers
    r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:қазақ тіліне аудар|аудармасы|қазақша жауап)",
    # Mongolian markers
    r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:монгол хэл рүү орчуул|орчуулга|монголоор хариул)",
    # Generic "Translation:" or "Final answer:"
    r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:translation|final answer|translated answer)",
]
STEP3_RE = re.compile("|".join(STEP3_PATTERNS), re.IGNORECASE | re.MULTILINE)


def extract_final_answer(response: str, condition: str) -> str:
    """
    For C3/C5, extract only the step 3 (final translated answer).
    For other conditions, return the full response.
    """
    if condition not in ("C3", "C5"):
        return response

    match = STEP3_RE.search(response)
    if match:
        # Everything after the step 3 header
        return response[match.end():].strip()

    # Fallback: take the last third of the response (heuristic)
    lines = response.strip().split("\n")
    if len(lines) >= 6:
        return "\n".join(lines[-(len(lines) // 3):]).strip()

    return response


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------


def compute_response_length(response: str, condition: str = "C1") -> dict:
    """
    Compute character count and estimated token count.
    For C3/C5, measures the FINAL ANSWER only (step 3), not the full multi-step output.
    """
    final_answer = extract_final_answer(response, condition)
    char_count = len(final_answer)
    # Rough token estimate: ~4 chars/token for English, ~2-3 for Cyrillic
    cyr_ratio = cyrillic_ratio(final_answer)
    chars_per_token = 2.5 if cyr_ratio > 0.5 else 4.0
    estimated_tokens = round(char_count / chars_per_token)
    word_count = len(final_answer.split())
    return {
        "char_count": char_count,
        "word_count": word_count,
        "estimated_tokens": estimated_tokens,
        "full_response_char_count": len(response),
        "measured_final_answer_only": condition in ("C3", "C5"),
    }


def compute_language_consistency(response: str, condition: str) -> dict:
    """
    Measure whether the response is in the expected language.

    For C1: expect mostly Latin (English)
    For C2/C4: expect mostly Cyrillic (Kazakh/Mongolian)
    For C3/C5: expect mixed (intermediate English + final target language)
    """
    cyr = count_cyrillic(response)
    lat = count_latin(response)
    ratio = cyrillic_ratio(response)

    if condition == "C1":
        expected = "latin"
        is_consistent = ratio < 0.1  # mostly Latin
    elif condition in ("C2", "C4"):
        expected = "cyrillic"
        is_consistent = ratio > 0.7  # mostly Cyrillic
    else:  # C3, C5
        expected = "mixed"
        # For transfer conditions, we expect both scripts present
        is_consistent = 0.15 < ratio < 0.85  # meaningful mix of both

    return {
        "cyrillic_chars": cyr,
        "latin_chars": lat,
        "cyrillic_ratio": round(ratio, 3),
        "expected_script": expected,
        "language_consistent": is_consistent,
    }


def compute_script_correctness(response: str, condition: str, language: str) -> dict:
    """
    Check if the model uses the correct writing system.

    Detects:
    - Latin script when Cyrillic is expected
    - Wrong Cyrillic variant (e.g., Russian chars only, missing Kazakh-specific)
    """
    if condition == "C1":
        return {"script_correct": True, "script_issues": []}

    issues = []
    ratio = cyrillic_ratio(response)

    if condition in ("C2", "C4"):
        # Direct response should be in target language
        if ratio < 0.5:
            issues.append("response_mostly_latin")

        if language == "kazakh" and ratio > 0.3:
            if not has_kazakh_specific_chars(response):
                issues.append("missing_kazakh_specific_chars")

    elif condition in ("C3", "C5"):
        # The final section (step 3) should be in target language
        # Try to find the last major section
        lines = response.strip().split("\n")
        last_section = "\n".join(lines[-max(3, len(lines) // 3) :])
        last_ratio = cyrillic_ratio(last_section)

        if last_ratio < 0.3:
            issues.append("final_answer_not_in_target_language")

    return {
        "script_correct": len(issues) == 0,
        "script_issues": issues,
    }


def compute_code_switching(response: str, condition: str) -> dict:
    """
    Detect sentence-level language switching within the response.

    Splits response into sentences and checks if language switches occur.
    Most relevant for C2/C4 where we expect monolingual responses.
    """
    if condition == "C1":
        return {"code_switch_count": 0, "code_switch_ratio": 0.0}

    # Split into sentences (rough)
    sentences = re.split(r"[.!?।。\n]+", response)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if len(sentences) < 2:
        return {"code_switch_count": 0, "code_switch_ratio": 0.0}

    switches = 0
    prev_is_cyrillic = None

    for sent in sentences:
        ratio = cyrillic_ratio(sent)
        is_cyrillic = ratio > 0.5
        if prev_is_cyrillic is not None and is_cyrillic != prev_is_cyrillic:
            switches += 1
        prev_is_cyrillic = is_cyrillic

    return {
        "code_switch_count": switches,
        "code_switch_ratio": round(switches / max(len(sentences) - 1, 1), 3),
        "sentence_count": len(sentences),
    }


def compute_transfer_compliance(response: str, condition: str) -> dict:
    """
    For C3/C5 only: check if the model followed the 3-step instruction.

    Looks for evidence of:
    1. English translation of the question
    2. English answer
    3. Back-translation to target language
    """
    if condition not in ("C3", "C5"):
        return {"transfer_applicable": False}

    response_lower = response.lower()

    # Check for step markers (numbered steps, headers, etc.)
    has_step_markers = bool(
        re.search(r"(step\s*[123]|қадам\s*[123]|алхам\s*[123]|1\.|2\.|3\.)", response_lower)
    )

    # Check for presence of both scripts in meaningful amounts
    ratio = cyrillic_ratio(response)
    has_english_section = count_latin(response) > 50
    has_target_section = count_cyrillic(response) > 50

    # Check for translation-related keywords
    translation_keywords = re.compile(
        r"(translat|аудар|орчуул)", re.IGNORECASE
    )
    has_translation_mention = bool(translation_keywords.search(response))

    all_steps_present = has_step_markers and has_english_section and has_target_section

    return {
        "transfer_applicable": True,
        "has_step_markers": has_step_markers,
        "has_english_section": has_english_section,
        "has_target_section": has_target_section,
        "has_translation_mention": has_translation_mention,
        "all_steps_present": all_steps_present,
    }


def compute_english_word_leakage(response: str, condition: str) -> dict:
    """
    For C2/C4: count English words that leak into target-language responses.
    Excludes proper nouns, technical terms, and brand names.
    """
    if condition not in ("C2", "C4"):
        return {"english_leakage_applicable": False}

    matches = ENGLISH_MARKERS.findall(response)
    return {
        "english_leakage_applicable": True,
        "english_word_count": len(matches),
        "english_words_found": list(set(w.lower() for w in matches)),
    }


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def process_response(resp: dict, condition: str, language: str) -> dict:
    """Compute all metrics for a single response."""
    text = resp.get("response", "")
    if text.startswith("[ERROR]"):
        return {"error": True}

    metrics = {"error": False}
    metrics.update(compute_response_length(text, condition))
    metrics.update(compute_language_consistency(text, condition))
    metrics.update(compute_script_correctness(text, condition, language))
    metrics.update(compute_code_switching(text, condition))
    metrics.update(compute_transfer_compliance(text, condition))
    metrics.update(compute_english_word_leakage(text, condition))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compute automated metrics")
    parser.add_argument("--export", action="store_true", help="Export enriched results + metrics CSV")
    parser.add_argument("--condition", nargs="+", help="Only process specific condition(s)")
    args = parser.parse_args()

    if not RESULTS_DIR.exists():
        print("No results directory found. Run experiments first.")
        return

    # Collect all metrics
    all_metrics = []

    for cond_dir in sorted(RESULTS_DIR.iterdir()):
        if not cond_dir.is_dir():
            continue
        for result_file in sorted(cond_dir.glob("*.json")):
            with open(result_file) as f:
                data = json.load(f)

            condition = data.get("condition", "")
            if args.condition and condition not in args.condition:
                continue

            language = data.get("language", "")
            model = data.get("model", result_file.stem)

            for resp in data.get("responses", []):
                metrics = process_response(resp, condition, language)
                metrics["question_id"] = resp["question_id"]
                metrics["category"] = resp["category"]
                metrics["model"] = model
                metrics["condition"] = condition
                metrics["language"] = language
                metrics["response_time_seconds"] = resp.get("response_time_seconds")
                all_metrics.append(metrics)

            # Export enriched JSON if requested
            if args.export:
                for resp in data.get("responses", []):
                    m = process_response(resp, condition, language)
                    resp["metrics"] = m
                enriched_path = result_file.parent / f"{result_file.stem}_enriched.json"
                with open(enriched_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

    if not all_metrics:
        print("No results to analyze.")
        return

    # ---------------------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------------------

    valid = [m for m in all_metrics if not m.get("error")]
    print(f"Total responses analyzed: {len(valid)}")
    print()

    # --- Response Length by Condition ---
    print("=" * 65)
    print("RESPONSE LENGTH BY CONDITION")
    print("=" * 65)
    for cond in ["C1", "C2", "C3", "C4", "C5"]:
        subset = [m for m in valid if m["condition"] == cond]
        if not subset:
            continue
        avg_chars = sum(m["char_count"] for m in subset) / len(subset)
        avg_words = sum(m["word_count"] for m in subset) / len(subset)
        avg_tokens = sum(m["estimated_tokens"] for m in subset) / len(subset)
        print(f"  {cond}: avg {avg_chars:.0f} chars | {avg_words:.0f} words | ~{avg_tokens:.0f} tokens")
    print()

    # --- Response Length by Model ---
    print("=" * 65)
    print("RESPONSE LENGTH BY MODEL")
    print("=" * 65)
    models = sorted(set(m["model"] for m in valid))
    for model in models:
        subset = [m for m in valid if m["model"] == model]
        avg_chars = sum(m["char_count"] for m in subset) / len(subset)
        print(f"  {model:25s}: avg {avg_chars:.0f} chars")
    print()

    # --- Language Consistency ---
    print("=" * 65)
    print("LANGUAGE CONSISTENCY (% responses in expected script)")
    print("=" * 65)
    for cond in ["C1", "C2", "C3", "C4", "C5"]:
        subset = [m for m in valid if m["condition"] == cond]
        if not subset:
            continue
        consistent = sum(1 for m in subset if m.get("language_consistent"))
        pct = consistent / len(subset) * 100
        avg_ratio = sum(m.get("cyrillic_ratio", 0) for m in subset) / len(subset)
        print(f"  {cond}: {consistent}/{len(subset)} consistent ({pct:.0f}%) | avg Cyrillic ratio: {avg_ratio:.2f}")
    print()

    # --- Language Consistency by Model (C2+C4 only) ---
    print("=" * 65)
    print("LANGUAGE CONSISTENCY BY MODEL (C2+C4 — direct low-resource)")
    print("=" * 65)
    for model in models:
        subset = [m for m in valid if m["model"] == model and m["condition"] in ("C2", "C4")]
        if not subset:
            continue
        consistent = sum(1 for m in subset if m.get("language_consistent"))
        pct = consistent / len(subset) * 100
        print(f"  {model:25s}: {consistent}/{len(subset)} consistent ({pct:.0f}%)")
    print()

    # --- Script Correctness ---
    print("=" * 65)
    print("SCRIPT CORRECTNESS")
    print("=" * 65)
    for cond in ["C2", "C3", "C4", "C5"]:
        subset = [m for m in valid if m["condition"] == cond]
        if not subset:
            continue
        correct = sum(1 for m in subset if m.get("script_correct"))
        pct = correct / len(subset) * 100
        issues = {}
        for m in subset:
            for issue in m.get("script_issues", []):
                issues[issue] = issues.get(issue, 0) + 1
        issue_str = ", ".join(f"{k}: {v}" for k, v in issues.items()) if issues else "none"
        print(f"  {cond}: {correct}/{len(subset)} correct ({pct:.0f}%) | issues: {issue_str}")
    print()

    # --- Code-Switching (C2+C4) ---
    print("=" * 65)
    print("CODE-SWITCHING (C2+C4 — should be monolingual)")
    print("=" * 65)
    for model in models:
        subset = [m for m in valid if m["model"] == model and m["condition"] in ("C2", "C4")]
        if not subset:
            continue
        avg_switches = sum(m.get("code_switch_count", 0) for m in subset) / len(subset)
        zero_switch = sum(1 for m in subset if m.get("code_switch_count", 0) == 0)
        print(f"  {model:25s}: avg {avg_switches:.1f} switches/response | {zero_switch}/{len(subset)} fully monolingual")
    print()

    # --- Transfer Step Compliance (C3+C5) ---
    print("=" * 65)
    print("TRANSFER STEP COMPLIANCE (C3+C5 — did model show all 3 steps?)")
    print("=" * 65)
    for model in models:
        subset = [m for m in valid if m["model"] == model and m["condition"] in ("C3", "C5") and m.get("transfer_applicable")]
        if not subset:
            continue
        compliant = sum(1 for m in subset if m.get("all_steps_present"))
        pct = compliant / len(subset) * 100
        print(f"  {model:25s}: {compliant}/{len(subset)} showed all 3 steps ({pct:.0f}%)")
    print()

    # --- English Word Leakage (C2+C4) ---
    print("=" * 65)
    print("ENGLISH WORD LEAKAGE (C2+C4 — English words in target-language responses)")
    print("=" * 65)
    for model in models:
        subset = [m for m in valid if m["model"] == model and m.get("english_leakage_applicable")]
        if not subset:
            continue
        avg_leakage = sum(m.get("english_word_count", 0) for m in subset) / len(subset)
        zero_leakage = sum(1 for m in subset if m.get("english_word_count", 0) == 0)
        print(f"  {model:25s}: avg {avg_leakage:.1f} English words/response | {zero_leakage}/{len(subset)} leak-free")
    print()

    # --- Response Time by Model and Condition ---
    timed = [m for m in valid if m.get("response_time_seconds") is not None]
    if timed:
        print("=" * 65)
        print("RESPONSE TIME (seconds)")
        print("=" * 65)
        # By model
        for model in models:
            subset = [m for m in timed if m["model"] == model]
            if not subset:
                continue
            times = [m["response_time_seconds"] for m in subset]
            avg_t = sum(times) / len(times)
            min_t = min(times)
            max_t = max(times)
            print(f"  {model:25s}: avg {avg_t:.1f}s | min {min_t:.1f}s | max {max_t:.1f}s")
        print()
        # By condition
        for cond in ["C1", "C2", "C3", "C4", "C5"]:
            subset = [m for m in timed if m["condition"] == cond]
            if not subset:
                continue
            avg_t = sum(m["response_time_seconds"] for m in subset) / len(subset)
            print(f"  {cond}: avg {avg_t:.1f}s")
        print()

    # --- Export CSV ---
    if args.export:
        csv_path = Path(__file__).parent / "metrics_report.csv"
        fieldnames = [
            "question_id", "category", "model", "condition", "language",
            "response_time_seconds", "char_count", "word_count", "estimated_tokens",
            "cyrillic_ratio", "language_consistent", "script_correct",
            "script_issues", "code_switch_count", "code_switch_ratio",
            "all_steps_present", "english_word_count",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for m in valid:
                row = dict(m)
                row["script_issues"] = "; ".join(m.get("script_issues", []))
                writer.writerow(row)
        print(f"Metrics CSV exported to: {csv_path}")
        print(f"Enriched JSON files saved alongside originals in results/")


if __name__ == "__main__":
    main()
