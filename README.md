# Left Behind: Cross-Lingual Transfer for Low-Resource Languages in LLMs

Data, code, and paper for our study benchmarking 8 large language models across English, Kazakh, and Mongolian.

## Overview

We evaluate how LLMs perform on low-resource languages by testing 8 models across 5 experimental conditions using 50 hand-crafted questions with human-authored parallel versions in English, Kazakh, and Mongolian. The study produces 2,000 graded responses and reveals a consistent 13.8--16.7 percentage point performance gap between English and low-resource language conditions.

### Key Findings

- **Performance gap**: Models score 13.8--16.7pp lower when prompted in Kazakh or Mongolian vs. English
- **Fluency illusion**: Models maintain surface-level fluency while producing significantly less accurate content
- **Selective CLT benefit**: Cross-lingual transfer helps bilingual models (+2.2pp to +4.3pp) but not English-dominant ones
- **Multilingual failure**: Aya Expanse (designed for 100+ languages) collapses to 15.7% on Kazakh and 4.7% on Mongolian, often producing Kyrgyz instead of Kazakh

## Models Evaluated

| Category | Models |
|---|---|
| English-First | Claude Opus 4.5, GPT-5.2, Gemini 3 Pro, Grok 4.1, Llama 4 Maverick |
| Bilingual | Qwen3, DeepSeek V3.2 |
| Multilingual | Aya Expanse |

## Experimental Conditions

| Condition | Description |
|---|---|
| C1 | English baseline (EN → EN) |
| C2 | Kazakh direct (KZ → KZ) |
| C3 | Kazakh cross-lingual transfer (KZ → EN → EN → KZ) |
| C4 | Mongolian direct (MN → MN) |
| C5 | Mongolian cross-lingual transfer (MN → EN → EN → MN) |

## Repository Structure

```
benchmark/          50 questions in 3 languages + reference answers
  questions.json                    Structured JSON (machine-readable)
  Cross_lingual_transfer_benchmark_questions.txt         Plain text
  Cross_lingual_transfer_benchmark_questions_answer_sheet.txt  Reference answers

results/            Raw model responses (8 models x 5 conditions = 40 JSON files)
  C1_english/
  C2_kazakh/
  C3_kazakh_transfer/
  C4_mongolian/
  C5_mongolian_transfer/

grades/             Graded responses (Accuracy, Fluency, Completeness per response)
  C1_english.csv
  C2_kazakh.csv
  C3_kazakh_transfer.csv
  C4_mongolian.csv
  C5_mongolian_transfer.csv

scripts/            Experiment and analysis code
  run_experiments.py      Run all model-condition combinations
  grade_responses.py      Auto-grade responses using LLM-as-judge
  compute_metrics.py      Compute aggregate metrics from grades
  requirements.txt        Python dependencies

paper/              ACL-formatted LaTeX source
  main.tex
  references.bib
  figs/
```

## Scoring Rubric

Each response is scored on three dimensions (0--2 each, max 6 per response):

- **Accuracy** (0--2): Factual correctness against reference answers
- **Fluency** (0--2): Grammatical quality and naturalness in the target language; wrong language = 0
- **Completeness** (0--2): Coverage of key points from the reference answer

Grading is performed using Claude Sonnet 4.5 following the LLM-as-judge paradigm, with the complete reference answer sheet provided as context.

## Results Summary

| Condition | Mean Score |
|---|---|
| C1: English | 90.7% |
| C2: Kazakh Direct | 76.9% |
| C3: Kazakh CLT | 77.2% |
| C4: Mongolian Direct | 74.0% |
| C5: Mongolian CLT | 74.6% |

## Citation

If you use this benchmark or data, please cite:

```bibtex
@inproceedings{beibitkhan2026leftbehind,
  title={Left Behind: Cross-Lingual Transfer as a Bridge for Low-Resource Languages in Large Language Models},
  author={Beibitkhan, Abdul-Salem},
  year={2026}
}
```

## License

This dataset and code are released for research purposes.
