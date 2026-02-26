# LLM Translation Project

Benchmarking LLMs for medical oncology translation. The pipeline translates drug indication descriptions from English into 8 languages, then evaluates quality through three complementary methods: NCI terminology matching, COMET/MetricX automatic metrics, and MCQ-based comprehension scoring.

---

## Input Data

**`descriptions.json`** — array of medical oncology drug description entries. Each entry has an `id` and an `en` (English) field.

---

## Pipeline Overview

```
descriptions.json
       │
       ├─► generate_translations.py          ─► results/descriptions_<provider>_test_N.json
       │                                              │
       └─► generate_translations_with_nci.py  ─► results/descriptions_<provider>_nci_test_N.json
                                                       │  (includes es_nci alongside all other langs)
                                                       │
                          ┌────────────────────────────┤
                          │                            │
                          ▼                            ▼
               match_with_nci.py           make_mcqs.py
               (terminology eval)          (generate MCQs from English)
                          │                            │
                          ▼                            ▼
               *_terminology_eval.json     mcqs_*.json
                                                       │
                                                       ▼
                                           evaluate_mcqs.py
                                           (translate + answer MCQs per language)
                                                       │
                                                       ▼
                                           eval_*.csv

               results/descriptions_*.json
                          │
                          ▼
               general_MT_benchmarking.py
               (COMET + MetricX scores)
```

---

## Files

### Translation

| File | Description |
|---|---|
| `generate_translations.py` | Core translation script. Translates each entry's `en` field into 8 languages. Output fields: `es`, `fr`, `ht`, `ar`, `so`, `ur`, `zh`, `yi`. |
| `generate_translations_with_nci.py` | Same as above, but also produces an `es_nci` field — a second Spanish translation where the prompt is augmented with a glossary of NCI-approved EN→ES terms found in the source text. |
| `translate_fields.py` | Earlier script that translates individual named fields (e.g. `description`) separately and writes one output file per field into `results_fields/`. Mostly superseded by `generate_translations.py`. |

**Supported providers** (set `MODEL_PROVIDER` at the top of each script): `openai`, `claude`, `gemini`, `translategemma` (via Ollama or HuggingFace).

### NCI Terminology

| File | Description |
|---|---|
| `match_with_nci.py` | Loads NCI cancer and genetic term dictionaries (EN→ES), checks which gold-standard Spanish terms appear correctly in each translation, and reports per-entry terminology accuracy. For missed terms, it also re-translates them in isolation and computes a fuzzy match score. |

NCI dictionary paths (from `match_with_nci.py`):
- `/home/baihesun/scrape-NCI-dictionaries/data/processed/cancer-terms.json`
- `/home/baihesun/scrape-NCI-dictionaries/data/processed/genetic-terms.json`

### MCQ Evaluation

| File | Description |
|---|---|
| `make_mcqs.py` | Generates N multiple-choice questions per entry from the English source text. Questions are designed to hinge on specific medical terms and clinical details. Output: `mcqs_<stem>_<provider>_n<N>.json`. |
| `evaluate_mcqs.py` | For each language, translates the MCQs into that language, then prompts the model to answer them using the translated passage. Accuracy = fraction of questions answered correctly. Output: `eval_<stem>_<provider>.csv`. Also handles `es_nci` as a separate column. |

### Automatic MT Metrics

| File | Description |
|---|---|
| `general_MT_benchmarking.py` | Runs **COMET** (`wmt22-cometkiwi-da`, reference-free) and **MetricX-24** (QE mode, lower = better) on all language columns present in a translations file. Scores all languages including `es_nci` automatically. |

---

## Results Directory

Output files follow a naming convention:

| Pattern | Contents |
|---|---|
| `descriptions_<provider>_test_N.json` | Translations (no NCI), N entries |
| `descriptions_<provider>_nci_test_N.json` | Translations + `es_nci`, N entries |
| `*_terminology_eval.json` | NCI term match/miss results |
| `mcqs_<stem>_<provider>_nN.json` | Generated MCQs |
| `eval_<stem>_<provider>.csv` | MCQ accuracy per language |

---

## Setup

```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
```

Set API keys as environment variables before running:
```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
```
