import json
import os
import re
import sys
from nltk.stem import SnowballStemmer
from fuzzywuzzy import fuzz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_translations import (
    translate_field, get_client, MODEL_CONFIGS, MODEL_PROVIDER, TEMPERATURE,
    USE_OLLAMA_FOR_TRANSLATEGEMMA,
)

stemmer = SnowballStemmer("spanish")

# --- Paths ---
INPUT_FILE = "/home/baihesun/LLM_translation_project/results/descriptions_openai_test_5.json"
CANCER_TERMS_FILE = "/home/baihesun/scrape-NCI-dictionaries/data/processed/cancer-terms.json"
GENETIC_TERMS_FILE = "/home/baihesun/scrape-NCI-dictionaries/data/processed/genetic-terms.json"

def term_in_text_en(term, text):
    """
    word boundary match for English terms, + unicode-aware boundaries to avoid partial matches
    (e.g. 'ac' matching inside 'taco').
    """
    pattern = r'(?<![\w\u00C0-\u024F])' + re.escape(term) + r'(?![\w\u00C0-\u024F])'
    return bool(re.search(pattern, text, re.IGNORECASE))


def term_in_text_es(term, text):
    """
    stemmed sequence match for Spanish terms.
    -- handles inflected/gendered forms (e.g. 'endocrina' matching 'endocrine')
    -- in multi-word phrases (e.g. 'cáncer de mama'), looks for each word 
    
    Approach:
    1. Split both term and text into individual words
    2. Stem each word to deal with gendered adjectives / conjugations ('cáncer' -> 'canc')
    3. Check if the stemmed term words appear as a consecutive sequence anywhere in the stemmed text words
    """
    term_words = re.findall(r'[\w\u00C0-\u024F]+', term, re.IGNORECASE)
    text_words = re.findall(r'[\w\u00C0-\u024F]+', text, re.IGNORECASE)

    stemmed_term = [stemmer.stem(w) for w in term_words]
    stemmed_text = [stemmer.stem(w) for w in text_words]

    # slide a window of len(stemmed_term) over stemmed_text and look for a match
    n = len(stemmed_term)
    return any(stemmed_text[i:i+n] == stemmed_term for i in range(len(stemmed_text) - n + 1))

def load_term_dict(*dict_files):
    """Load and merge NCI dictionary files into {en_term: es_term} mapping."""
    term_dict = {}
    for path in dict_files:
        with open(path, "r") as f:
            entries = json.load(f)
        for entry in entries:
            en = entry.get("term_en") or ""
            es = entry.get("term_es") or ""
            en = en.strip().lower()
            es = es.strip().lower()
            if en and es:
                term_dict[en] = es
    print(f"Loaded {len(term_dict)} terms from {len(dict_files)} dictionaries")
    return term_dict


def evaluate_entry(en_text, es_text, term_dict):
    """Check which gold standard terms appear correctly in the Spanish translation."""
    en_lower = en_text.lower()
    es_lower = es_text.lower()

    matched, missed = [], []

    for en_term, es_gold in term_dict.items():
        if term_in_text_en(en_term, en_lower):
            if term_in_text_es(es_gold, es_lower):
                matched.append({"en": en_term, "es_gold": es_gold})
            else:
                missed.append({"en": en_term, "es_gold": es_gold})

    total = len(matched) + len(missed)
    accuracy = len(matched) / total if total > 0 else None
    return {"matched": matched, "missed": missed, "total": total, "accuracy": accuracy}


def main():
    term_dict = load_term_dict(CANCER_TERMS_FILE, GENETIC_TERMS_FILE)

    # Set up translation client for re-translating missed terms in isolation.
    # NOTE: es_translated reflects what the model produces for the term alone,
    # not the actual word used in the full sentence (which we can't recover
    # without word alignment). A high fuzzy_score means the model knows the
    # NCI term when prompted directly; a low score suggests a genuine gap.
    config = MODEL_CONFIGS[MODEL_PROVIDER]
    if MODEL_PROVIDER == "translategemma":
        model_name = config["ollama_model"] if USE_OLLAMA_FOR_TRANSLATEGEMMA else config["hf_model"]
    else:
        model_name = config["model"]
    client = get_client(MODEL_PROVIDER, config)
    translation_cache = {}  # avoid redundant API calls for repeated terms

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    all_en_text = " ".join(e.get("en", "").lower() for e in data)
    term_dict = {k: v for k, v in term_dict.items() if k in all_en_text}
    print(f"Filtered to {len(term_dict)} relevant terms")

    results = []
    all_matched, all_total = 0, 0

    for entry in data:
        en_text = entry.get("en", "")
        es_text = entry.get("es", "")

        if not en_text or not es_text:
            continue

        eval_result = evaluate_entry(en_text, es_text, term_dict)
        all_matched += len(eval_result["matched"])
        all_total += eval_result["total"]

        for missed_item in eval_result["missed"]:
            en_term = missed_item["en"]
            es_gold = missed_item["es_gold"]

            if en_term not in translation_cache:
                es_translated = translate_field(
                    client, en_term, "Spanish", MODEL_PROVIDER, model_name, TEMPERATURE
                ).strip().lower()
                translation_cache[en_term] = es_translated
            else:
                es_translated = translation_cache[en_term]

            missed_item["es_translated"] = es_translated
            missed_item["fuzzy_score"] = fuzz.token_set_ratio(es_translated, es_gold)

        results.append({
            "id": entry.get("id"),
            "accuracy": eval_result["accuracy"],
            "matched_count": len(eval_result["matched"]),
            "total_terms_found": eval_result["total"],
            "matched": eval_result["matched"],
            "missed": eval_result["missed"],
        })

        acc = f"{eval_result['accuracy']:.0%}" if eval_result['accuracy'] is not None else "N/A"
        print(f"{entry.get('id')}: {len(eval_result['matched'])}/{eval_result['total']} terms matched ({acc})")

    overall = all_matched / all_total if all_total > 0 else 0
    print(f"\nOverall: {all_matched}/{all_total} terms matched ({overall:.0%})")

    output_path = INPUT_FILE.replace(".json", "_terminology_eval.json")
    with open(output_path, "w") as f:
        json.dump({"overall_accuracy": overall, "entries": results}, f, indent=2, ensure_ascii=False)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()