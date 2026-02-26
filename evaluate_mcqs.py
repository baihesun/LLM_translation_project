import json
import os
import time
import sys
import argparse
import csv


# CONFIGURATION
MCQ_PATH = "/home/baihesun/LLM_translation_project/results/mcqs_descriptions_openai_nci_test_5_openai_n3.json"
TRANSLATIONS_PATH = "/home/baihesun/LLM_translation_project/results/descriptions_openai_nci_test_5.json"
OUTPUT_DIR = "/home/baihesun/LLM_translation_project/results/"
MODEL_PROVIDER = "openai"
TEMPERATURE = 0.1

LANG_NAMES = {
    "es": "Spanish",
    "fr": "French",
    "ht": "Haitian Creole",
    "ar": "Arabic",
    "so": "Somali",
    "ur": "Urdu",
    "zh": "Chinese",
    "yi": "Yiddish",
}

MODEL_CONFIGS = {
    "openai": {"model": "gpt-5.2", "api_key_env": "OPENAI_API_KEY"},
    "claude": {"model": "claude-sonnet-4-6", "api_key_env": "ANTHROPIC_API_KEY"},
    "gemini": {"model": "gemini-2.5-flash-lite", "api_key_env": "GOOGLE_API_KEY"},
}


def get_client(provider, config):
    api_key = os.environ.get(config["api_key_env"])
    if not api_key:
        raise ValueError(f"Set {config['api_key_env']} environment variable")

    if provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    elif provider == "claude":
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    elif provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(config["model"])


def _extract_json_object(raw):
    """Extract first {...} block from raw string."""
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in: {raw}")
    return json.loads(raw[start:end])


def translate_mcq(client, mcq, lang_name, provider, model, temperature):
    """Translate a MCQ's question and options to lang_name, preserving A/B/C/D keys."""
    if lang_name == "es_nci":
        lang_name = "es"
    mcq_input = {k: mcq[k] for k in ["question", "A", "B", "C", "D"]}
    prompt = (
        f"Translate the following multiple-choice question and its answer options from English to {lang_name}. "
        f"Use standard medical terminology. Output only a JSON object with keys: "
        f'"question", "A", "B", "C", "D". Do not include any other text.\n\n'
        f"{json.dumps(mcq_input, ensure_ascii=False)}"
    )

    if provider == "openai":
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content)
    elif provider == "claude":
        response = client.messages.create(
            model=model, max_tokens=1024, temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = _extract_json_object(response.content[0].text)
    elif provider == "gemini":
        response = client.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": 1024},
        )
        parsed = _extract_json_object(response.text)

    return {k: parsed.get(k, mcq[k]) for k in ["question", "A", "B", "C", "D"]}


def answer_mcq(client, passage, translated_mcq, lang_name, provider, model, temperature):
    """Ask the model to answer a translated MCQ using the translated passage."""
    prompt = (
        f"Read the following passage in {lang_name} and answer the multiple-choice question. "
        f"Reply with ONLY the letter of the correct answer: A, B, C, or D.\n\n"
        f"Passage:\n{passage}\n\n"
        f"Question: {translated_mcq['question']}\n"
        f"A: {translated_mcq['A']}\n"
        f"B: {translated_mcq['B']}\n"
        f"C: {translated_mcq['C']}\n"
        f"D: {translated_mcq['D']}\n\n"
        f"Answer (A, B, C, or D):"
    )

    if provider == "openai":
        raw = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,
            max_completion_tokens=5,
        ).choices[0].message.content.strip()
    elif provider == "claude":
        raw = client.messages.create(
            model=model, max_tokens=5, temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        ).content[0].text.strip()
    elif provider == "gemini":
        raw = client.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": 5},
        ).text.strip()

    for ch in raw.upper():
        if ch in "ABCD":
            return ch
    return None


def evaluate_entry(client, entry, translations_by_id, lang_codes, provider, model, temperature):
    """For one entry, evaluate accuracy of each language. Returns {lang_code: float|None}."""
    trans_entry = translations_by_id.get(entry["id"], {})
    mcqs = entry.get("mcqs", [])
    results = {}

    for lang_code, lang_name in lang_codes.items():
        passage = trans_entry.get(lang_code, "")
        if not passage or not mcqs:
            results[lang_code] = None
            continue

        correct = 0
        for mcq in mcqs:
            try:
                translated_mcq = translate_mcq(client, mcq, lang_name, provider, model, temperature)
                answer = answer_mcq(client, passage, translated_mcq, lang_name, provider, model, temperature)
                if answer == mcq["answer"]:
                    correct += 1
            except Exception as e:
                print(f"\n  Error ({entry['id']}, {lang_code}): {e}")

        results[lang_code] = correct / len(mcqs)

    return results


def print_table(all_rows, lang_codes):
    id_w = max(len(r["id"]) for r in all_rows) + 2
    col_w = 16
    header = f"{'ID':<{id_w}}" + "".join(f"{LANG_NAMES[lc]:<{col_w}}" for lc in lang_codes)
    print("\n" + header)
    print("-" * len(header))
    for row in all_rows:
        cells = [
            f"{row[lc]:.2f}" if row[lc] is not None else "N/A"
            for lc in lang_codes
        ]
        print(f"{row['id']:<{id_w}}" + "".join(f"{c:<{col_w}}" for c in cells))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MCQ accuracy for translated medical passages."
    )
    parser.add_argument("--mcqs", default=MCQ_PATH, help="Path to MCQ JSON file")
    parser.add_argument("--translations", default=TRANSLATIONS_PATH, help="Path to translations JSON file")
    parser.add_argument("--output", default=None, help="Output CSV path (default: auto-named in results/)")
    parser.add_argument("--provider", default=MODEL_PROVIDER, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--test-size", type=int, default=None, help="Only process first N entries")
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.provider]
    model_name = config["model"]

    output_path = args.output
    if output_path is None:
        stem = os.path.splitext(os.path.basename(args.mcqs))[0]
        output_path = os.path.join(OUTPUT_DIR, f"eval_{stem}_{args.provider}.csv")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Provider:      {args.provider} ({model_name})")
    print(f"MCQs:          {args.mcqs}")
    print(f"Translations:  {args.translations}")
    print(f"Output:        {output_path}\n")

    client = get_client(args.provider, config)

    with open(args.mcqs) as f:
        mcq_data = json.load(f)
    with open(args.translations) as f:
        trans_data = json.load(f)

    translations_by_id = {e["id"]: e for e in trans_data}

    # Detect languages present in the translations file
    sample = trans_data[0]
    lang_codes = {k: LANG_NAMES[k] for k in sample if k in LANG_NAMES}

    if args.test_size:
        mcq_data = mcq_data[:args.test_size]

    print(f"Entries: {len(mcq_data)} | Languages: {', '.join(lang_codes.values())}\n")

    all_rows = []
    start = time.time()

    for i, entry in enumerate(mcq_data):
        lang_results = evaluate_entry(
            client, entry, translations_by_id, lang_codes,
            args.provider, model_name, TEMPERATURE,
        )
        row = {"id": entry["id"], **lang_results}
        all_rows.append(row)

        elapsed = time.time() - start
        remaining = (elapsed / (i + 1)) * (len(mcq_data) - i - 1)
        sys.stdout.write(
            f"\r{i+1}/{len(mcq_data)} ({100*(i+1)/len(mcq_data):.1f}%) | "
            f"{elapsed:.0f}s elapsed | {remaining:.0f}s remaining"
        )
        sys.stdout.flush()

    print()

    lang_code_list = list(lang_codes.keys())

    # Save CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id"] + [LANG_NAMES[lc] for lc in lang_code_list])
        for row in all_rows:
            writer.writerow(
                [row["id"]] + [
                    f"{row[lc]:.2f}" if row[lc] is not None else "N/A"
                    for lc in lang_code_list
                ]
            )

    print(f"Saved to {output_path}")

    print_table(all_rows, lang_code_list)


if __name__ == "__main__":
    main()
