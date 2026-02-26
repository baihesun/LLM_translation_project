import json
import os
import re
import time
import sys


# CONFIGURATION
INDICATIONS_PATH = "/home/baihesun/LLM_translation_project/descriptions.json"
OUTPUT_DIR = "/home/baihesun/LLM_translation_project/results/"
LANGUAGES = ["Spanish", "French", "Haitian Creole", "Arabic", "Somali", "Urdu", "Chinese", "Yiddish"]
MODEL_PROVIDER = "openai"  # Options: "openai", "claude", "gemini", "translategemma"
TEMPERATURE = 0.1
TEST_SIZE = 5
LANG_CODES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "Haitian Creole": "ht",
    "Arabic": "ar",
    "Somali": "so",
    "Urdu": "ur",
    "Chinese": "zh",
    "Yiddish": "yi"
}

# NCI Dictionary paths (from match_with_nci.py)
CANCER_TERMS_FILE = "/home/baihesun/scrape-NCI-dictionaries/data/processed/cancer-terms.json"
GENETIC_TERMS_FILE = "/home/baihesun/scrape-NCI-dictionaries/data/processed/genetic-terms.json"

# TranslateGemma Configuration: Use Ollama (faster local inference) or HuggingFace
USE_OLLAMA_FOR_TRANSLATEGEMMA = True  # Set to False to use HuggingFace Transformers instead

MODEL_CONFIGS = {
    "openai": {"model": "gpt-5.2", "api_key_env": "OPENAI_API_KEY"},
    "claude": {"model": "claude-sonnet-4-5-20250929", "api_key_env": "ANTHROPIC_API_KEY"},
    "gemini": {"model": "gemini-2.5-flash-lite", "api_key_env": "GOOGLE_API_KEY"},
    "translategemma": {
        "ollama_model": "translategemma:12b",
        "hf_model": "google/translategemma-4b-it",
        "api_key_env": "HF_TOKEN"
    }
}


def load_nci_term_dict(*dict_files):
    """Load and merge NCI dictionary files into {en_term: es_term} mapping."""
    term_dict = {}
    for path in dict_files:
        with open(path, "r") as f:
            entries = json.load(f)
        for entry in entries:
            en = (entry.get("term_en") or "").strip().lower()
            es = (entry.get("term_es") or "").strip()
            if en and es:
                term_dict[en] = es
    print(f"Loaded {len(term_dict)} NCI terms from {len(dict_files)} dictionaries")
    return term_dict


def find_nci_terms_in_text(text, term_dict):
    """Return {en_term: es_term} for NCI terms found in the given English text."""
    text_lower = text.lower()
    found = {}
    for en_term, es_term in term_dict.items():
        pattern = r'(?<![\w\u00C0-\u024F])' + re.escape(en_term) + r'(?![\w\u00C0-\u024F])'
        if re.search(pattern, text_lower, re.IGNORECASE):
            found[en_term] = es_term
    return found


def get_client(provider, config):
    api_key = os.environ.get(config["api_key_env"])

    if provider != "translategemma" and not api_key:
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

    elif provider == "translategemma":
        if USE_OLLAMA_FOR_TRANSLATEGEMMA:
            import ollama
            print(f"TranslateGemma: Using Ollama with model {config['ollama_model']}")
            return ollama
        else:
            from transformers import pipeline
            import torch

            if torch.cuda.is_available():
                device, dtype = "cuda", torch.bfloat16
            elif torch.backends.mps.is_available():
                device, dtype = "mps", torch.float32
            else:
                device, dtype = "cpu", torch.float32

            print(f"TranslateGemma: Using HuggingFace Transformers with {device}, {dtype}")
            return pipeline("image-text-to-text", model=config["hf_model"], device=device, dtype=dtype, token=api_key)


def translate_field(client, text, language, provider, model, temperature, nci_terms=None):
    """Translate text, optionally injecting an NCI glossary into the prompt.

    nci_terms: dict of {en_term: es_term} to include as a terminology glossary.
    """
    lang_codes = {"Spanish": "es", "French": "fr", "English": "en", "German": "de", "Portuguese": "pt"}
    source_code = "en"
    target_code = lang_codes.get(language, "es")

    glossary_section = ""
    if nci_terms:
        glossary_lines = "\n".join(f"  {en} → {es}" for en, es in sorted(nci_terms.items()))
        glossary_section = (
            f"\n\nUse the following NCI-approved terminology for the specific medical terms listed below:\n"
            f"{glossary_lines}"
        )

    prompt = (
        f"Translate the following medical oncology text from English to {language}. "
        f"Use standard medical terminology in {language}-speaking medical practice. "
        f"Output only the translation, no explanations."
        f"{glossary_section}\n\n{text}"
    )

    if provider == "translategemma":
        if USE_OLLAMA_FOR_TRANSLATEGEMMA:
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response['message']['content']
        else:
            messages = [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "source_lang_code": source_code,
                    "target_lang_code": target_code,
                    "text": prompt
                }]
            }]
            output = client(text=messages, max_new_tokens=1024, generate_kwargs={"do_sample": False})
            return output[0]["generated_text"][-1]["content"]

    if provider == "openai":
        return client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model, temperature=temperature
        ).choices[0].message.content

    elif provider == "claude":
        return client.messages.create(
            model=model, max_tokens=1024, temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        ).content[0].text

    elif provider == "gemini":
        return client.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": 1024}
        ).text


def translate_entry(client, entry, languages, provider, model, temperature, term_dict):
    translated = entry.copy()
    if entry.get('en'):
        en_text = entry['en']
        # Standard translations for all languages (no NCI)
        for language in languages:
            translated[LANG_CODES.get(language)] = translate_field(
                client, en_text, language, provider, model, temperature
            )
        # NCI-augmented Spanish translation
        nci_terms = find_nci_terms_in_text(en_text, term_dict)
        translated["es_nci"] = translate_field(
            client, en_text, "Spanish", provider, model, temperature,
            nci_terms=nci_terms if nci_terms else None
        )
    return translated


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config = MODEL_CONFIGS[MODEL_PROVIDER]

    if MODEL_PROVIDER == "translategemma":
        model_name = config['ollama_model'] if USE_OLLAMA_FOR_TRANSLATEGEMMA else config['hf_model']
    else:
        model_name = config['model']

    print(f"Initializing {MODEL_PROVIDER} ({model_name})...")
    client = get_client(MODEL_PROVIDER, config)

    term_dict = load_nci_term_dict(CANCER_TERMS_FILE, GENETIC_TERMS_FILE)

    with open(INDICATIONS_PATH, 'r') as f:
        data = json.load(f)

    if TEST_SIZE:
        data = data[:TEST_SIZE]

    print(f"Translating {len(data)} entries to {LANGUAGES}...\n")

    start = time.time()
    translated = []

    for i, entry in enumerate(data):
        translated.append(translate_entry(client, entry, LANGUAGES, MODEL_PROVIDER, model_name, TEMPERATURE, term_dict))
        elapsed = time.time() - start
        remaining = (elapsed / (i + 1)) * (len(data) - i - 1)
        sys.stdout.write(f"\r{i+1}/{len(data)} ({100*(i+1)/len(data):.1f}%) | {elapsed:.0f}s elapsed | {remaining:.0f}s remaining")
        sys.stdout.flush()

    suffix = f"_test_{TEST_SIZE}" if TEST_SIZE else ""
    if MODEL_PROVIDER == "translategemma":
        method_suffix = "_ollama" if USE_OLLAMA_FOR_TRANSLATEGEMMA else "_hf"
        output = os.path.join(OUTPUT_DIR, f"descriptions_{MODEL_PROVIDER}{method_suffix}_nci{suffix}.json")
    else:
        output = os.path.join(OUTPUT_DIR, f"descriptions_{MODEL_PROVIDER}_nci{suffix}.json")

    with open(output, 'w') as f:
        json.dump(translated, f, indent=2, ensure_ascii=False)

    total = time.time() - start
    print(f"\n\nDone! {len(data)} entries in {total:.1f}s ({total/len(data):.2f}s/entry)")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
