import json
import os
import time
import sys


"""
how to push to github bc I always forget:
cd LLM_translation_project
git add .
git commit -m "commit added ollama support for translategemma"
git push origin main
"""

# CONFIGURATION
INDICATIONS_PATH = "/Users/baihesun/moalmanac-db/referenced/indications.json"
OUTPUT_DIR = "/Users/baihesun/Desktop/python/LLM_translation_project/results_fields/"
LANGUAGE = "Spanish"
MODEL_PROVIDER = "openai"  # Options: "openai", "claude", "gemini", "translategemma"
TEMPERATURE = 0.2
TEST_SIZE = 5
LANG_CODES = {"Spanish": "es", "French": "fr", "English": "en"}

# TranslateGemma Configuration: Use Ollama (faster local inference) or HuggingFace
USE_OLLAMA_FOR_TRANSLATEGEMMA = True  # Set to False to use HuggingFace Transformers instead

MODEL_CONFIGS = {
    "openai": {"model": "gpt-3.5-turbo", "api_key_env": "OPENAI_API_KEY"},
    "claude": {"model": "claude-sonnet-4-5-20250929", "api_key_env": "ANTHROPIC_API_KEY"},
    "gemini": {"model": "gemini-2.5-flash-lite", "api_key_env": "GOOGLE_API_KEY"},
    "translategemma": {
        "ollama_model": "translategemma:12b",
        "hf_model": "google/translategemma-4b-it",
        "api_key_env": "HF_TOKEN"
    }
}

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
            return ollama  # Return the ollama module itself
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


def translate_field(client, text, language, provider, model, temperature):
    # Language codes for translategemma
    source_code = "en"
    target_code = LANG_CODES.get(language, "es")

    # Professional translation prompt template for medical oncology database (used by all providers)
    prompt = f"""You are a professional English ({source_code}) to {language} ({target_code}) translator specializing in medical oncology terminology. Your goal is to accurately convey the meaning and nuances of the original English text while adhering to {language} grammar, vocabulary, and cultural sensitivities.

Given the medical oncology context, ensure the following:

- **Prioritize Complete and Accurate Terminology:** When translating any medical or oncology 
term, always strive for the most complete and technically accurate Spanish equivalent. This 
means using the full, established term, rather than a shortened or informal version.  Consider 
the context carefully to determine the best term to use.

- **Maintain Clinical Precision:** Preserve the clinical meaning and technical accuracy of the 
original English text in the translation.

- **Adhere to Standard Medical Terminology:**  Use terminology consistent with standard 
medical practices in the Spanish-speaking world.

- **Avoid Slang and Informal Expressions:**  Do not use any slang, colloquialisms, or informal 
language.

Produce only the {language} translation, without any additional explanations or commentary. Please translate the following English text into {language}:

{text}"""

    if provider == "translategemma":
        if USE_OLLAMA_FOR_TRANSLATEGEMMA:
            # Use Ollama for translation
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response['message']['content']
        else:
            # Use HuggingFace Transformers pipeline
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


def translate_fields_to_files(client, data, fields, original_language, target_languages, provider, model, temperature, output_dir=OUTPUT_DIR, test_size=None):
    """
    translate multiple fields from data to multiple languages and save to separate JSON files.
    creates one JSON file per field with format:
        {field}_{MODEL_PROVIDER}.json containing entries:
        [{"id": "...", "en": "...", "es": "...", "fr": "..."}, ...]
    """
    os.makedirs(output_dir, exist_ok=True)

    # Limit data to test_size if specified
    if test_size:
        data = data[:test_size]
        print(f"testing: with {test_size} entries\n")

    original_language_code = LANG_CODES.get(original_language)

    # Process each field separately
    for field in fields:
        print(f"\n{'='*60}")
        print(f"Processing field: {field}")
        print(f"{'='*60}")

        field_results = []

        # process each entry requested 
        for i, entry in enumerate(data):

            # create result entry with id and original text
            result_entry = {
                "id": entry.get("id"), 
                f"{original_language_code}": entry[field]
            }

            # translating 
            for language in target_languages:
                language_code = LANG_CODES.get(language)
                print(f"  Entry {i+1}/{len(data)} - Translating to {language}...", end="")
                translation = translate_field(client, entry[field], language, provider, model, temperature)
                result_entry[language_code] = translation

            field_results.append(result_entry)

        # save
        output_path = os.path.join(output_dir, f"{field}_{MODEL_PROVIDER}.json")
        with open(output_path, 'w') as f:
            json.dump(field_results, f, indent=2, ensure_ascii=False)

        print(f"\nSaved {len(field_results)} entries to: {output_path}")

    print(f"\n{'='*60}")
    print("All done!")
    print(f"{'='*60}")


def main_multi_field():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config = MODEL_CONFIGS[MODEL_PROVIDER]

    # Get the model name based on provider
    if MODEL_PROVIDER == "translategemma":
        model_name = config['ollama_model'] if USE_OLLAMA_FOR_TRANSLATEGEMMA else config['hf_model']
        print(f"Initializing {MODEL_PROVIDER} ({model_name})...")
    else:
        model_name = config['model']
        print(f"Initializing {MODEL_PROVIDER} ({model_name})...")

    client = get_client(MODEL_PROVIDER, config)

    with open(INDICATIONS_PATH, 'r') as f:
        data = json.load(f)

    # Define which fields to translate and which languages
    fields_to_translate = ["indication", "description"]
    original_language = "English"
    target_languages = ["Spanish", "French"]  # Can add more languages

    # Show what will be processed
    num_entries = TEST_SIZE if TEST_SIZE else len(data)
    print(f"\nTranslating {num_entries} entries")
    print(f"Fields: {fields_to_translate}")
    print(f"From: {original_language}")
    print(f"To: {target_languages}")

    start = time.time()

    # Call the multi-field translation function
    translate_fields_to_files(
        client=client,
        data=data,
        fields=fields_to_translate,
        original_language=original_language,
        target_languages=target_languages,
        provider=MODEL_PROVIDER,
        model=model_name,
        temperature=TEMPERATURE,
        output_dir=OUTPUT_DIR,
        test_size=TEST_SIZE
    )

    total = time.time() - start
    print(f"\nTotal time: {total:.1f}s")


if __name__ == "__main__":
    main_multi_field()