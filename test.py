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
    lang_codes = {"Spanish": "es", "French": "fr", "English": "en", "German": "de", "Portuguese": "pt"}
    source_code = "en"
    target_code = lang_codes.get(language, "es")

    # Professional translation prompt template for medical oncology database (used by all providers)
    prompt = prompt = f"""Translate the following medical oncology text from English to {language}. Use standard medical terminology in {language}-speaking medical practice. Output only the translation, no explanations.

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


def translate_entry(client, entry, languages, provider, model, temperature):
    translated = entry.copy()
    if entry.get('en'):
        for language in languages:
            translated[LANG_CODES.get(language)] = translate_field(client, entry['en'], language, provider, model, temperature)
    return translated


def main():
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

    if TEST_SIZE:
        data = data[:TEST_SIZE]

    print(f"Translating {len(data)} entries to {LANGUAGES}...\n")

    start = time.time()
    translated = []
    
    for i, entry in enumerate(data):
        translated.append(translate_entry(client, entry, LANGUAGES, MODEL_PROVIDER, model_name, TEMPERATURE))
        elapsed = time.time() - start
        remaining = (elapsed / (i + 1)) * (len(data) - i - 1)
        sys.stdout.write(f"\r{i+1}/{len(data)} ({100*(i+1)/len(data):.1f}%) | {elapsed:.0f}s elapsed | {remaining:.0f}s remaining")
        sys.stdout.flush()

    suffix = f"_test_{TEST_SIZE}" if TEST_SIZE else ""
    # Add method suffix for translategemma (ollama vs hf)
    if MODEL_PROVIDER == "translategemma":
        method_suffix = "_ollama" if USE_OLLAMA_FOR_TRANSLATEGEMMA else "_hf"
        output = os.path.join(OUTPUT_DIR, f"descriptions_{MODEL_PROVIDER}{method_suffix}{suffix}.json")
    else:
        output = os.path.join(OUTPUT_DIR, f"descriptions_{MODEL_PROVIDER}{suffix}.json")

    with open(output, 'w') as f:
        json.dump(translated, f, indent=2, ensure_ascii=False)

    total = time.time() - start
    print(f"\n\nDone! {len(data)} entries in {total:.1f}s ({total/len(data):.2f}s/entry)")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()