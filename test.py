import json
import os
import time
import sys
from openai import OpenAI

# CONFIGURATION
INDICATIONS_PATH = "/Users/baihesun/moalmanac-db/referenced/indications.json"
OUTPUT_DIR = "/Users/baihesun/Desktop/python/LLM_translation_project/results/"
LANGUAGE = "Spanish"
MODEL_PROVIDER = "openai"  # Options: "openai", "claude", "gemini", "translategemma"
TEMPERATURE = 0.3
TEST_SIZE = 10



MODEL_CONFIGS = {
    "openai": {"model": "gpt-3.5-turbo", "api_key_env": "OPENAI_API_KEY"},
    "claude": {"model": "claude-sonnet-4-5-20250929", "api_key_env": "ANTHROPIC_API_KEY"},
    "gemini": {"model": "gemini-2.5-flash-lite", "api_key_env": "GOOGLE_API_KEY"},
    "translategemma": {"model": "google/translategemma-4b-it", "api_key_env": "HF_TOKEN"}
}


def get_client(provider, config):
    api_key = os.environ.get(config["api_key_env"])

    if provider != "translategemma" and not api_key:
        raise ValueError(f"Set {config['api_key_env']} environment variable")

    if provider == "openai":
        return OpenAI(api_key=api_key)

    elif provider == "claude":
        import anthropic
        return anthropic.Anthropic(api_key=api_key)

    elif provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(config["model"])

    elif provider == "translategemma":
        from transformers import pipeline
        import torch

        if torch.cuda.is_available():
            device, dtype = "cuda", torch.bfloat16
        elif torch.backends.mps.is_available():
            device, dtype = "mps", torch.float32
        else:
            device, dtype = "cpu", torch.float32

        print(f"TranslateGemma: {device}, {dtype}")
        return pipeline("image-text-to-text", model=config["model"], device=device, dtype=dtype, token=api_key)


def translate_field(client, text, language, provider, model, temperature):
    if provider == "translategemma":
        lang_codes = {"Spanish": "es", "French": "fr", "English": "en"}
        target = lang_codes.get(language, "es")
        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "source_lang_code": "en",
                "target_lang_code": target,
                "text": text
            }]
        }]
        output = client(text=messages, max_new_tokens=1024, generate_kwargs={"do_sample": False})
        return output[0]["generated_text"][-1]["content"]

    prompt = f"Translate the following medical text into {language}:\n\n{text}"

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


def translate_entry(client, entry, language, provider, model, temperature):
    translated = entry.copy()
    if entry.get('indication'):
        translated['indication'] = translate_field(client, entry['indication'], language, provider, model, temperature)
    if entry.get('description'):
        translated['description'] = translate_field(client, entry['description'], language, provider, model, temperature)
    return translated


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config = MODEL_CONFIGS[MODEL_PROVIDER]

    print(f"Initializing {MODEL_PROVIDER} ({config['model']})...")
    client = get_client(MODEL_PROVIDER, config)

    with open(INDICATIONS_PATH, 'r') as f:
        data = json.load(f)

    if TEST_SIZE:
        data = data[:TEST_SIZE]

    print(f"Translating {len(data)} entries to {LANGUAGE}...\n")

    start = time.time()
    translated = []

    for i, entry in enumerate(data):
        translated.append(translate_entry(client, entry, LANGUAGE, MODEL_PROVIDER, config['model'], TEMPERATURE))
        elapsed = time.time() - start
        remaining = (elapsed / (i + 1)) * (len(data) - i - 1)
        sys.stdout.write(f"\r{i+1}/{len(data)} ({100*(i+1)/len(data):.1f}%) | {elapsed:.0f}s elapsed | {remaining:.0f}s remaining")
        sys.stdout.flush()

    suffix = f"_test_{TEST_SIZE}" if TEST_SIZE else ""
    output = os.path.join(OUTPUT_DIR, f"indications_{LANGUAGE}_{MODEL_PROVIDER}{suffix}.json")

    with open(output, 'w') as f:
        json.dump(translated, f, indent=2, ensure_ascii=False)

    total = time.time() - start
    print(f"\n\nDone! {len(data)} entries in {total:.1f}s ({total/len(data):.2f}s/entry)")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()