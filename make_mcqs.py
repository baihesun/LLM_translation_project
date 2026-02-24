import json
import os
import time
import sys
import argparse


# CONFIGURATION
INPUT_PATH = "/home/baihesun/LLM_translation_project/results/descriptions_openai_test_5.json"
OUTPUT_DIR = "/home/baihesun/LLM_translation_project/results/"
MODEL_PROVIDER = "openai"
TEMPERATURE = 0.3
N_QUESTIONS = 3  # Number of MCQs per entry

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


def generate_mcqs(client, text, n, provider, model, temperature):
    prompt = f"""You are a medical knowledge assessment expert. Based on the following medical text, generate exactly {n} multiple-choice questions that test comprehension of key facts in the passage.

Medical text:
{text}

Output a JSON object with a single key "questions" whose value is an array of {n} objects. Each object must have:
- "question": the question string
- "A": option A
- "B": option B
- "C": option C
- "D": option D
- "answer": the correct letter (A, B, C, or D)

Rules:
- Each question must have exactly one correct answer
- Distractors should be plausible but clearly incorrect based on the text
- All answers must be grounded in the provided text
- Output only the JSON object, no extra text or markdown

Example format:
{{
  "questions": [
    {{
      "question": "What drug was approved?",
      "A": "letrozole",
      "B": "abemaciclib",
      "C": "tamoxifen",
      "D": "anastrozole",
      "answer": "B"
    }}
  ]
}}"""

    if provider == "openai":
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            # Try the expected key first, then fall back to any list value
            if "questions" in parsed and isinstance(parsed["questions"], list):
                return parsed["questions"]
            for v in parsed.values():
                if isinstance(v, list):
                    return v
        raise ValueError(f"Unexpected JSON structure: {raw}")


def main():
    parser = argparse.ArgumentParser(description="Generate multiple-choice questions from English medical entries.")
    parser.add_argument("--input", default=INPUT_PATH, help="Path to input JSON file")
    parser.add_argument("--output", default=None, help="Path to output JSON file (default: auto-named in results/)")
    parser.add_argument("--n", type=int, default=N_QUESTIONS, help="Number of MCQs per entry")
    parser.add_argument("--provider", default=MODEL_PROVIDER, choices=list(MODEL_CONFIGS.keys()), help="Model provider")
    parser.add_argument("--test-size", type=int, default=None, help="Only process first N entries (for testing)")
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.provider]
    model_name = config["model"]

    output_path = args.output
    if output_path is None:
        input_stem = os.path.splitext(os.path.basename(args.input))[0]
        output_path = os.path.join(OUTPUT_DIR, f"mcqs_{input_stem}_{args.provider}_n{args.n}.json")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Provider: {args.provider} ({model_name})")
    print(f"Input:    {args.input}")
    print(f"Output:   {output_path}")
    print(f"MCQs per entry: {args.n}\n")

    client = get_client(args.provider, config)

    with open(args.input, "r") as f:
        data = json.load(f)

    if args.test_size:
        data = data[:args.test_size]

    results = []
    start = time.time()

    for i, entry in enumerate(data):
        en_text = entry.get("en", "")
        if not en_text:
            results.append({"id": entry.get("id"), "en": en_text, "mcqs": []})
            continue

        try:
            mcqs = generate_mcqs(client, en_text, args.n, args.provider, model_name, TEMPERATURE)
        except Exception as e:
            print(f"\nError on entry {entry.get('id')}: {e}")
            mcqs = []

        results.append({
            "id": entry.get("id"),
            "en": en_text,
            "mcqs": mcqs,
        })

        elapsed = time.time() - start
        remaining = (elapsed / (i + 1)) * (len(data) - i - 1)
        sys.stdout.write(
            f"\r{i+1}/{len(data)} ({100*(i+1)/len(data):.1f}%) | "
            f"{elapsed:.0f}s elapsed | {remaining:.0f}s remaining"
        )
        sys.stdout.flush()

    print()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} entries to {output_path}")


if __name__ == "__main__":
    main()
