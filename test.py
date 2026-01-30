import json
import os
import time
import sys
from openai import OpenAI

"""
to-dos: 
- understand the hyperparam fields and test different settings
- expand to more fields if needed
- allow flexibility for models 
"""

# configuration
INDICATIONS_PATH = "/Users/baihesun/moalmanac-db/referenced/indications.json"
OUTPUT_DIR = "/Users/baihesun/Desktop/python/LLM_translation_project/results/"
LANGUAGE = "Spanish"
OUTPUT_NAME = f"indications_{LANGUAGE}.json"
MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.3
TEST_SIZE = 10  # set to None to translate all entries


def translate_field(client, text, language, model, temperature):
    response = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": f"Translate the following medical text into {language}:\n\n{text}"
        }],
        model=model,
        temperature=temperature
    )
    return response.choices[0].message.content


def translate_entry(client, entry, language, model, temperature):
    """translate indication and description fields in an entry.--> edit later to include more fields? """
    translated_entry = entry.copy()

    if 'indication' in entry and entry['indication']:
        translated_entry['indication'] = translate_field(
            client, entry['indication'], language, model, temperature
        )
    if 'description' in entry and entry['description']:
        translated_entry['description'] = translate_field(
            client, entry['description'], language, model, temperature
        )
    return translated_entry


def print_progress(idx, total, start_time):
    """print progress bar with time estimates."""
    elapsed = time.time() - start_time
    avg_time = elapsed / (idx + 1)
    remaining = avg_time * (total - idx - 1)
    percent = ((idx + 1) / total) * 100

    sys.stdout.write(
        f"\rProgress: {idx + 1}/{total} ({percent:.1f}%) | "
        f"Elapsed: {elapsed:.0f}s | Remaining: {remaining:.0f}s"
    )
    sys.stdout.flush()


def main():
    # setup output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # initialize openai client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # load data
    print("Loading indications data...")
    with open(INDICATIONS_PATH, 'r') as f:
        indications_data = json.load(f)

    # use subset for testing if specified
    if TEST_SIZE:
        indications_data = indications_data[:TEST_SIZE]
        OUTPUT_NAME = f"indications_{LANGUAGE}_test_{TEST_SIZE}.json"

    print(f"Found {len(indications_data)} entries to translate\n")

    # translate all entries
    start_time = time.time()
    translated_data = []

    for idx, entry in enumerate(indications_data):
        translated_entry = translate_entry(client, entry, LANGUAGE, MODEL, TEMPERATURE)
        translated_data.append(translated_entry)
        print_progress(idx, len(indications_data), start_time)

    # save results
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    print(f"\n\nSaving translated data to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(translated_data, f, indent=2, ensure_ascii=False)

    # print summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("Translation complete!")
    print(f"Total entries: {len(indications_data)}")
    print(f"Total runtime: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Average time per entry: {total_time/len(indications_data):.2f}s")
    print(f"Output saved to: {output_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()


