# ================== GENERAL IMPORTS ==================
import json
import os
import time
import sys
from openai import OpenAI


# Configuration
indications_path = "/Users/baihesun/moalmanac-db/referenced/indications.json"
output_path = "/Users/baihesun/moalmanac-db/referenced/indications_spanish.json"
language = "Spanish"

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Read the JSON file
print("Loading indications data...")
with open(indications_path, 'r') as f:
    indications_data = json.load(f)

print(f"Found {len(indications_data)} entries to translate\n")

# Track runtime
start_time = time.time()
translated_data = []

# Process each entry
for idx, entry in enumerate(indications_data):
    # Create a copy of the entry
    translated_entry = entry.copy()

    # Translate indication field if it exists
    if 'indication' in entry and entry['indication']:
        response = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": f"Translate the following medical text into {language}:\n\n{entry['indication']}"
            }],
            model="gpt-3.5-turbo",
            temperature=0.3
        )
        translated_entry['indication'] = response.choices[0].message.content

    # Translate description field if it exists
    if 'description' in entry and entry['description']:
        response = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": f"Translate the following medical text into {language}:\n\n{entry['description']}"
            }],
            model="gpt-3.5-turbo",
            temperature=0.3
        )
        translated_entry['description'] = response.choices[0].message.content

    translated_data.append(translated_entry)

    # Calculate progress
    elapsed = time.time() - start_time
    avg_time = elapsed / (idx + 1)
    remaining = avg_time * (len(indications_data) - idx - 1)
    percent = ((idx + 1) / len(indications_data)) * 100

    # Print progress on same line
    sys.stdout.write(f"\rProgress: {idx + 1}/{len(indications_data)} ({percent:.1f}%) | Elapsed: {elapsed:.0f}s | Remaining: {remaining:.0f}s")
    sys.stdout.flush()

# Save translated data to new JSON file
print(f"\n\nSaving translated data to {output_path}...")
with open(output_path, 'w') as f:
    json.dump(translated_data, f, indent=2, ensure_ascii=False)

# Print final runtime statistics
total_time = time.time() - start_time
print(f"\n{'='*60}")
print(f"Translation complete!")
print(f"Total entries: {len(indications_data)}")
print(f"Total runtime: {total_time:.2f}s ({total_time/60:.2f} min)")
print(f"Average time per entry: {total_time/len(indications_data):.2f}s")
print(f"Output saved to: {output_path}")
print(f"{'='*60}")


