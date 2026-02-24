import json
import os
import pandas as pd
import subprocess
import sys
from comet import load_from_checkpoint
from comet import download_model

JSON_PATH = "/home/baihesun/LLM_translation_project/results/descriptions_openai_test_5.json"
with open(JSON_PATH, "r") as f:
    translations = json.load(f)

# for language in file
lang_codes = [key for key in translations[0].keys() if key not in ("id", "en")]
index = [entry.get("id") for entry in translations] + ["all"]

# ── COMET ─────────────────────────────────────────────────────────────────

MODEL = "Unbabel/wmt22-cometkiwi-da"

# Download and load the model
model_path = download_model(MODEL)
model = load_from_checkpoint(model_path)

comet_results = pd.DataFrame(index=[entry.get("id") for entry in translations] + ["all"], columns=lang_codes)

for lang in lang_codes:
    # build data
    data = [
        {"src": entry["en"], "mt": entry[lang]} for entry in translations if entry.get("en") and entry.get(lang)
    ]
    data_ids = [entry["id"] for entry in translations if entry.get("en") and entry.get(lang)]

    # Get scores
    model_output = model.predict(data, batch_size=8, gpus=0)  # set gpus=1 if available.
    comet_results.loc[data_ids, lang] = model_output.scores
    comet_results.loc["all", lang] = model_output.system_score

print("\nCOMET Scores")
print(comet_results)


# ── MetricX-23 ─────────────────────────────────────────────────────────────────
metricx_results = pd.DataFrame(index=index, columns=lang_codes)

for lang in lang_codes:
    valid = [e for e in translations if e.get("en") and e.get(lang)]
    valid_ids = [e["id"] for e in valid]

    mx_input = os.path.abspath("mx_input.jsonl")
    mx_output = os.path.abspath("mx_output.jsonl")

    # Write input JSONL
    with open(mx_input, "w") as f:
        for e in valid:
            f.write(json.dumps({"source": e["en"], "hypothesis": e[lang], "reference": ""}) + "\n")

    # Run MetricX-23
    env = os.environ.copy()
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    env["DATASETS_DISABLE_CACHING"] = "1"
    subprocess.run([
        sys.executable, "-m", "metricx24.predict",
        "--tokenizer", "google/mt5-large",
        "--model_name_or_path", "google/metricx-24-hybrid-large-v2p6",
        "--max_input_length", "1536",
        "--batch_size", "1",
        "--input_file", mx_input,
        "--output_file", mx_output,
        "--qe",
    ], check=True, cwd="/home/baihesun/metricx", env=env)

    # Read output
    with open(mx_output) as f:
        scores = [json.loads(line)["prediction"] for line in f]

    metricx_results.loc[valid_ids, lang] = scores
    metricx_results.loc["all", lang] = sum(scores) / len(scores)

    os.remove(mx_input)
    os.remove(mx_output)

print("\nMetricX-23 Scores (lower = better)")
print(metricx_results)

    
