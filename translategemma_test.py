from transformers import pipeline, AutoModelForImageTextToText, AutoProcessor
import torch
import os

HF_TOKEN = os.environ.get("HF_TOKEN")

# Detect best available device
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon GPU
    dtype = torch.float32  # MPS doesn't support bfloat16
else:
    device = "cpu"
    dtype = torch.float32

print(f"Using device: {device}, dtype: {dtype}")

pipe = pipeline(
    "image-text-to-text",
    model="google/translategemma-4b-it",
    device=device,
    dtype=dtype,
    token=HF_TOKEN,
)

text = r"""
Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, “and what is the use of a book,” thought Alice “without pictures or conversations?”

So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.
"""

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "source_lang_code": "en",
                "target_lang_code": "es",
                "text": text,
            }
        ],
    }
]

output = pipe(text=messages, max_new_tokens=200, generate_kwargs={"do_sample": False})

print(output[0]["generated_text"][-1]["content"])