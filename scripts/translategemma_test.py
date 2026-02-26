import os

# CONFIGURATION
USE_OLLAMA = True  # Set to True to use Ollama for faster local inference

if USE_OLLAMA:
    import ollama
    MODEL_NAME = "translategemma:4b"  # Change to your preferred Ollama model
    print(f"Using Ollama with model: {MODEL_NAME}")
else:
    from transformers import pipeline
    import torch

    HF_TOKEN = os.environ.get("HF_TOKEN")

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"Using Transformers with device: {device}, dtype: {dtype}")

    pipe = pipeline(
        "image-text-to-text",
        model="google/translategemma-4b-it",
        device=device,
        dtype=dtype,
        token=HF_TOKEN,
    )

text = r"""
Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "and what is the use of a book," thought Alice "without pictures or conversations?"

So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.
"""

SOURCE_LANG = "English"
SOURCE_CODE = "en"
TARGET_LANG = "Spanish"
TARGET_CODE = "es"

# Professional translation prompt template for medical oncology database
TRANSLATION_PROMPT_TEMPLATE = """You are a professional {SOURCE_LANG} ({SOURCE_CODE}) to {TARGET_LANG} ({TARGET_CODE}) translator specializing in medical oncology terminology. Your goal is to accurately convey the meaning and nuances of the original {SOURCE_LANG} text while adhering to {TARGET_LANG} grammar, vocabulary, and cultural sensitivities.

Given the medical oncology context, please ensure:
- Accurate translation of medical and oncology-specific terminology
- Preservation of clinical precision and technical accuracy
- Consistency with standard medical terminology in the target language
- Maintenance of any critical medical information without loss of meaning

Produce only the {TARGET_LANG} translation, without any additional explanations or commentary. Please translate the following {SOURCE_LANG} text into {TARGET_LANG}:

{TEXT}"""

if USE_OLLAMA:
    prompt = TRANSLATION_PROMPT_TEMPLATE.format(
        SOURCE_LANG=SOURCE_LANG,
        SOURCE_CODE=SOURCE_CODE,
        TARGET_LANG=TARGET_LANG,
        TARGET_CODE=TARGET_CODE,
        TEXT=text
    )
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    print(response['message']['content'])
else:
    # For Transformers pipeline, we'll use the prompt as a system instruction
    prompt = TRANSLATION_PROMPT_TEMPLATE.format(
        SOURCE_LANG=SOURCE_LANG,
        SOURCE_CODE=SOURCE_CODE,
        TARGET_LANG=TARGET_LANG,
        TARGET_CODE=TARGET_CODE,
        TEXT=text
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": SOURCE_CODE,
                    "target_lang_code": TARGET_CODE,
                    "text": prompt,
                }
            ],
        }
    ]
    output = pipe(text=messages, max_new_tokens=200, generate_kwargs={"do_sample": False})
    print(output[0]["generated_text"][-1]["content"])