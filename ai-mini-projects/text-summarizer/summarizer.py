from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

# Load tokenizer (safe)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model using SafeTensors ONLY
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    use_safetensors=True
)

text = """
The deployment involved multiple services across regions.
Initial rollout caused latency issues in one zone,
which were mitigated by scaling and configuration fixes.
Overall system stability improved after the changes.
"""

# Tokenize
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=512
)

# Generate summary
summary_ids = model.generate(
    inputs["input_ids"],
    max_length=50,
    min_length=20,
    do_sample=False
)

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("Original text:\n", text)
print("\nSummary:\n", summary)
