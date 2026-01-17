from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)

# Labels we define for TPM context
labels = ["LOW RISK", "MEDIUM RISK", "HIGH RISK"]

text = """
Autoscaling changes were deployed without full load testing.
Rollback plan exists but has not been validated.
Customer impact is possible during peak traffic.
"""

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=512
)

with torch.no_grad():
    outputs = model(**inputs)

scores = torch.softmax(outputs.logits, dim=1)
risk_level = labels[torch.argmax(scores)]

print("\n=== INPUT TEXT ===\n")
print(text)

print("\n=== RISK ASSESSMENT ===\n")
print(risk_level)
