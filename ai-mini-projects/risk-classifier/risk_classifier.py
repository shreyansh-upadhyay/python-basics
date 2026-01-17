from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)

labels = ["LOW RISK", "MEDIUM RISK", "HIGH RISK"]

text = """
Autoscaling changes were deployed without full load testing.
Rollback plan exists but has not been validated.
Customer impact is possible during peak traffic.
"""

# --- Tokenize input ---
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=512
)

# --- Run model ---
with torch.no_grad():
    outputs = model(**inputs)

# --- Convert logits to probabilities ---
scores = torch.softmax(outputs.logits, dim=1)[0]

# --- Get predicted risk ---
predicted_index = torch.argmax(scores).item()
risk_level = labels[predicted_index]
confidence = scores[predicted_index].item()

# --- Simple rule-based explanation ---
explanations = []

if "without" in text.lower():
    explanations.append("Change performed without full validation")
if "rollback" in text.lower():
    explanations.append("Rollback plan exists but may be untested")
if "customer impact" in text.lower():
    explanations.append("Potential customer impact mentioned")
if "peak" in text.lower():
    explanations.append("Risk during peak traffic window")

# Fallback explanation
if not explanations:
    explanations.append("General execution uncertainty detected")

# --- Output ---
print("\n=== INPUT TEXT ===\n")
print(text)

print("\n=== AI RISK ASSESSMENT ===\n")
print(f"Risk Level   : {risk_level}")
print(f"Confidence   : {confidence:.2%}")

print("\nReasoning:")
for reason in explanations:
    print(f"- {reason}")
