import sys

if len(sys.argv) < 2:
    print("Usage: python jira_risk_analyzer.py <input_file.txt>")
    sys.exit(1)

with open(sys.argv[1], "r") as f:
    jira_text = f.read()


from transformers import ( # type: ignore
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification
)
import torch # type: ignore

# ---------- MODELS ----------
SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"
RISK_MODEL = "distilbert-base-uncased"

# ---------- LOAD SUMMARIZER ----------
sum_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
sum_model = AutoModelForSeq2SeqLM.from_pretrained(
    SUMMARIZER_MODEL,
    use_safetensors=True
)

# ---------- LOAD RISK CLASSIFIER ----------
risk_tokenizer = AutoTokenizer.from_pretrained(RISK_MODEL)
risk_model = AutoModelForSequenceClassification.from_pretrained(
    RISK_MODEL,
    num_labels=3
)

risk_labels = ["LOW RISK", "MEDIUM RISK", "HIGH RISK"]

# ---------- INPUT ----------
jira_text = """
Autoscaling changes were deployed without full load testing.
Rollback plan exists but has not been validated.
Customer impact is possible during peak traffic.
"""

# ---------- SUMMARY ----------
sum_inputs = sum_tokenizer(
    jira_text,
    return_tensors="pt",
    truncation=True,
    max_length=512
)

summary_ids = sum_model.generate(
    sum_inputs["input_ids"],
    max_length=80,
    min_length=40,
    do_sample=False
)

summary = sum_tokenizer.decode(
    summary_ids[0],
    skip_special_tokens=True
)

# ---------- RISK ----------
risk_inputs = risk_tokenizer(
    jira_text,
    return_tensors="pt",
    truncation=True,
    max_length=512
)

with torch.no_grad():
    outputs = risk_model(**risk_inputs)

scores = torch.softmax(outputs.logits, dim=1)[0]
predicted_index = torch.argmax(scores).item()
risk_level = risk_labels[predicted_index]
confidence = scores[predicted_index].item()

# ---------- EXPLANATION ----------
reasons = []
text_lower = jira_text.lower()

if "without" in text_lower:
    reasons.append("Change performed without full validation")
if "rollback" in text_lower:
    reasons.append("Rollback plan exists but may be untested")
if "customer impact" in text_lower:
    reasons.append("Potential customer impact mentioned")
if "peak" in text_lower:
    reasons.append("Risk during peak traffic window")

if not reasons:
    reasons.append("General execution uncertainty detected")

# ---------- OUTPUT ----------
print("\n=== EXECUTIVE SUMMARY ===\n")
print(summary)

print("\n=== RISK ASSESSMENT ===\n")
print(f"Risk Level : {risk_level}")
print(f"Confidence : {confidence:.2%}")

print("\nReasoning:")
for r in reasons:
    print(f"- {r}")
