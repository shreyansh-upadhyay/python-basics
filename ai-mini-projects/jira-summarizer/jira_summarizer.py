from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    use_safetensors=True
)

jira_text = """
Jira Ticket: SCP-2145

The Scale & Performance team worked on improving request latency
for the customer onboarding workflow.

Initial rollout revealed elevated p95 latency in one region.
The issue was traced to misconfigured autoscaling thresholds.
Fixes were applied and verified in staging.

Current status is stable, but follow-up monitoring is required.
"""

inputs = tokenizer(
    jira_text,
    return_tensors="pt",
    truncation=True,
    max_length=512
)

summary_ids = model.generate(
    inputs["input_ids"],
    max_length=80,
    min_length=40,
    do_sample=False
)

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("\n=== RAW JIRA INPUT ===\n")
print(jira_text)

print("\n=== EXECUTIVE SUMMARY ===\n")
print(summary)
