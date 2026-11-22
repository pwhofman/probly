from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

for m in ["HuggingFaceTB/SmolLM2-135M-Instruct",
          "Qwen/Qwen2.5-0.5B-Instruct",
          "Qwen/Qwen2.5-1.5B-Instruct"]:
    print("Indiriliyor:", m)
    AutoTokenizer.from_pretrained(m)
    AutoModelForCausalLM.from_pretrained(m)

print("Indiriliyor: microsoft/deberta-base-mnli")
AutoTokenizer.from_pretrained("microsoft/deberta-base-mnli")
AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base-mnli")

print("BITTI - tum modeller cache'te")
