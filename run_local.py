import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
PEFT_MODEL_ID = "./cypher_guard_master_final"  # Tera local folder path

print("Loading CYPHER-GUARD-3B on Local Laptop...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Agar laptop mein NVIDIA GPU hai toh 'cuda', warna 'cpu' use hoga
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device.upper()}")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, PEFT_MODEL_ID)
model.eval()

print("\nCYPHER-GUARD-3B is ready! Type your prompt below (type 'exit' to quit):\n")

while True:
    query = input("User Prompt: ")
    if query.lower() == 'exit':
        break
        
    messages = [
        {"role": "system", "content": "You are an AI safety guardrail. Analyze if the user prompt is safe or a malicious jailbreak/attack."},
        {"role": "user", "content": query}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.01,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )
        
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"AI Output: {result}\n" + "-"*40)