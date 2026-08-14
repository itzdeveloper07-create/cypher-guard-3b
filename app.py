import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = FastAPI(title="CYPHER-GUARD-3B API")

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
PEFT_MODEL_ID = "./cypher_guard_real_final"

print("Loading CYPHER-GUARD-3B for production API...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, PEFT_MODEL_ID)
model.eval()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/check-safety")
async def check_safety(request: PromptRequest):
    formatted_input = f"User: {request.prompt}\nAssistant:"
    inputs = tokenizer(formatted_input, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=30, 
            temperature=0.1,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    if "Assistant:" in result:
        response_part = result.split("Assistant:")[-1].strip()
    else:
        response_part = result
        
    return {"query": request.prompt, "guard_decision": response_part}