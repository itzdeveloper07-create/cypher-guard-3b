# cypher-guard-3b
Enterprise-grade AI safety guardrail model based on Qwen2.5-3B, achieving 99.40% accuracy on 1000 jailbreak prompts. Developed under Sky Matrix World.

# CYPHER-GUARD-3B 🛡️
> Enterprise-grade AI Safety Guardrail & Prompt Firewall Model

**CYPHER-GUARD-3B** is a high-performance fine-tuned AI model based on **Qwen2.5-3B-Instruct**, built to detect and neutralize malicious jailbreaks, prompt injections, and adversarial threats in real-time. 

Developed under **Sky Matrix World**.

---

## 📊 Benchmark & Performance
Tested rigorously on an extreme dataset consisting of mixed safe prompts and hardcore jailbreaks:
* **Total Prompts Tested:** 1,000 (532 Safe / 468 Unsafe)
* **Final Accuracy:** **99.40%**
* **False Positives:** Minimized using strict system prompt engineering and low-temperature inference.

---

## 🚀 Key Features
* **Zero False Positives:** Safely processes legitimate developer code and security questions without blocking normal workflows.
* **Low Latency:** Optimized for high-throughput production environments via FastAPI and PyTorch.
* **API Ready:** Fully compatible with standard LLM gateway routing and corporate security layers.

---

## 💻 Tech Stack & Ecosystem
* **Base Model:** Qwen/Qwen2.5-3B-Instruct
* **Fine-Tuning:** QLoRA (Peft)
* **Inference & Backend:** Python, PyTorch, Transformers, FastAPI
* **Hosting & Testing:** RunPod, Hugging Face Hub

---

## 🏢 Developed By
**Sky Matrix World**  
*Lead Developer:* Shree Sanjay Sonawane  
🌐 Official Domain: [skymatrixworld.co.in](https://skymatrixworld.co.in/)
