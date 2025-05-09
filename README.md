# 🧠 Agent Designer Challenge (Langchain + Ollama)

Welcome to the **Agent Designer Challenge**! Your goal is to design an intelligent agent using a free LLM (via [Ollama](https://ollama.com)) and [Langchain](https://www.langchain.com/).

You’ll define the agent’s **prompt**, build **custom tools**, and connect everything using Langchain's agent framework.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-org/agent-designer-challenge.git
cd agent-designer-challenge
```

### 2. Set up Python environment (recommended: Python 3.10+)

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install and Run Ollama (Free LLM Runtime)
Ollama runs language models locally on your machine. Download and install it from:

👉 https://ollama.com/download

🔹 Pull a model (e.g., Mistral)

```bash
ollama pull mistral
```
Supported models include:
* mistral
* llama3
* gemma
(others listed at ollama.com/library)

### 4. Run the agent:
```bash
python main.py
```
And you will see something like:
```bash
User: Hello! What can you do?
Agent: [Your agent’s response]
```
Type exit or quit to end the session.

---

## Project Structure:
```bash
agent_challenge/
├── main.py                  # Entry point: runs the agent loop
├── agent_config.py          # Configure prompt + tools here
├── tools.py                 # Example custom tool  
├── requirements.txt
└── README.md
```