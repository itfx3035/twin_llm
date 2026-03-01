# Twins: Twin-Agent Discussion System

A sophisticated dual-agent LLM discussion framework that leverages **Ollama** to run local language models. Two AI agents (Agent-Alpha and Agent-Beta) engage in multi-round discussions about questions to reach consensus answers.

## 🎯 Overview

This project implements an intelligent discussion system where:
- **Two autonomous agents** analyze questions from different perspectives
- **Multi-round discussions** allow agents to respond to each other's arguments
- **Consensus building** synthesizes both perspectives into a final answer
- **Offline-first approach** using Ollama for complete privacy and control
- **Configurable parameters** for model selection, temperature, and discussion depth

Perfect for scenarios requiring thoughtful multi-perspective analysis, debate simulation, or exploring complex topics from different angles.

## 🚀 Quick Start

### Prerequisites

1. **Ollama** - Download and install from [ollama.ai](https://ollama.ai)
2. **Python 3.8+**
3. **pip** package manager

### Installation

1. **Install Ollama**:
   - Download from [ollama.ai](https://ollama.ai)
   - Follow platform-specific installation instructions
   - Start the Ollama server

2. **Pull a model** (required):
   ```bash
   ollama pull mistral
   ```
   Other recommended models:
   ```bash
   ollama pull neural-chat
   ollama pull dolphin-mixtral
   ollama pull hf.co/unsloth/Ministral-3-8B-Instruct-2512-GGUF:UD-Q5_K_XL
   ```

3. **Install Python dependencies**:
   ```bash
   pip install langchain-ollama
   ```

### Basic Usage

```python
from twin_lib import ask_question

answer = ask_question(
    question="What are the benefits of renewable energy?",
    model_name="mistral",
    ollama_url="http://localhost:11434"
)

print(answer)
```

## 📋 Configuration Guide

### Ollama Server Setup

#### Local Machine
By default, Ollama runs on `http://localhost:11434`:
```python
answer = ask_question(
    question="Your question here",
    model_name="mistral"
)
```

#### Remote Machine
Connect to Ollama running on another machine:
```python
answer = ask_question(
    question="Your question here",
    model_name="mistral",
    ollama_url="http://{server_ip _address}:11434"
)
```

#### Docker/Network Setup
If running Ollama in Docker or on a network:
```bash
# Start Ollama with network access
ollama serve 0.0.0.0:11434
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | str | Required | The question for agents to discuss |
| `model_name` | str | Required | Ollama model name (must be pulled first) |
| `ollama_url` | str | `http://localhost:11434` | Ollama server URL |
| `temperature` | float | `0.7` | Model creativity (0.0-1.0, higher=more creative) |
| `max_rounds` | int | `5` | Maximum discussion rounds before conclusion |

## 🔧 Advanced Usage

### Using `TwinAgentDiscussion` Directly

```python
from twin_lib import TwinAgentDiscussion

discussion = TwinAgentDiscussion(
    model_name="neural-chat",
    ollama_url="http://192.168.0.1:11434",
    temperature=0.5,  # More deterministic
    max_rounds=3      # Fewer discussion rounds
)

answer = discussion.get_answer("What makes a good software architecture?")
discussion_log = discussion.get_discussion_log()

for message in discussion_log:
    print(message)
```

### Accessing Agent Objects Directly

```python
from twin_lib import Agent

agent = Agent(
    name="ExpertAgent",
    model_name="mistral",
    ollama_url="http://localhost:11434",
    temperature=0.8
)

response = agent.get_initial_response("Explain quantum computing")
```

## 📚 Examples

### Example 1: Mathematical Reasoning
```python
from twin_lib import ask_question

q = 'calculate: 5 + 7 + 3 - 2 / 34 + 1 * 4'

answer = ask_question(
    question=q,
    model_name="mistral",
    ollama_url="http://localhost:11434"
)

print(answer)
```

### Example 2: Complex Topic Discussion
```python
question = """
What are the pros and cons of artificial intelligence in healthcare?
Consider both benefits and risks.
"""

answer = ask_question(
    question=question,
    model_name="neural-chat",
    ollama_url="http://localhost:11434",
    temperature=0.7,
    max_rounds=5
)
```

### Example 3: Multiple Languages (Model Dependent)
```python
answer = ask_question(
    question="¿Cuáles son los beneficios del aprendizaje automático?",
    model_name="mistral",
    ollama_url="http://192.168.1.100:11434"
)
```

## 🎛️ Model Selection

### Recommended Models for This Task

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| `mistral` | 7B | Fast | Good | General discussions |
| `neural-chat` | 7B | Fast | Very Good | Conversational tasks |
| `dolphin-mixtral` | 8x7B | Moderate | Excellent | Complex reasoning |
| `llama2` | 7B | Fast | Good | Balanced performance |

Pull models with:
```bash
ollama pull mistral
ollama pull neural-chat
ollama pull dolphin-mixtral
```

## 🔍 How It Works

1. **Initialization**: Two agents are created with the same model and configuration
2. **Round 0**: Both agents provide initial responses to the question
3. **Discussion Rounds**: 
   - Agent-Alpha responds to Agent-Beta's previous input
   - Agent-Beta responds to Agent-Alpha's response
   - Agents check if they agree
4. **Convergence**: Process continues until agreement or max_rounds reached
5. **Consensus**: A final answer synthesizes both perspectives

## 📊 Output

The system provides:
- **Real-time round-by-round discussion output**
- **Agreement status** after each round
- **Final consensus answer** synthesizing both perspectives
- **Discussion log** with complete message history

```
============================================================
Question: Your question here
============================================================

Round 0: Initial responses
Agent-Alpha: Response text...
Agent-Beta: Response text...

Round 1: Discussing
Agent-Alpha: Response text...
Agent-Beta: Response text...
Agent-Alpha agrees: True, Agent-Beta agrees: True

✓ Agents reached agreement after 1 discussion round(s)!

============================================================
Final Consensus Answer:
============================================================
Synthesized answer here...
============================================================
```

## 🤝 Integration

Use in your projects:
```python
from twin_lib import ask_question
```
