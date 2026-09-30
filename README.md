# RAG-Based Customer Service Chatbot
**Gen AI Project**

## Overview
A Retrieval-Augmented Generation (RAG) chatbot powered by Google Gemini LLM and FAISS vector store for intelligent customer support automation.

## Architecture
```
User Query
    │
    ▼
Text Preprocessing (clean, normalize)
    │
    ▼
Sentence Embedding (all-MiniLM-L6-v2)
    │
    ▼
FAISS Vector Store (cosine similarity search)
    │
    ▼
Top-K Context Retrieval
    │
    ▼
Gemini LLM (prompt + context → answer)
    │
    ▼
Response to User
```

## Project Structure
```
rag_chatbot/
├── app.py              # Streamlit frontend
├── rag_pipeline.py     # Core RAG engine
├── evaluate.py         # Evaluation script
├── requirements.txt    # Dependencies
├── data/
│   └── sample_kb.csv   # Sample knowledge base
└── README.md
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the chatbot
```bash
streamlit run app.py
```

### 3. Run evaluation
```bash
python evaluate.py
```

## Usage
1. Open the app in your browser (http://localhost:8501)
2. (Optional) Enter your Google Gemini API key in the sidebar
3. Choose "Use Sample KB" for demo or upload your own CSV/TXT/JSON
4. Click **Initialize Chatbot**
5. Ask questions in the chat box

## Without Gemini API Key
The chatbot works in **Demo Mode** without an API key using rule-based retrieval. To enable AI-generated responses, get a free key from https://makersuite.google.com/app/apikey

## Knowledge Base Format
Your CSV should have these columns:
- `question` — customer query
- `answer` — official response
- `category` — issue type (optional)

## Evaluation Metrics
- **Response Accuracy**: keyword match against expected answers
- **Contextual Relevance**: cosine similarity between query and retrieved chunks
- **Latency**: average response time in seconds

## Tech Stack
| Component | Technology |
|---|---|
| LLM | Google Gemini 1.5 Flash |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | FAISS (Facebook AI Similarity Search) |
| Frontend | Streamlit |
| Language | Python 3.9+ |

## Skills Demonstrated
- RAG pipeline implementation
- Vector database creation and search (FAISS)
- Prompt engineering with Gemini
- NLP text preprocessing and chunking
- Streamlit chatbot interface
- Evaluation metrics design
