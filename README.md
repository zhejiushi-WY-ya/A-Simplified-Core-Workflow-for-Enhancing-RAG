## 🚀 Setup Instructions

### 1. Clone LightRAG

```bash
git clone https://github.com/HKUDS/LightRAG.git
```

### 2. Enter the project directory and install dependencies

```bash
cd LightRAG
uv tool install -e .
uv add langgraph
```

### 3. Configure the LLM environment
```bash
cp env.example .env
make env-base
```
Edit the .env file according to your setup (e.g., OpenAI API or local LLM endpoint).

### 4. Clone this project
```bash
git clone https://github.com/zhejiushi-WY-ya/A-Simplified-Core-Workflow-for-Enhancing-RAG.git
```

### 5. Enter the project directory and run the download script
```bash
cd A-Simplified-Core-Workflow-for-Enhancing-RAG
python download.py
```

### 6. Enter the project directory and run the download script
```bash
cd A-Simplified-Core-Workflow-for-Enhancing-RAG
python download.py
```

### 7. Run the simplified LightRAG pipeline
```bash
cd lightrag_core_simplified
python -m src/main_index
```

