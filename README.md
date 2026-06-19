# Document QnA Bot

A production-ready Streamlit app for asking questions about an uploaded PDF using Google Gemini, LangChain, and an in-memory vector store.

## Features

- PDF upload and text chunking
- Gemini embeddings for semantic search
- Gemini chat model for grounded answers
- Environment-based configuration
- Docker deployment support
- Safe temporary file handling for uploads

## Requirements

- Python 3.12+
- Google AI Studio API key

## Local Setup

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set `GOOGLE_API_KEY`.

Run the app:

```powershell
streamlit run app.py
```

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `GOOGLE_API_KEY` | Required | Google Gemini API key |
| `GOOGLE_CHAT_MODEL` | `gemini-2.5-flash` | Chat model used for answers |
| `GOOGLE_EMBEDDING_MODEL` | `gemini-embedding-2-preview` | Embedding model used for retrieval |
| `CHUNK_SIZE` | `1000` | Text chunk size |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_K` | `3` | Number of chunks retrieved per question |
| `MAX_UPLOAD_MB` | `25` | Maximum PDF upload size |

## Docker Deployment

Build the image:

```powershell
docker build -t document-qna-bot .
```

Run the container:

```powershell
docker run --rm -p 8501:8501 --env-file .env document-qna-bot
```

Open `http://localhost:8501`.

## Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create a new Streamlit app from the repo.
3. Set `GOOGLE_API_KEY` in the app secrets or environment settings.
4. Use `app.py` as the entrypoint.

## Production Notes

- Do not commit `.env` or API keys.
- Rotate any API key that was previously committed or shared.
- This app uses an in-memory vector store, so each user/session processes the uploaded PDF again.
- For larger documents, multi-user workloads, or persistent indexes, replace `InMemoryVectorStore` with a production vector database.
