# Backend

FastAPI wrapper for the existing Malayalam summarizer pipeline.

## Run

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The frontend expects the API at `http://127.0.0.1:8000`.
