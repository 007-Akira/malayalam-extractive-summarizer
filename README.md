# Malayalam Extractive Summarizer

A Malayalam extractive text-summarization application with a FastAPI backend
and a React/Vite frontend. It ranks sentences from the source article using
multilingual sentence embeddings, learned classifiers, Malayalam-aware
features, and redundancy-aware sentence selection.

The generated summary contains sentences selected directly from the source
article; it does not generate or rewrite facts.

## Features

- Malayalam sentence segmentation
- Extractive sentence ranking
- Multiple trained checkpoint options
- Dynamic summary length and configurable sentence count
- Redundancy-aware D-MMR selection
- Chronological ordering of selected sentences
- FastAPI REST API
- React/Vite web interface

## Available models

The trained application checkpoints are stored in `backend/models/` and are
included in the repository:

| Application option | Checkpoint |
|---|---|
| Chotta Bheem | `chotta_bheem.pt` |
| Chotta Bheem V2 | `chotta_bheem_finetuned.pt` |
| Sentence Classifier | `malayalam_sentence_classifier.pt` |
| Hybrid Classifier | `malayalam_hybrid_classifier.pt` |
| IndicBERT | `indicbert_classifier.pt` |

Chotta Bheem is the default model used by the application.

The classifiers use either
[`sentence-transformers/LaBSE`](https://huggingface.co/sentence-transformers/LaBSE)
or
[`l3cube-pune/indic-sentence-bert-nli`](https://huggingface.co/l3cube-pune/indic-sentence-bert-nli)
for sentence embeddings. These encoder models are downloaded from Hugging Face
on first use and are cached locally, so the first run requires internet access.

## Requirements

- Python 3.10 or newer
- Node.js 18 or newer
- npm
- Internet access for initial encoder download

## Installation

Clone the repository and enter the project directory:

```bash
git clone git@gitlab.com:icfoss/Internship-projects/malayalam-extractive-summarizer.git
cd malayalam-extractive-summarizer
```

Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the backend and frontend dependencies:

```bash
./scripts.sh install
```

## Run the application

Start the backend and frontend together:

```bash
./scripts.sh dev
```

Open the frontend at:

```text
http://127.0.0.1:5173
```

The API is available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

The services can also be started separately:

```bash
./scripts.sh backend
./scripts.sh frontend
```

## API example

Send a Malayalam article to `POST /summarize`:

```bash
curl -X POST http://127.0.0.1:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "മലയാളം ലേഖനം ഇവിടെ നൽകുക.",
    "sentence_count": 3,
    "diversity": "auto",
    "model": "chotta_bheem"
  }'
```

Supported model keys are:

- `chotta_bheem`
- `chotta_bheem_v2`
- `sentence_classifier`
- `hybrid_classifier`
- `indicbert_classifier`

## Frontend configuration

The frontend connects to `http://127.0.0.1:8000` by default. To use a different
API address, set `VITE_API_URL` before starting or building the frontend:

```bash
VITE_API_URL=https://example.com/api ./scripts.sh build
```

## Project structure

```text
.
├── backend/
│   ├── data/                    # Training and evaluation datasets
│   ├── models/                  # Trained classifier checkpoints
│   ├── tests/                   # Backend tests
│   ├── main.py                  # FastAPI application
│   ├── summarize.py             # Inference pipeline
│   ├── dmmr.py                  # Redundancy-aware selection
│   ├── sentence_splitter.py     # Malayalam sentence segmentation
│   └── requirements.txt         # Python dependencies
├── frontend/
│   ├── public/
│   ├── src/                     # React application
│   └── package.json             # Frontend dependencies
└── scripts.sh                   # Installation, development, and build commands
```

## Testing and build

Run the backend test suite:

```bash
python -m unittest discover -s backend/tests -v
```

Build the frontend:

```bash
./scripts.sh build
```

## Notes

- The repository contains the trained classifier checkpoints, but not the full
  pretrained sentence encoders downloaded from Hugging Face.
- CPU inference is supported. A compatible accelerator may be selected by
  PyTorch when available.
- Some files under `backend/` are research, training, or evaluation utilities;
  `backend/main.py` and `backend/summarize.py` form the application inference
  path.
