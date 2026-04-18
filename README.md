# 🍌 BananaTimer

An AI-powered banana ripeness detector. Upload a photo of a banana and get back its ripeness stage along with an estimate of how many days until it becomes inedible.

## How It Works

BananaTimer uses a fine-tuned **Microsoft ResNet-50** model (via Hugging Face `transformers`) trained on banana images across 4 ripeness stages:

| Stage | Description |
|---|---|
| `unripe` | Green, not ready yet |
| `ripe` | Perfect to eat |
| `overripe` | Soft and spotty, eat soon |
| `inedible` | Past the point of no return |

The model predicts the stage and returns a human-readable time estimate based on the result.

## Project Structure

```
app/
├── main.py          # FastAPI app and lifespan (trains on startup)
├── api/
│   └── routes/
│       ├── health.py    # GET /api/health
│       └── predict.py   # POST /api/predict
├── services/
│   ├── model.py     # Loads and modifies ResNet-50
│   ├── train.py     # Training loop
│   └── predict.py   # Single image prediction logic
└── core/
    ├── config.py    # Environment config
    └── database.py  # Async PostgreSQL session
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the environment file and configure your database URL:

```bash
cp .env.example .env
```

## Training

Place your training images in the following structure:

```
datasets/
├── unripe/
├── ripe/
├── overripe/
└── inedible/
```

Training runs automatically on server startup. The trained weights are saved to `banana_clock_model.pth`.

## Running the API

```bash
fastapi dev app/main.py
```

The server starts at `http://127.0.0.1:8000`. Interactive docs at `http://127.0.0.1:8000/docs`.

## Example API Response

**Request:**
```
POST /api/predict
Content-Type: multipart/form-data
file: banana.jpg
```

**Response:**
```json
{
  "ripeness": "ripe",
  "days_until_inedible": "Perfect now! 4-6 days until overripe"
}
```

## Stack

- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [FastAPI](https://fastapi.tiangolo.com/)
- [PostgreSQL](https://www.postgresql.org/) + [SQLAlchemy](https://www.sqlalchemy.org/)
