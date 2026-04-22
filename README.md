# 🍌 BananaClock

An AI-powered banana ripeness tracker. Upload a photo of a banana, get back its ripeness stage and a days-until-inedible estimate. Scan daily and the app predicts exactly when your banana will go bad using linear regression on your scan history.

## How It Works

BananaClock uses a fine-tuned **ResNet-50** model trained on banana images across 4 ripeness stages:

| Stage      | Description                 |
| ---------- | --------------------------- |
| `unripe`   | Green, not ready yet        |
| `ripe`     | Perfect to eat              |
| `overripe` | Soft and spotty, eat soon   |
| `rotten`   | Past the point of no return |

The model predicts the stage, saves the scan, and after at least 2 scans uses linear regression to forecast when the banana becomes inedible.

## Stack

- **ML:** PyTorch + Hugging Face Transformers (ResNet-50)
- **Backend:** FastAPI + SQLAlchemy (async) + PostgreSQL
- **Frontend:** Streamlit
- **Auth:** JWT (python-jose + passlib/bcrypt)
- **Migrations:** Alembic

## Project Structure

```
app/
├── main.py               # FastAPI entry point (trains model on startup)
├── api/routes/
│   ├── health.py         # GET /api/health
│   ├── auth.py           # POST /api/auth/register, /api/auth/login
│   └── scans.py          # POST/GET /api/scans, GET /api/scans/predict-inedible-day
├── services/
│   ├── model.py          # ResNet-50 architecture
│   ├── train.py          # Training loop
│   ├── predict.py        # Single image inference
│   └── scan_service.py   # Scan CRUD + linear regression prediction
└── core/
    ├── config.py         # Environment config
    ├── database.py       # Async PostgreSQL session
    ├── security.py       # JWT + password hashing
    └── deps.py           # Auth dependency injection
streamlit_app.py          # Streamlit UI
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your database URL and JWT secret.

### Environment Variables

| Variable             | Default | Description                                                          |
| -------------------- | ------- | -------------------------------------------------------------------- |
| `DATABASE_URL`       | —       | PostgreSQL connection string (`postgresql+asyncpg://...`)            |
| `JWT_SECRET`         | —       | Secret key for signing JWT tokens                                    |
| `JWT_ALGORITHM`      | `HS256` | JWT algorithm                                                        |
| `JWT_EXPIRE_MINUTES` | `60`    | Token expiry in minutes                                              |
| `RETRAIN`            | `true`  | Set to `false` to skip retraining and use existing `banana_clock_model.pth` |
| `TORCH_DEVICE`       | auto    | Force device: `mps`, `cuda`, or `cpu`. Auto-detected if unset.      |

## Database

Run migrations before starting the app:

```bash
alembic upgrade head
```

## Training

This project uses the **Banana Ripeness Classification Dataset** from Kaggle:
[shahriar26s/banana-ripeness-classification-dataset](https://www.kaggle.com/datasets/shahriar26s/banana-ripeness-classification-dataset)

```bash
pip install kaggle
kaggle datasets download -d shahriar26s/banana-ripeness-classification-dataset
unzip banana-ripeness-classification-dataset.zip -d datasets/
```

Dataset structure expected:

```
datasets/
├── train/
│   ├── overripe/
│   ├── ripe/
│   ├── rotten/
│   └── unripe/
├── valid/
└── test/
```

Training runs automatically on FastAPI startup when `RETRAIN=true`. Weights are saved to `banana_clock_model.pth`.

To retrain manually:

```bash
python -c "from app.services.train import train_model; train_model()"
```

## Running

**API:**
```bash
fastapi dev app/main.py
```
Docs available at `http://127.0.0.1:8000/docs`.

**Streamlit UI:**
```bash
streamlit run streamlit_app.py
```
Opens at `http://localhost:8501`.

## API Reference

| Method | Endpoint                          | Auth | Description                                    |
| ------ | --------------------------------- | ---- | ---------------------------------------------- |
| `GET`  | `/api/health`                     | ❌   | Health check                                   |
| `POST` | `/api/auth/register`              | ❌   | Register, returns JWT                          |
| `POST` | `/api/auth/login`                 | ❌   | Login, returns JWT                             |
| `POST` | `/api/scans`                      | ✅   | Upload image → predict → save scan             |
| `GET`  | `/api/scans`                      | ✅   | Get all scans for current user                 |
| `GET`  | `/api/scans/predict-inedible-day` | ✅   | Predict inedible date via linear regression    |

**Scan response:**

```json
{
  "id": "uuid",
  "user_id": "uuid",
  "scan_date": "2026-04-19T10:00:00Z",
  "ripeness": "ripe",
  "stage_index": 2,
  "days_until_inedible": "Perfect now! 4-6 days until overripe"
}
```

**Predict inedible day response:**

```json
{
  "days_left": 3.5,
  "predicted_inedible_day": 7.5,
  "scans": [
    { "date": "2026-04-17", "ripeness": "unripe", "stage": 4 },
    { "date": "2026-04-19", "ripeness": "ripe",   "stage": 2 }
  ]
}
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for a full breakdown of how the system works, including the ML pipeline, prediction logic, and how FastAPI and Streamlit fit together.
