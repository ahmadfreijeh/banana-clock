# BananaClock — Architecture

BananaClock is an AI-powered banana ripeness tracker. You upload a photo of a banana, and the system tells you its current ripeness stage. Over multiple scans it fits a linear regression to your scan history and predicts exactly how many days before the banana becomes inedible.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Layout](#2-repository-layout)
3. [How All Files Connect](#3-how-all-files-connect)
4. [Image Upload → Prediction: Full Flow](#4-image-upload--prediction-full-flow)
5. [Why ResNet-50](#5-why-resnet-50)
6. [ML Training Pipeline](#6-ml-training-pipeline)
7. [Daily Banana Tracker](#7-daily-banana-tracker)
8. [Linear Regression Prediction](#8-linear-regression-prediction)
9. [FastAPI Backend](#9-fastapi-backend)
10. [Streamlit Frontend](#10-streamlit-frontend)
11. [Authentication](#11-authentication)
12. [Database Schema & Migrations](#12-database-schema--migrations)
13. [Configuration & Environment](#13-configuration--environment)
14. [Technology Stack](#14-technology-stack)

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      User's Browser                             │
│                    (Streamlit UI)                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │  in-process calls (no HTTP)
                            │
          ┌─────────────────┼──────────────────┐
          │                 │                  │
    ┌─────▼──────┐  ┌───────▼──────┐  ┌───────▼───────┐
    │  predict() │  │ scan_service │  │  scan_service  │
    │  (ML infer)│  │  create_scan │  │predict_inedible│
    └─────┬──────┘  └───────┬──────┘  └───────┬───────┘
          │                 │                  │
    ┌─────▼──────┐          │                  │
    │ResNet-50   │          └──────────┬───────┘
    │.pth weights│                     │
    └────────────┘              ┌──────▼──────┐
                                │  PostgreSQL │
                                │  (scans,    │
                                │   users)    │
                                └─────────────┘

    ┌──────────────────────────────────────────┐
    │            FastAPI (optional)            │
    │  /api/auth  /api/scans  /api/predict     │
    │  (runs separately; Streamlit calls DB    │
    │   directly, not through FastAPI)         │
    └──────────────────────────────────────────┘
```

**Key architectural decision:** Streamlit imports and calls the service layer directly (in the same Python process). It does not make HTTP calls to the FastAPI server. FastAPI is a separate, independently deployable REST API — useful for integrating mobile apps or third-party clients, but Streamlit bypasses it entirely for performance.

---

## 2. Repository Layout

```
banana-clock/
│
├── streamlit_app.py          # Entire Streamlit UI (login, register, scan, history)
│
├── app/
│   ├── main.py               # FastAPI application entry point
│   ├── api/
│   │   ├── router.py         # Aggregates all route modules
│   │   └── routes/
│   │       ├── health.py     # GET /api/health
│   │       ├── auth.py       # POST /api/auth/register, /api/auth/login
│   │       ├── scans.py      # POST/GET /api/scans, GET /api/scans/predict-inedible-day
│   │       └── predict.py    # POST /api/predict (standalone, unused by Streamlit)
│   │
│   ├── core/
│   │   ├── config.py         # Reads env vars (DATABASE_URL, JWT_SECRET, etc.)
│   │   ├── database.py       # SQLAlchemy async engine + session factory
│   │   ├── security.py       # bcrypt hashing, JWT encode/decode
│   │   └── deps.py           # FastAPI dependency: get_current_user (JWT → User)
│   │
│   ├── models/
│   │   ├── base.py           # DeclarativeBase, UUIDMixin, TimestampMixin
│   │   ├── user.py           # User ORM model (email, hashed_password, full_name)
│   │   └── scan.py           # Scan ORM model + STAGE_INDEX_MAP + DAYS_LABEL
│   │
│   └── services/
│       ├── model.py          # load_model() — builds ResNet-50 with custom head
│       ├── train.py          # train_model() — full training loop, saves .pth
│       ├── predict.py        # predict(image) — loads weights, returns ripeness
│       └── scan_service.py   # create_scan, get_scans, predict_inedible_day
│
├── migrations/
│   ├── env.py                # Alembic async migration runner
│   └── versions/
│       ├── eefaab522c4e_initial.py           # Empty initial migration
│       ├── 0407d8696171_add_users_and_scans_tables.py
│       └── 15c979576760_add_full_name_to_users.py
│
├── tests/
│   └── test_health.py        # FastAPI health check test
│
├── banana_clock_model.pth    # Saved model weights (~94 MB)
├── requirements.txt
├── alembic.ini
└── .env.example
```

---

## 3. How All Files Connect

The dependency graph flows from top-level entry points down through layers:

```
streamlit_app.py
  └── app/services/predict.py          (ML inference)
        └── app/services/model.py      (model architecture)
        └── banana_clock_model.pth     (trained weights)
  └── app/services/scan_service.py     (business logic)
        └── app/models/scan.py         (ORM + constants)
        └── app/models/user.py         (ORM)
        └── app/services/predict.py    (inference)
  └── app/core/database.py             (DB sessions)
        └── app/core/config.py         (DATABASE_URL)

app/main.py  (FastAPI entry point)
  └── app/api/router.py
        └── app/api/routes/*.py        (HTTP handlers)
              └── app/services/        (same service layer)
              └── app/core/deps.py     (auth middleware)
                    └── app/core/security.py
  └── app/services/train.py            (runs on startup if RETRAIN=true)
        └── app/services/model.py
```

`app/core/config.py` is the single source of truth for all configuration. It reads environment variables first, then falls back to Streamlit secrets (for cloud deployment), making the same config module work in both Streamlit and FastAPI contexts.

---

## 4. Image Upload → Prediction: Full Flow

This is the core user-facing feature. Here is every step from clicking "Predict" to seeing the result:

```
1. User uploads a JPG/PNG via st.file_uploader()
   └── streamlit_app.py reads image_bytes = uploaded_file.read()

2. PIL opens and decodes the image
   └── image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

3. predict(image) is called directly (app/services/predict.py)

4. Model weights are loaded from disk
   └── model = load_model()             ← builds architecture (model.py)
   └── model.load_state_dict(torch.load("banana_clock_model.pth"))

5. Image is preprocessed
   └── Resize to 224×224 pixels
   └── Convert to PyTorch tensor [C, H, W]
   └── Normalize: subtract ImageNet mean [0.485, 0.456, 0.406]
                  divide by ImageNet std  [0.229, 0.224, 0.225]
   └── Add batch dimension → shape [1, 3, 224, 224]

6. Forward pass through ResNet-50
   └── Image passes through 50-layer frozen backbone
   └── Backbone outputs a 2048-dimensional feature vector
   └── Custom head: Flatten → Linear(2048 → 4)
   └── Outputs 4 raw logit scores (one per class)

7. Predicted class is selected
   └── predicted_idx = torch.argmax(logits, dim=1).item()
   └── CLASS_NAMES[predicted_idx] → one of: overripe, ripe, rotten, unripe

8. Human-readable estimate is looked up
   └── DAYS_LABEL dict in scan.py maps ripeness label to a string
       - "unripe"   → "5-7 days until ripe, 12-14 days until inedible"
       - "ripe"     → "Perfect now! 4-6 days until overripe"
       - "overripe" → "1-2 days left, eat soon!"
       - "rotten"   → "Too late! Time to throw it away"

9. Result displayed in Streamlit
   └── Ripeness label + stage emoji
   └── Days estimate text

10. Scan saved to PostgreSQL
    └── run(_save_scan, image_bytes, user_id)   ← async wrapper
    └── scan_service.create_scan() writes a Scan row
        - user_id, scan_date (UTC now), ripeness, stage_index
```

---

## 5. Why ResNet-50

ResNet-50 (Residual Network, 50 layers) was chosen for several concrete reasons:

**Residual connections solve the vanishing gradient problem.** Deep networks struggle to train because gradients shrink to near-zero during backpropagation. ResNet adds shortcut connections that let gradients flow directly through the network, making 50 layers practically trainable.

**ImageNet pretraining is directly applicable.** The backbone was pretrained on 1.2 million images across 1000 classes. The low-level features it learned (edges, textures, colors, shapes) transfer directly to distinguishing banana ripeness stages, which are primarily visual color and texture differences.

**Transfer learning with a frozen backbone reduces data requirements significantly.** By freezing all backbone parameters and only training the final 4-class linear head, the model converges well even on a small dataset. The only trainable layer is `Linear(2048 → 4)`.

**2048-dimensional feature vector is rich enough.** The backbone's output before the classification head is a 2048-dim vector that encodes high-level visual semantics. For a 4-class problem like ripeness stages, this is more than sufficient — the linear head only needs to learn a simple decision boundary.

**Practical alternatives considered:**
- ResNet-18/34: lighter, but less expressive; the 94 MB model size is acceptable
- EfficientNet/MobileNet: better accuracy/size ratio, but ResNet-50 from Hugging Face with the `microsoft/resnet-50` checkpoint is a well-tested baseline
- Training from scratch: would require orders of magnitude more labeled data

**Architecture diagram:**

```
Input Image [1, 3, 224, 224]
      │
      ▼
ResNet-50 Backbone (FROZEN — ~23M parameters)
  ├── Conv layers, BatchNorm, ReLU
  ├── 4 residual stages (3, 4, 6, 3 blocks)
  └── Global average pooling
      │
      ▼ [1, 2048, 1, 1]  feature map
      │
      ▼
Custom Classifier Head (TRAINABLE — ~8K parameters)
  ├── Flatten → [1, 2048]
  └── Linear(2048 → 4) → [1, 4] logits
      │
      ▼
Argmax → class index 0–3
      │
      ▼
CLASS_NAMES[idx] → "overripe" | "ripe" | "rotten" | "unripe"
```

---

## 6. ML Training Pipeline

Training is triggered at FastAPI startup if the `RETRAIN` environment variable is `"true"`. Set `RETRAIN=false` to skip training and use the existing `banana_clock_model.pth`.

**Dataset structure expected:**
```
datasets/
  train/
    overripe/  *.jpg
    ripe/      *.jpg
    rotten/    *.jpg
    unripe/    *.jpg
  valid/  (same structure)
  test/   (same structure)
```

**Training configuration (`app/services/train.py`):**

| Parameter | Value |
|-----------|-------|
| Epochs | 10 |
| Batch size | 16 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | CrossEntropyLoss |
| Trainable params | Classifier head only |

**Augmentations (training only):**
- `RandomHorizontalFlip` — bananas look the same flipped
- `RandomRotation(15°)` — simulate different photo angles
- `ColorJitter(brightness=0.3, contrast=0.3)` — simulate lighting variation

**Device selection:** MPS (Apple Silicon) → CUDA (NVIDIA GPU) → CPU. Falls back gracefully.

**Output:** saves `banana_clock_model.pth` to the project root.

---

## 7. Daily Banana Tracker

The "tracker" is the combination of the Scan page and History page in Streamlit:

1. **Scan page** — upload a banana photo each day. The result is saved as a `Scan` row in PostgreSQL with the current timestamp, ripeness label, and numeric stage index.

2. **History page** — loads all past scans for the logged-in user, renders a line chart of stage index over time, and computes the linear regression prediction.

The `stage_index` field is the numeric encoding of ripeness used for regression:

| Ripeness | `stage_index` |
|----------|--------------|
| overripe | 1 |
| ripe | 2 |
| rotten | 3 |
| unripe | 4 |

Note: the ordering reflects the enum position in the `RIPENESS_VALUES` list (`['overripe', 'ripe', 'rotten', 'unripe']`), not a biological ordering. The regression solves for `stage_index = 4` as the "inedible" threshold because `unripe` and `rotten` both map near or at 4 in this encoding.

---

## 8. Linear Regression Prediction

`app/services/scan_service.predict_inedible_day()` is the core prediction function.

**Goal:** given a sequence of scans, predict the day number on which the banana becomes inedible (`stage_index ≥ 4`).

**Algorithm step by step:**

```python
# Step 1: Load scans ordered oldest-first
scans = [sorted by scan_date ASC]

# Step 2: Convert dates to relative day numbers
first_date = scans[0].scan_date
days   = [(scan.scan_date - first_date).total_seconds() / 86400  for scan in scans]
stages = [scan.stage_index for scan in scans]

# Step 3: Fit a degree-1 polynomial (straight line) through (days, stages)
# numpy.polyfit returns [slope, intercept] of:  stage = slope * day + intercept
coeffs = numpy.polyfit(days, stages, deg=1)
slope, intercept = coeffs[0], coeffs[1]

# Step 4: Solve for when stage_index reaches 4
#   4 = slope * predicted_day + intercept
#   predicted_day = (4 - intercept) / slope
predicted_day = (4 - intercept) / slope

# Step 5: Days remaining from the most recent scan
days_left = predicted_day - days[-1]
```

**Validation guards** (raises `ValueError` with a descriptive message):
- Fewer than 2 scans
- All scans on the same calendar day (no x-axis spread → can't fit a line)
- All scans at the same stage (horizontal line → slope = 0, division by zero)

**What the returned dict contains:**

```json
{
  "days_left": 3.2,
  "predicted_inedible_day": 7.2,
  "scans": [
    {"date": "2024-01-01", "ripeness": "unripe",   "stage": 4},
    {"date": "2024-01-03", "ripeness": "ripe",     "stage": 2},
    {"date": "2024-01-04", "ripeness": "overripe", "stage": 1}
  ]
}
```

Streamlit uses `scans` to draw a line chart (`df.set_index("date")[["stage"]]`) and displays `days_left` and `predicted_inedible_day` as metric widgets.

---

## 9. FastAPI Backend

The FastAPI app lives in `app/main.py` and is served with `uvicorn app.main:app`.

**Startup lifecycle (`lifespan` context manager):**
- If `RETRAIN=true`: calls `train_model()` in a thread pool (`asyncio.to_thread`) so it does not block the event loop
- If `RETRAIN=false`: skips training, loads the existing `.pth` at inference time

**Endpoints:**

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/health` | None | Liveness check |
| POST | `/api/auth/register` | None | Create account, returns JWT |
| POST | `/api/auth/login` | None | Verify credentials, returns JWT |
| POST | `/api/scans` | Bearer JWT | Upload image, save scan, return result |
| GET | `/api/scans` | Bearer JWT | List all scans for current user |
| GET | `/api/scans/predict-inedible-day` | Bearer JWT | Run linear regression prediction |
| POST | `/api/predict` | None | Standalone prediction (not used by Streamlit) |

**Authentication middleware (`app/core/deps.py`):**
```
Authorization: Bearer <jwt_token>
        │
        ▼
decode_access_token(token)  →  user_id (UUID string)
        │
        ▼
SELECT * FROM users WHERE id = user_id
        │
        ▼
User ORM object injected into route handler via Depends(get_current_user)
```

**Database sessions** use SQLAlchemy async sessions (`AsyncSession`) injected via `Depends(get_db)`. Each request gets its own session, committed and closed automatically.

---

## 10. Streamlit Frontend

`streamlit_app.py` is the complete UI. It imports the service layer directly — no HTTP requests to FastAPI.

**Thread-safe async pattern:**

Streamlit runs in a synchronous context, but the service layer and database calls are all `async`. The `run()` helper function solves this by creating a fresh event loop and database engine in a new thread for each call:

```python
def run(coro_fn, *args, **kwargs):
    # Spawns a new thread with a fresh asyncio event loop
    # Creates a new SQLAlchemy engine + session factory
    # Injects session factory as _sf kwarg into the coroutine
    # Blocks until the coroutine completes, returns the result
```

This pattern is necessary because asyncpg connections are tied to the event loop they were created on — sharing them across Streamlit's re-render threads causes transport errors.

**Page structure:**

```
Sidebar navigation
│
├── Not authenticated
│   ├── 🔐 Login   → streamlit-authenticator login widget
│   └── 📝 Register → form → hash password → save user to DB
│                            → update in-memory credential dict
│
└── Authenticated
    ├── 🍌 Scan
    │   ├── File uploader (jpg, jpeg, png, webp)
    │   ├── Predict button → predict(image) + display result
    │   └── Auto-saves scan to DB after prediction
    │
    └── 📈 History & Prediction
        ├── Load History button
        ├── Line chart: stage index over time
        ├── Metric widgets: days_left, predicted_inedible_day
        └── DataFrame: scan history table
```

**Credential caching:** `_load_credentials()` is decorated with `@st.cache_data(ttl=60)`. It queries all users from the database and builds the dict format expected by `streamlit-authenticator`. The 60-second TTL means new registrations appear within a minute without a page reload.

---

## 11. Authentication

BananaClock uses a stateless JWT-based auth system shared between Streamlit and FastAPI.

**Registration flow:**
1. User submits email + password in Streamlit form
2. Password is truncated to 72 bytes (bcrypt hard limit) and hashed with bcrypt
3. `User` row saved to PostgreSQL
4. In Streamlit, the in-memory credential dict is updated immediately (no TTL wait)

**Login flow:**
1. streamlit-authenticator calls `verify_password(plain, hashed)` against the cached credentials
2. On success, sets `st.session_state["authentication_status"] = True`
3. FastAPI login endpoint does the same check and returns a JWT token

**JWT token structure:**
```json
{
  "sub": "<user_uuid>",
  "exp": "<unix_timestamp>"
}
```
Signed with `JWT_SECRET` using `JWT_ALGORITHM` (default: HS256). Expiry defaults to 60 minutes, configurable via `JWT_EXPIRE_MINUTES`.

**Important note:** Streamlit does not use JWT tokens internally — it uses streamlit-authenticator's cookie-based session. JWT tokens are only used when calling the FastAPI REST API (e.g., from a mobile client).

---

## 12. Database Schema & Migrations

**Schema:**

```sql
-- users
CREATE TABLE users (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email        VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name    VARCHAR(255) NOT NULL DEFAULT '',
    created_at   TIMESTAMPTZ NOT NULL,
    updated_at   TIMESTAMPTZ NOT NULL
);
CREATE INDEX ix_users_email ON users(email);

-- scans
CREATE TABLE scans (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID NOT NULL REFERENCES users(id),
    scan_date    TIMESTAMPTZ NOT NULL,
    ripeness     VARCHAR(20) NOT NULL,
    stage_index  INTEGER NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL,
    updated_at   TIMESTAMPTZ NOT NULL
);
CREATE INDEX ix_scans_user_id ON scans(user_id);
```

**Running migrations:**
```bash
alembic upgrade head
```

**Migration history:**
1. `eefaab522c4e` — empty initial revision
2. `0407d8696171` — creates `users` and `scans` tables
3. `15c979576760` — adds `full_name` to `users` (uses a safe three-step: add nullable, backfill, enforce NOT NULL)

To create a new migration after changing ORM models:
```bash
alembic revision --autogenerate -m "describe your change"
alembic upgrade head
```

---

## 13. Configuration & Environment

All configuration is in `app/core/config.py`. It reads environment variables first, then Streamlit secrets as a fallback (for cloud deployments).

**Required variables:**

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | Async PostgreSQL URL | `postgresql+asyncpg://user:pass@localhost:5432/banana_clock_db` |
| `JWT_SECRET` | Secret key for signing JWTs | any long random string |
| `JWT_ALGORITHM` | JWT signing algorithm | `HS256` |
| `JWT_EXPIRE_MINUTES` | Token lifetime | `60` |
| `RETRAIN` | Train model on FastAPI startup | `true` or `false` |

**Optional:**

| Variable | Description |
|----------|-------------|
| `TORCH_DEVICE` | Override device (`cpu`, `cuda`, `mps`) |

Copy `.env.example` to `.env` and fill in values before running.

---

## 14. Technology Stack

| Layer | Library | Why |
|-------|---------|-----|
| **UI** | Streamlit | Rapid data app framework; built-in widgets, caching, and state |
| **UI Auth** | streamlit-authenticator | Integrates login/logout with Streamlit's session state |
| **REST API** | FastAPI | Async-native, automatic OpenAPI docs, Pydantic validation |
| **ASGI server** | Uvicorn | Async server required by FastAPI |
| **ORM** | SQLAlchemy 2 (async) | Async session support; type-safe mapped columns |
| **DB driver** | asyncpg | Native async PostgreSQL driver |
| **Migrations** | Alembic | Version-controlled schema changes |
| **Database** | PostgreSQL | Reliable, supports UUID natively, timezone-aware timestamps |
| **ML framework** | PyTorch | Industry standard; integrates with Hugging Face |
| **Model hub** | Hugging Face Transformers | `microsoft/resnet-50` checkpoint with consistent API |
| **Vision utils** | torchvision | Standard image transforms and tensor utilities |
| **Regression** | NumPy (`polyfit`) | Simple, reliable 1D polynomial fitting |
| **Image I/O** | Pillow | Decodes all common image formats to RGB tensors |
| **Auth tokens** | python-jose | JWT encode/decode |
| **Password hash** | passlib + bcrypt | Industry-standard password hashing |
| **Validation** | Pydantic v2 | Request/response schema validation in FastAPI |
| **Config** | python-dotenv | `.env` file loading |

---

## Quick Start for New Developers

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your DATABASE_URL and JWT_SECRET

# 3. Run database migrations
alembic upgrade head

# 4. Start the FastAPI server (optional; Streamlit doesn't require it)
#    Set RETRAIN=false if you don't have the training dataset
RETRAIN=false uvicorn app.main:app --reload

# 5. Start Streamlit
streamlit run streamlit_app.py
```

To retrain the model, place the dataset under `datasets/train|valid|test` with one subfolder per class, set `RETRAIN=true`, and start the FastAPI server. The model will train for 10 epochs and save weights to `banana_clock_model.pth`.
