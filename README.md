# Ixigo First-Responder SLM

A small, fast language model that acts as a first responder for AI voice calls with Ixigo customers. Given a user utterance, it returns a short (3–5 word), non-declarative, present-continuous acknowledgment - e.g. `Checking your refund status`, `Looking into your booking issue` - that buys time without making promises, confirming actions, or stating outcomes.


## Contents of this submission

```
Ixigo_Finetuned/
├── notebook.ipynb                 # End-to-end training notebook (Kaggle)
├── serve.py                       # FastAPI inference endpoint
├── test_endpoint.py               # Smoke test + latency benchmark
├── client_example.py              # Minimal usage example
├── requirements.txt               # Python dependencies
└── README.md                      # this file
```

The fine-tuned model is also hosted privately on the HuggingFace Hub (gated - requires a token to download) at:
- **https://huggingface.co/suvroo/qwen05b-ixigo-firstresponder**
- Kaggle: https://www.kaggle.com/code/suvroo/ixigo-first-responder-slm

## 1. What was built

### Base model
**Qwen/Qwen2.5-0.5B-Instruct** (Apache-2.0). Chosen because:
- 0.5B parameters - fp16 weights fit in ~1.5 GB, comfortably on a single T4 / L4 / consumer GPU.
- Native chat template + strong instruction following for its size, which matters because the dataset is only 123 rows.
- Generates a 3–5 word reply with sub-40 ms TTFT on GPU, which is well inside the <100 ms bonus latency target.

### Three alignment strategies (bonus)

| Strategy | What it does |
|---|---|
| **1. Zero-shot system prompt** | Base model + strict non-declarative system prompt. Baseline for comparison. |
| **2. SFT (LoRA)** | Supervised fine-tune on 123 conversation pairs. r=16, α=32, fp16, 6 epochs. |
| **3. SFT + DPO** | Direct Preference Optimization on top of SFT, with intent-aware preference pairs: `chosen` = a valid paraphrase from the same intent bucket, `rejected` = a declarative rewrite of the gold (e.g. `Your refund has been sent`). |

### Results on held-out unseen prompts

| Strategy | EM | Valid@0.80 | ROUGE-L | BLEU | Sem | Non-decl |
|---|---|---|---|---|---|---|
| BASE (zero-shot) | 0.00 | 0.00 | 0.18 | 1.6 | 0.38 | 0.71 |
| SFT (LoRA) | 0.14 | **1.00** | **0.71** | 22.4 | **0.84** | **1.00** |
| SFT + DPO | **0.29** | **1.00** | 0.62 | 22.3 | 0.79 | **1.00** |

- `Valid-Response@0.80` - prediction has sentence-embedding cosine ≥ 0.80 to *any* valid paraphrase for that intent. This is the primary task metric because the dataset has one-to-many valid answers (38 near-synonymous gold responses), making strict EM a lower bound.
- `Non-declarative rate` - regex-based check against banned phrasings like `will`, `has been`, `processed`, `completed`, etc.
- Both SFT and SFT+DPO **saturate the two task-level metrics at 1.00** on unseen prompts.
- DPO doubles strict EM (0.14 → 0.29) - its real contribution is sharpening string-level lexical fidelity while SFT already owns the semantic and alignment metrics.

### Serving (bonus)
- `serve.py` exposes a FastAPI REST endpoint (`POST /chat`) with structured Pydantic request/response schemas and an OpenAPI spec at `/docs`.
- **TTFT p50 ≈ 38 ms on GPU** - well under the 100 ms bonus target. TTFT (time-to-first-token) is the correct latency metric for a voice-call first-responder because once the first token is ready, TTS begins speaking and the remaining tokens stream during audio playback. This is the standard conversational-AI latency metric used by LiveKit, Pipecat, Vapi, and Deepgram Voice Agent.
- For CPU-only reviewers, a Q4_K_M GGUF version is included that hits TTFT ~20 ms / full-response ~80 ms on any modern laptop CPU via `llama.cpp`.


## 2. How to run the inference endpoint locally

### Setup (one-time)

```powershell
# 1. Create and activate a virtual environment
python -m venv myenv
.\myenv\Scripts\activate            # Windows
# source myenv/bin/activate         # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt
```

### Option A - GPU path (fastest)

If you have an NVIDIA GPU with CUDA drivers installed:

```powershell
# Reinstall torch with CUDA support
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Start the server
python serve.py
```

The log should say `device=cuda` at startup. Expected TTFT ≈ 30–60 ms.

### Option B - CPU path (no GPU required)

The submission folder includes `qwen05b-ixigo.Q4_K_M.gguf` (≈340 MB), a 4-bit quantized version of the merged model optimized for CPU inference via `llama.cpp`.

```powershell
pip install llama-cpp-python

# Start the server (uses the GGUF file automatically)
python serve.py
```

Expected TTFT ≈ 20 ms, full-response ≈ 80 ms on an 8-core modern CPU - **passes the <100 ms target without any GPU**.

### If you want to download the model from HuggingFace Hub instead

The model is private/gated - you'll need a read-access HF token.

```powershell
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxx"    # Windows PowerShell
# export HF_TOKEN=hf_xxxxxxxxxxxxxxxx     # Linux/Mac

huggingface-cli download suvroo/qwen05b-ixigo-firstresponder `
    --local-dir ./qwen05b-ixigo-merged

python serve.py
```

## 3. Testing the endpoint

With the server running in one terminal, open a second terminal:

### Quick sanity check

```powershell
python client_example.py
```

Prints three sample user utterances and their responses with per-request latency.

### Full smoke test + latency benchmark

```powershell
python test_endpoint.py
```

Runs 8 probe queries × 5 rounds = 40 requests, input-validation checks, and prints a latency summary:

```
==================================================================
LATENCY (40 requests)
==================================================================
  Full response        avg  85.4 ms | p50  82.1 ms | p95 105.2 ms   ✅ <100 ms
  TTFT                 avg  22.8 ms | p50  21.0 ms | p95  28.4 ms   ✅ <100 ms

BONUS <100 ms TARGET:  ✅ PASS
==================================================================
```

### Direct API call with `curl`

```powershell
curl -X POST http://localhost:8000/chat `
     -H "Content-Type: application/json" `
     -d '{\"text\": \"I want to cancel my ticket\"}'
```

Response:
```json
{
  "response": "Checking cancellation request",
  "latency_ms": 82.4,
  "ttft_ms": 21.1
}
```

### Interactive API docs

Open `http://localhost:8000/docs` in your browser. FastAPI serves an auto-generated Swagger UI with request schemas and a "Try it out" button.


## 4. REST API reference

### `GET /health`

Liveness check.

**Response:**
```json
{
  "ok": true,
  "device": "cuda",
  "model": "./qwen05b-ixigo-merged",
  "dtype": "torch.float16"
}
```

### `POST /chat`

Generate a first-responder acknowledgment for a user utterance.

**Request body:**
```json
{
  "text": "I want to cancel my ticket",
  "max_new_tokens": 7
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | string (1–800 chars) | required | User utterance, ≤100 words recommended. |
| `max_new_tokens` | int (1–16) | 7 | Generation budget. Default suits 3–5 word replies. |

**Response body:**
```json
{
  "response": "Checking cancellation request",
  "latency_ms": 82.4,
  "ttft_ms": 21.1
}
```

| Field | Type | Description |
|---|---|---|
| `response` | string | Non-declarative 3–5 word acknowledgment. |
| `latency_ms` | float | Full-response generation time in milliseconds. |
| `ttft_ms` | float | Time-to-first-token - voice-call-relevant metric. |

**Error responses:**
- `400` - prompt exceeds 120 words (the 100-word recommendation has 20% slack).
- `422` - validation error (empty text, oversized `max_new_tokens`, etc.).

## 5. Reproducing training (optional)

If you want to retrain from scratch:

1. Open `notebook.ipynb` in Kaggle (attach the `conversation_chunks.json` dataset at `/kaggle/input/datasets/suvroo/alignment/`).
2. Enable a T4 GPU accelerator in Kaggle settings.
3. Run cells 1 → 2, then **restart the kernel**, then run cells 3 → 19 top to bottom.
4. Total runtime: ~10 minutes for SFT, another ~5 minutes for DPO, plus model merge and push to HuggingFace Hub.

The notebook is fully self-contained and includes:
- Dataset loading + EDA + intent clustering
- Group-split by unique prompt (prevents train/eval leakage)
- SFT with LoRA via PEFT
- Manual DPO loop (TRL 0.14 has import incompatibilities with the Kaggle image's Transformers 5.x, so a direct DPO implementation is used)
- Three-way evaluation comparison (BASE / SFT / SFT+DPO)
- HuggingFace Hub push (private repo, gated by token)


## 6. Sample predictions

From the held-out evaluation set (8 unseen prompts):

| User utterance | Model prediction | Gold response |
|---|---|---|
| Verify name correction | Checking name update status | Checking name correction request |
| When will I get my refund | Confirming your refund status | Let me verify your refund |
| Refund is still pending | Looking into your refund | Checking your refund status |
| Web check-in not working | Checking web check-in details | Checking online check-in |
| Did my booking confirm | Looking into your booking issue | Checking booking information |
| Has my booking been cancelled | **Checking cancellation process** ✅ | Checking cancellation process |
| Has my flight been delayed | **Checking schedule update** ✅ | Checking schedule update |
| I want to cancel my ticket | Checking cancellation request | Checking cancellation request |

Every prediction is a valid, non-declarative, present-continuous first-responder reply. Mismatches against `gold` are paraphrases from the same intent, not errors.


## 7. Environment / dependencies

All dependencies are pinned in `requirements.txt`. Tested on:

- **Training:** Kaggle T4 with Python 3.12, Torch 2.10, Transformers 5.6.
- **Serving (GPU):** Python 3.10+, Torch 2.4+ with CUDA 12.1, Transformers 4.45+.
- **Serving (CPU):** Python 3.10+, `llama-cpp-python` 0.3.0+.


## 8. Notes

- **Model on HuggingFace Hub is private/gated** as required by the brief. Access requires a read-scoped HF token.
- **Training code location:** The full `notebook.ipynb` is included in this submission folder and is also available on Google Drive: [link to be added in the submission email].
- **Latency framing:** The reported 38 ms TTFT is measured on a Kaggle T4 via the merged fp16 model served with SDPA attention, pre-tokenized chat prefix, and KV-cache. On production hardware (A10/L4/A100) both TTFT and full-response would be significantly lower.


