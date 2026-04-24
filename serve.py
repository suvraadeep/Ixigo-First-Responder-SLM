"""
REST inference endpoint.

Optimizations stacked for <100 ms TTFT on a single consumer GPU:
  1. Merged fp16 weights (no LoRA adapter overhead)
  2. SDPA attention (faster than eager)
  3. Pre-tokenized system+user chat prefix (no Jinja rebuild per call)
  4. Dual EOS stopping (<|eos|> + <|im_end|>)
  5. 10-iter startup warmup so first request is not slow
"""
import os
import time
import logging

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "./qwen05b-ixigo-merged")
HF_TOKEN   = os.environ.get("HF_TOKEN")          # only needed for private HF repos
PORT       = int(os.environ.get("PORT", "8000"))
HOST       = os.environ.get("HOST", "0.0.0.0")

SYSTEM_PROMPT = (
    "You are Ixigo's AI first-responder on a live voice call. "
    "Reply with a SHORT (3-5 word) present-continuous acknowledgment that BUYS TIME. "
    "NEVER promise outcomes, confirm completed actions, state refund amounts, "
    "or say anything is done. Only acknowledge that you are looking into it."
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("serve")

# Model + tokenizer (loaded once at process start)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32
log.info(f"Loading model from {MODEL_PATH} onto {device} ({dtype})...")

_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
tok = AutoTokenizer.from_pretrained(MODEL_PATH, **_kwargs)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=dtype,
    attn_implementation="sdpa",
    **_kwargs,
).to(device).eval()
mdl.config.use_cache = True

# Pre-tokenize the static chat prefix ONCE — avoids per-call Jinja rebuild (~30 ms saved)
SYSTEM_PREFIX = tok.apply_chat_template(
    [{"role": "system", "content": SYSTEM_PROMPT}],
    add_generation_prompt=False, tokenize=False,
)
USER_PRE  = "<|im_start|>user\n"
USER_POST = "<|im_end|>\n<|im_start|>assistant\n"

IM_END_ID = tok.convert_tokens_to_ids("<|im_end|>")
EOS_ID    = tok.eos_token_id
EOS_IDS   = [EOS_ID] + ([IM_END_ID] if IM_END_ID != tok.unk_token_id else [])

SYS_IDS  = tok(SYSTEM_PREFIX + USER_PRE, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
POST_IDS = tok(USER_POST,                 add_special_tokens=False, return_tensors="pt").input_ids.to(device)


@torch.inference_mode()
def _gen(user_text: str, max_new_tokens: int) -> str:
    user_ids  = tok(user_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    input_ids = torch.cat([SYS_IDS, user_ids, POST_IDS], dim=1)
    out = mdl.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=EOS_ID,
        eos_token_id=EOS_IDS,
        use_cache=True,
    )
    return tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


# Warmup (runs at import time so first request is not slow)
log.info("Warming up model (10 iterations)...")
for _ in range(10):
    _gen("hello", 7)
if device == "cuda":
    torch.cuda.synchronize()
log.info("Ready.")


# FastAPI schema
class ChatIn(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=800,
        description="User utterance — prompts up to 100 words recommended.",
    )
    max_new_tokens: int = Field(
        7, ge=1, le=16,
        description="Generation budget. Default 7 fits the 3-5 word output style.",
    )


class ChatOut(BaseModel):
    response: str     = Field(..., description="Non-declarative acknowledgment.")
    latency_ms: float = Field(..., description="Full-response latency in milliseconds.")
    ttft_ms: float    = Field(..., description="Time-to-first-token (voice-call-relevant).")


# App
app = FastAPI(
    title="Ixigo First-Responder SLM",
    description="Non-declarative acknowledgments for live voice-call first-response.",
    version="1.0",
)


@app.get("/health")
def health():
    return {
        "ok": True,
        "device": device,
        "model": MODEL_PATH,
        "dtype": str(dtype),
    }


@app.post("/chat", response_model=ChatOut)
@torch.inference_mode()
def chat(req: ChatIn):
    if len(req.text.split()) > 120:
        raise HTTPException(status_code=400, detail="Prompt exceeds 100-word recommendation.")

    # Measure TTFT via a 1-token pre-gen, then do the full response
    t0 = time.perf_counter()
    _gen(req.text, 1)
    ttft_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    resp = _gen(req.text, req.max_new_tokens)
    latency_ms = (time.perf_counter() - t1) * 1000

    return ChatOut(response=resp, latency_ms=latency_ms, ttft_ms=ttft_ms)


# Run directly: `python serve.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")