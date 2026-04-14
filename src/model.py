"""
BERT Question Answering — singleton model loader + inference.

Key improvements over the original:
  * Thread-safe lazy singleton (load once per worker, guarded by a lock)
  * Uses AutoTokenizer/AutoModel so the model is swappable via env var
  * Optional warmup pass to eliminate first-request latency
  * No silent re-loading; failures surface as real exceptions
"""
import logging
import threading
import time
from typing import Optional

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from . import config

logger = logging.getLogger(__name__)

_model: Optional[AutoModelForQuestionAnswering] = None
_tokenizer: Optional[AutoTokenizer] = None
_load_lock = threading.Lock()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_model(warmup: bool = None) -> None:
    """Load the model + tokenizer exactly once. Safe to call repeatedly."""
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return

    with _load_lock:
        if _model is not None and _tokenizer is not None:
            return  # another thread won the race

        start = time.time()
        logger.info(f"Loading model '{config.MODEL_NAME}' on {_device}…")
        _tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        _model = AutoModelForQuestionAnswering.from_pretrained(config.MODEL_NAME)
        _model.to(_device)
        _model.eval()
        logger.info(f"Model loaded in {time.time() - start:.1f}s")

    do_warmup = config.WARMUP_ON_START if warmup is None else warmup
    if do_warmup:
        try:
            _warmup()
        except Exception:
            logger.warning("Warmup inference failed", exc_info=True)


def _warmup() -> None:
    """Run a tiny inference so the first real request isn't slow."""
    t0 = time.time()
    predict_qa(
        question="What is this?",
        context="This is a warmup context used to prime the model cache.",
        _persist=False,
    )
    logger.info(f"Warmup inference completed in {(time.time() - t0) * 1000:.0f}ms")


def _require_model():
    if _model is None or _tokenizer is None:
        init_model()


def _truncate_context(context: str, max_chars: int) -> str:
    if len(context) <= max_chars:
        return context
    truncated = context[:max_chars]
    last_dot = truncated.rfind(".")
    if last_dot > max_chars * 0.7:
        truncated = truncated[: last_dot + 1]
    return truncated


def predict_qa(question: str, context: str, _persist: bool = True) -> dict:
    """
    Extract an answer span for `question` from `context` using BERT.

    Returns a dict with the answer, confidence, token breakdown,
    answer character offsets, and inference time — a superset of
    what the frontend needs to render the full UI.
    """
    _require_model()

    question = (question or "").strip()
    context = (context or "").strip()
    if not question or not context:
        raise ValueError("Both question and context are required.")

    ctx = _truncate_context(context, config.MAX_CONTEXT_CHARS)

    inputs = _tokenizer(
        question,
        ctx,
        return_tensors="pt",
        max_length=config.MAX_SEQ_LENGTH,
        truncation="only_second",
        return_offsets_mapping=True,
        padding=False,
    )
    offset_mapping = inputs.pop("offset_mapping")[0].tolist()
    inputs_on_device = {k: v.to(_device) for k, v in inputs.items()}

    input_ids = inputs["input_ids"]
    token_type_ids = inputs.get("token_type_ids")
    tokens_raw = _tokenizer.convert_ids_to_tokens(input_ids[0])

    t0 = time.time()
    with torch.no_grad():
        outputs = _model(**inputs_on_device)
    inference_ms = int((time.time() - t0) * 1000)

    # Back to CPU for post-processing
    start_logits = outputs.start_logits[0].detach().cpu()
    end_logits = outputs.end_logits[0].detach().cpu()

    best_score, best_s, best_e = -float("inf"), 0, 0
    k = min(5, start_logits.size(0))
    top_starts = torch.topk(start_logits, k).indices.tolist()
    top_ends = torch.topk(end_logits, k).indices.tolist()

    for s in top_starts:
        for e in top_ends:
            if e < s or (e - s) >= 50:
                continue
            # Only accept spans that fall inside the context segment
            if token_type_ids is not None and token_type_ids[0][s].item() != 1:
                continue
            score = start_logits[s].item() + end_logits[e].item()
            if score > best_score:
                best_score, best_s, best_e = score, s, e

    if best_score == -float("inf"):
        best_s = int(torch.argmax(start_logits).item())
        best_e = int(torch.argmax(end_logits).item())
        if best_e < best_s:
            best_e = best_s

    answer_ids = input_ids[0][best_s : best_e + 1]
    answer = _tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    if not answer:
        answer = "(No answer found in the given context)"

    s_probs = torch.softmax(start_logits, dim=0)
    e_probs = torch.softmax(end_logits, dim=0)
    conf = float(s_probs[best_s] * e_probs[best_e])
    conf_level = "high" if conf > 0.6 else ("medium" if conf > 0.2 else "low")

    tokens = []
    for i, tok in enumerate(tokens_raw):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            t = "special"
        elif token_type_ids is not None and token_type_ids[0][i].item() == 0:
            t = "question"
        else:
            t = "context"
        if best_s <= i <= best_e and t == "context":
            t = "answer"
        tokens.append({"text": tok.replace("##", ""), "type": t})

    ans_start_char, ans_end_char = -1, -1
    if best_s < len(offset_mapping) and best_e < len(offset_mapping):
        so, eo = offset_mapping[best_s], offset_mapping[best_e]
        if so and eo:
            ans_start_char, ans_end_char = int(so[0]), int(eo[1])

    logger.info(
        f"QA: q={question[:60]!r} → answer={answer[:60]!r} "
        f"conf={conf:.3f} ({conf_level}) in {inference_ms}ms"
    )

    return {
        "answer": answer,
        "confidence": round(conf, 4),
        "confidence_pct": f"{conf * 100:.1f}%",
        "confidence_level": conf_level,
        "answer_start_char": ans_start_char,
        "answer_end_char": ans_end_char,
        "context_used": ctx,
        "tokens": tokens,
        "num_tokens": len(tokens_raw),
        "inference_time_ms": inference_ms,
    }
