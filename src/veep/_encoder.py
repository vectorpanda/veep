"""Lightweight text encoder bundled with the SDK for the quickstart demo.

Wraps a quantized ONNX export of ``sentence-transformers/all-MiniLM-L6-v2``
(384-dim, cosine-normalized) with an in-house WordPiece tokenizer so the
``samples.encode(text)`` quickstart helper can produce real embeddings for
arbitrary text without depending on torch, sentence-transformers, or any
vendor SDK.

The model file (~22MB, INT8 quantized) and BERT vocab (~250KB) ship in
``veep/_sample_data/``. ``onnxruntime`` is an optional dependency installed
via ``pip install veep[samples]``; without it ``encode()`` raises a
``MissingExtraError`` with the install hint.

Customers wanting full-quality production embeddings should still bring
their own model (OpenAI, Cohere, sentence-transformers in their own
environment, etc.). This encoder is for the quickstart, not a production
embedding service.
"""

from __future__ import annotations

import threading
import unicodedata
from importlib import resources
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import onnxruntime as ort

from .exceptions import VeepError

_DATA_PKG = "veep._sample_data"
_MODEL_FILE = "all-MiniLM-L6-v2.int8.onnx"
_VOCAB_FILE = "vocab.txt"
_MAX_LEN = 256

# Cached lazily on first encode() call. Guarded so concurrent threads don't
# race on session creation (ORT session init is expensive — ~50ms on cold
# CPU; we only want to pay it once per process).
_session: ort.InferenceSession | None = None
_vocab: dict[str, int] | None = None
_init_lock = threading.Lock()


class MissingExtraError(VeepError):
    """Raised when samples.encode() is called without onnxruntime installed."""


def _ensure_loaded() -> None:
    global _session, _vocab
    if _session is not None and _vocab is not None:
        return
    with _init_lock:
        if _session is not None and _vocab is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise MissingExtraError(
                "samples.encode() needs the onnxruntime extra. "
                "Install it with `pip install veep[samples]`."
            ) from exc

        with resources.files(_DATA_PKG).joinpath(_MODEL_FILE).open("rb") as f:
            model_bytes = f.read()
        # Single-threaded session — the bundled model is small and customer
        # demos run one query at a time. Customers with throughput needs
        # bring their own embedding pipeline.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        _session = ort.InferenceSession(model_bytes, sess_options=opts, providers=["CPUExecutionProvider"])

        vocab: dict[str, int] = {}
        with resources.files(_DATA_PKG).joinpath(_VOCAB_FILE).open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                vocab[line.rstrip("\n")] = i
        _vocab = vocab


# --- BasicTokenizer: lowercase, NFD strip-accents, split on whitespace + punctuation ---

def _is_punct(ch: str) -> bool:
    cp = ord(ch)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(ch).startswith("P")


def _is_whitespace(ch: str) -> bool:
    if ch in (" ", "\t", "\n", "\r"):
        return True
    return unicodedata.category(ch) == "Zs"


def _is_control(ch: str) -> bool:
    if ch in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(ch).startswith("C")


def _basic_tokenize(text: str) -> list[str]:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    tokens: list[str] = []
    cur: list[str] = []
    for ch in text:
        if _is_control(ch) or ch == "�":
            continue
        if _is_whitespace(ch):
            if cur:
                tokens.append("".join(cur))
                cur = []
        elif _is_punct(ch):
            if cur:
                tokens.append("".join(cur))
                cur = []
            tokens.append(ch)
        else:
            cur.append(ch)
    if cur:
        tokens.append("".join(cur))
    return tokens


# --- WordPiece: greedy longest-match against vocab, ## prefix for continuations ---

def _wordpiece(token: str, vocab: dict[str, int], unk: str = "[UNK]") -> list[str]:
    if len(token) > 100:
        return [unk]
    out: list[str] = []
    start = 0
    while start < len(token):
        end = len(token)
        cur = None
        while start < end:
            sub = token[start:end]
            if start > 0:
                sub = "##" + sub
            if sub in vocab:
                cur = sub
                break
            end -= 1
        if cur is None:
            return [unk]
        out.append(cur)
        start = end
    return out


def _encode_ids(text: str, vocab: dict[str, int]) -> list[int]:
    cls_id = vocab["[CLS]"]
    sep_id = vocab["[SEP]"]
    unk_id = vocab["[UNK]"]
    ids = [cls_id]
    for word in _basic_tokenize(text):
        for sub in _wordpiece(word, vocab):
            ids.append(vocab.get(sub, unk_id))
    ids.append(sep_id)
    return ids[:_MAX_LEN]


def encode(text: str) -> list[float]:
    """Encode ``text`` into a 384-dim cosine-normalized embedding.

    Uses the bundled ``all-MiniLM-L6-v2`` ONNX model (INT8 quantized) with
    an in-house WordPiece tokenizer. Output is L2-normalized so cosine
    similarity equals dot product on the server side.

    Raises:
        MissingExtraError: if ``onnxruntime`` is not installed. Install
            with ``pip install veep[samples]``.
    """
    import numpy as np

    _ensure_loaded()
    assert _session is not None and _vocab is not None

    ids = _encode_ids(text, _vocab)
    seq_len = len(ids)
    input_ids = np.array([ids], dtype=np.int64)
    attention_mask = np.array([[1] * seq_len], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids)

    outputs = _session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )
    last_hidden = outputs[0]  # (1, seq, dim)

    # Mean pool over sequence with attention mask
    mask = attention_mask[:, :, None].astype(np.float32)
    summed = (last_hidden * mask).sum(axis=1)
    counts = mask.sum(axis=1).clip(min=1e-9)
    pooled = summed / counts

    # L2 normalize for cosine
    norm = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
    pooled = pooled / norm
    return pooled[0].tolist()
