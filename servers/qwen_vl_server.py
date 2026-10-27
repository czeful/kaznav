"""OpenAI-compatible chat server for Qwen3-VL (Qwen/Qwen3-VL-4B-Instruct) with the controls the bench needs.

The same API and response shape as servers/qolda_server.py, so the bench talks to both models through
the same backend code (D-040):

* first-token ``top_logprobs`` from the raw logits, for calibrated yes/no verification;
* an image pixel budget instead of InternVL tiles: every image is resized (aspect ratio kept) to between
  ``min_pixels`` and ``max_pixels``. The defaults match what Qolda sees with 4 tiles: at least one 448x448
  tile, at most four (D-040). The pixels and visual tokens actually used are returned in ``kznav``;
* No-Think: the Instruct checkpoint has no thinking mode, and a request asking for it is refused;
* ``GET /kznav/stats`` / ``POST /kznav/reset_peak`` report CUDA memory (VRAM measurement, D-006);
* greedy decoding, bf16 weights, binds to 127.0.0.1 by default.

Run in the Qolda venv (transformers >= 4.57):
    python servers/qwen_vl_server.py --model Qwen/Qwen3-VL-4B-Instruct --served-name qwen3-vl-4b --port 23334
"""
from __future__ import annotations

import argparse
import base64
import io
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict

TILE = 448
DEFAULT_MIN_PIXELS = TILE * TILE          # one InternVL tile
DEFAULT_MAX_PIXELS = 4 * TILE * TILE      # four tiles (D-007's default for Qolda)


def decode_image(data_uri: str) -> Image.Image:
    """Decode a base64 image or ``data:`` URI to an RGB image."""
    b64 = data_uri.split(",", 1)[1] if "," in data_uri else data_uri
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def to_qwen_messages(messages: List[dict], decode: Callable[[str], Image.Image] = decode_image
                     ) -> Tuple[List[dict], List[Image.Image]]:
    """OpenAI chat messages -> Qwen chat-template messages + the images in order. Text-only messages keep a
    plain string; in a message with images the images come first, then the text (as the Qolda server
    puts its <image> tokens before the text)."""
    out: List[dict] = []
    images: List[Image.Image] = []
    for m in messages:
        c = m["content"]
        if isinstance(c, str):
            out.append({"role": m["role"], "content": c})
            continue
        texts = [p.get("text", "") for p in c if p.get("type") == "text"]
        imgs = [decode(p["image_url"]["url"]) for p in c if p.get("type") == "image_url"]
        if not imgs:
            out.append({"role": m["role"], "content": " ".join(texts)})
            continue
        images += imgs
        out.append({"role": m["role"], "content": [{"type": "image"} for _ in imgs]
                    + [{"type": "text", "text": " ".join(texts)}]})
    return out, images


def openai_response(model: str, r: Dict[str, Any], want_logprobs: bool, kznav: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap an engine result as an OpenAI chat completion, with the bench's extra fields under ``kznav``."""
    lp = None
    if want_logprobs and r["top_logprobs"]:
        first = r["top_logprobs"][0]
        lp = {"content": [{"token": first["token"], "logprob": first["logprob"], "bytes": None,
                           "top_logprobs": [{"token": t["token"], "logprob": t["logprob"], "bytes": None}
                                            for t in r["top_logprobs"]]}], "refusal": None}
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion", "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": r["text"]}, "finish_reason": "stop",
                     "logprobs": lp}],
        "usage": {"prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                  "total_tokens": r["prompt_tokens"] + r["completion_tokens"]},
        "kznav": kznav,
    }


# ---------------------------------------------------------------------------- model
class QwenVLEngine:
    """Qwen3-VL loaded in bfloat16 on one GPU; answers one chat request at a time.

    Args:
        model_path: Hugging Face id or local path of the checkpoint.
        device: Torch device.
    """
    def __init__(self, model_path: str, device: str = "cuda"):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(device)
        self.device = device
        ip = self.processor.image_processor
        self.merge = int(getattr(ip, "merge_size", 2))

    def _set_pixel_budget(self, min_pixels: int, max_pixels: int) -> None:
        # Requests are served one at a time (measurement protocol), so the processor can be reconfigured.
        ip = self.processor.image_processor
        ip.size = {"shortest_edge": int(min_pixels), "longest_edge": int(max_pixels)}
        ip.min_pixels, ip.max_pixels = int(min_pixels), int(max_pixels)

    @torch.no_grad()
    def chat(self, messages: List[dict], *, temperature: float, max_tokens: int, want_logprobs: bool,
             top_logprobs: int, min_pixels: int, max_pixels: int) -> Dict[str, Any]:
        """Generate one answer to OpenAI-style messages.

        Args:
            messages: OpenAI chat messages (text and ``image_url`` parts).
            temperature: Sampling temperature; 0 decodes greedily.
            max_tokens: Maximum number of new tokens.
            want_logprobs: Also return the top log-probabilities of the first generated token.
            top_logprobs: How many of them.
            min_pixels: Minimum pixel budget per image.
            max_pixels: Maximum pixel budget per image (D-040).

        Returns:
            The text, latency, token counts, first-token top log-probabilities, image grids and image token counts.
        """
        qmsgs, images = to_qwen_messages(messages)
        prompt = self.processor.apply_chat_template(qmsgs, tokenize=False, add_generation_prompt=True)
        self._set_pixel_budget(min_pixels, max_pixels)
        inputs = self.processor(text=[prompt], images=images or None, return_tensors="pt").to(self.device)
        grid = inputs["image_grid_thw"].tolist() if images else []
        gen: Dict[str, Any] = dict(max_new_tokens=max_tokens, do_sample=temperature > 0, temperature=None,
                                   top_p=None, top_k=None, repetition_penalty=None,
                                   output_logits=want_logprobs, return_dict_in_generate=True,
                                   pad_token_id=self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id)
        if temperature > 0:
            gen["temperature"] = temperature
        out = None
        try:
            t0 = time.perf_counter()
            out = self.model.generate(**inputs, **gen)
            latency = time.perf_counter() - t0
            n_prompt = int(inputs["input_ids"].shape[-1])
            new = out.sequences[:, n_prompt:]
            top: List[Dict[str, Any]] = []
            if want_logprobs:
                lp = torch.log_softmax(out.logits[0][0].float(), dim=-1)
                k = torch.topk(lp, k=top_logprobs)
                top = [{"token": self.processor.tokenizer.decode([i]), "logprob": v}
                       for i, v in zip(k.indices.tolist(), k.values.tolist())]
            text = self.processor.batch_decode(new, skip_special_tokens=True)[0].strip()
            return {"text": text, "latency_s": latency, "prompt_tokens": n_prompt, "completion_tokens": int(new.shape[-1]),
                    "top_logprobs": top, "image_grid_thw": grid,
                    "image_tokens": [int(t * h * w) // (self.merge ** 2) for t, h, w in grid]}
        finally:
            del inputs, out
            torch.cuda.empty_cache()


class ChatRequest(BaseModel):
    """Body of ``/v1/chat/completions``: the OpenAI fields the bench uses plus the per-request pixel budget."""
    model_config = ConfigDict(extra="allow")
    model: str
    messages: List[dict]
    temperature: float = 0.0
    max_tokens: int = 512
    logprobs: bool = False
    top_logprobs: int = 5
    enable_thinking: bool = False
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None


def create_app(engine: QwenVLEngine, served_name: str, min_pixels: int, max_pixels: int) -> FastAPI:
    """FastAPI app with the OpenAI chat endpoint plus ``/kznav/stats`` and ``/kznav/reset_peak`` (GPU memory)."""
    app = FastAPI(title="kznav Qwen-VL server")

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": served_name, "object": "model", "owned_by": "local"}]}

    @app.get("/kznav/stats")
    def stats():
        g = 2 ** 30
        return {"allocated_gb": torch.cuda.memory_allocated() / g, "reserved_gb": torch.cuda.memory_reserved() / g,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / g, "default_min_pixels": min_pixels,
                "default_max_pixels": max_pixels, "dtype": "bf16", "torch": torch.__version__}

    @app.post("/kznav/reset_peak")
    def reset_peak():
        torch.cuda.reset_peak_memory_stats()
        return stats()

    @app.post("/v1/chat/completions")
    def completions(req: ChatRequest):
        if req.enable_thinking:
            return JSONResponse({"error": {"message": "the Instruct checkpoint has no thinking mode"}}, status_code=400)
        mn, mx = int(req.min_pixels or min_pixels), int(req.max_pixels or max_pixels)
        r = engine.chat(req.messages, temperature=req.temperature, max_tokens=req.max_tokens,
                        want_logprobs=req.logprobs, top_logprobs=req.top_logprobs, min_pixels=mn, max_pixels=mx)
        kz = {"image_tokens": r["image_tokens"], "image_grid_thw": r["image_grid_thw"], "min_pixels": mn,
              "max_pixels": mx, "server_latency_s": r["latency_s"], "enable_thinking": False}
        return JSONResponse(openai_response(req.model, r, req.logprobs, kz))

    return app


def main() -> None:
    """Parse the command line, load the model and serve it."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--served-name", default="qwen3-vl-4b")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS, help="one 448x448 tile (D-040)")
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS, help="four 448x448 tiles (D-040)")
    args = ap.parse_args()
    engine = QwenVLEngine(args.model)
    uvicorn.run(create_app(engine, args.served_name, args.min_pixels, args.max_pixels),
                host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
