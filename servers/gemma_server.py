"""OpenAI-compatible chat server for Gemma 3 (google/gemma-3-4b-it), the third model of the comparison (D-083).

The same API and response shape as servers/qolda_server.py and servers/qwen_vl_server.py, so the bench talks to it
through the same backend code:

* first-token ``top_logprobs`` from the raw logits, for calibrated yes/no verification;
* images go through Gemma's own processor (a fixed 896x896 view, 256 visual tokens; no pan-and-scan), so there is
  no pixel budget to set; the visual tokens used are returned in ``kznav``;
* the instruction-tuned checkpoint has no thinking mode, and a request asking for it is refused;
* ``GET /kznav/stats`` / ``POST /kznav/reset_peak`` report CUDA memory (VRAM measurement, D-006);
* greedy decoding, bf16 weights, binds to 127.0.0.1 by default.

Gemma's chat template folds the system message into the first user turn and requires user / assistant turns to
alternate; the bench's prompts do.

Run in the Qolda venv (transformers >= 4.50):
    python servers/gemma_server.py --model google/gemma-3-4b-it --served-name gemma-3-4b --port 23335
"""
from __future__ import annotations

import argparse
import time
from typing import Any, Callable, Dict, List, Tuple

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict

from qwen_vl_server import decode_image, openai_response  # noqa: E402  (servers/ is the script's directory)


def to_gemma_messages(messages: List[dict], decode: Callable[[str], Image.Image] = decode_image
                      ) -> Tuple[List[dict], List[Image.Image]]:
    """OpenAI chat messages -> Gemma chat-template messages (content as typed parts) + the images in order. In a
    message with images the images come first, then the text, as for the other two servers."""
    out: List[dict] = []
    images: List[Image.Image] = []
    for m in messages:
        c = m["content"]
        if isinstance(c, str):
            out.append({"role": m["role"], "content": [{"type": "text", "text": c}]})
            continue
        texts = [p.get("text", "") for p in c if p.get("type") == "text"]
        imgs = [decode(p["image_url"]["url"]) for p in c if p.get("type") == "image_url"]
        images += imgs
        out.append({"role": m["role"], "content": [{"type": "image"} for _ in imgs]
                    + [{"type": "text", "text": " ".join(texts)}]})
    return out, images


class GemmaEngine:
    """Gemma 3 loaded in bfloat16 on one GPU; answers one chat request at a time.

    Args:
        model_path: Hugging Face id or local path of the checkpoint.
        device: Torch device.
    """
    def __init__(self, model_path: str, device: str = "cuda"):
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(device)
        self.device = device
        self.image_token_id = getattr(self.model.config, "image_token_index", None)

    @torch.no_grad()
    def chat(self, messages: List[dict], *, temperature: float, max_tokens: int, want_logprobs: bool,
             top_logprobs: int) -> Dict[str, Any]:
        """Generate one answer to OpenAI-style messages.

        Args:
            messages: OpenAI chat messages (text and ``image_url`` parts).
            temperature: Sampling temperature; 0 decodes greedily.
            max_tokens: Maximum number of new tokens.
            want_logprobs: Also return the top log-probabilities of the first generated token.
            top_logprobs: How many of them.

        Returns:
            The text, latency, token counts, first-token top log-probabilities and image token counts.
        """
        gmsgs, images = to_gemma_messages(messages)
        prompt = self.processor.apply_chat_template(gmsgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[images] if images else None, return_tensors="pt").to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        n_image_tokens = int((inputs["input_ids"] == self.image_token_id).sum()) if self.image_token_id is not None else 0
        tok = self.processor.tokenizer
        gen: Dict[str, Any] = dict(max_new_tokens=max_tokens, do_sample=temperature > 0, temperature=None,
                                   top_p=None, top_k=None, output_logits=want_logprobs, return_dict_in_generate=True,
                                   pad_token_id=tok.pad_token_id or tok.eos_token_id)
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
                top = [{"token": tok.decode([i]), "logprob": v} for i, v in zip(k.indices.tolist(), k.values.tolist())]
            text = self.processor.batch_decode(new, skip_special_tokens=True)[0].strip()
            return {"text": text, "latency_s": latency, "prompt_tokens": n_prompt, "completion_tokens": int(new.shape[-1]),
                    "top_logprobs": top, "image_tokens": [n_image_tokens] if images else []}
        finally:
            del inputs, out
            torch.cuda.empty_cache()


class ChatRequest(BaseModel):
    """Body of ``/v1/chat/completions``: the OpenAI fields the bench uses (other fields are accepted and ignored)."""
    model_config = ConfigDict(extra="allow")
    model: str
    messages: List[dict]
    temperature: float = 0.0
    max_tokens: int = 512
    logprobs: bool = False
    top_logprobs: int = 5
    enable_thinking: bool = False


def create_app(engine: GemmaEngine, served_name: str) -> FastAPI:
    """FastAPI app with the OpenAI chat endpoint plus ``/kznav/stats`` and ``/kznav/reset_peak`` (GPU memory)."""
    app = FastAPI(title="kznav Gemma server")

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": served_name, "object": "model", "owned_by": "local"}]}

    @app.get("/kznav/stats")
    def stats():
        g = 2 ** 30
        return {"allocated_gb": torch.cuda.memory_allocated() / g, "reserved_gb": torch.cuda.memory_reserved() / g,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / g, "dtype": "bf16", "torch": torch.__version__}

    @app.post("/kznav/reset_peak")
    def reset_peak():
        torch.cuda.reset_peak_memory_stats()
        return stats()

    @app.post("/v1/chat/completions")
    def completions(req: ChatRequest):
        if req.enable_thinking:
            return JSONResponse({"error": {"message": "the instruction-tuned checkpoint has no thinking mode"}},
                                status_code=400)
        r = engine.chat(req.messages, temperature=req.temperature, max_tokens=req.max_tokens,
                        want_logprobs=req.logprobs, top_logprobs=req.top_logprobs)
        kz = {"image_tokens": r["image_tokens"], "server_latency_s": r["latency_s"], "enable_thinking": False}
        return JSONResponse(openai_response(req.model, r, req.logprobs, kz))

    return app


def main() -> None:
    """Parse the command line, load the model and serve it."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--served-name", default="gemma-3-4b")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=23335)
    args = ap.parse_args()
    uvicorn.run(create_app(GemmaEngine(args.model), args.served_name), host=args.host, port=args.port,
                log_level="warning")


if __name__ == "__main__":
    main()
