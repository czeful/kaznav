"""OpenAI-compatible chat server for Qolda (issai/Qolda) with the controls the bench needs.

Adapted from the pilot's fallback server (qolda check/local_server/qolda_server.py), which exists
because lmdeploy 0.17.0 cannot load Qolda's internvl_chat config. Changes for the bench:

* ``max_num_tiles`` per request (or ``--max-tiles`` default): InternVL dynamic tiling otherwise
  picks 1..13 tiles from the image aspect ratio and makes verification latency bimodal (D-007).
  The number of tiles actually used is returned in ``kznav.image_tiles``.
* ``enable_thinking`` per request, passed to the Qwen3 chat template (No-Think everywhere, D-006).
* first-token ``top_logprobs`` for calibrated yes/no verification.
* ``GET /kznav/stats`` reports CUDA memory (VRAM measurement, D-006).
* binds to 127.0.0.1 by default.

Run in the Qolda venv:  python servers/qolda_server.py --model issai/Qolda --served-name qolda-4b --port 23333
"""
from __future__ import annotations

import argparse
import base64
import io
import time
import uuid
from typing import Any, Dict, List, Optional

import torch
import torchvision.transforms as T
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN = "<img>", "</img>", "<IMG_CONTEXT>"


# ---------------------------------------------------------------------------- InternVL tiling
def build_transform(input_size: int) -> T.Compose:
    """Image transform of InternVL tiles: RGB, bicubic resize to a square, tensor, ImageNet normalisation."""
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Tile grid ``(columns, rows)`` whose aspect ratio is closest to the image's (InternVL reference code)."""
    best_ratio_diff, best_ratio = float("inf"), (1, 1)
    area = width * height
    for ratio in target_ratios:
        diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if diff < best_ratio_diff:
            best_ratio_diff, best_ratio = diff, ratio
        elif diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image: Image.Image, max_num: int, image_size: int = 448, use_thumbnail: bool = True):
    """Split an image into at most ``max_num`` square tiles on the closest grid, plus a thumbnail if more than one."""
    w, h = image.size
    ratios = sorted({(i, j) for n in range(1, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
                     if 1 <= i * j <= max_num}, key=lambda x: x[0] * x[1])
    ar = find_closest_aspect_ratio(w / h, ratios, w, h, image_size)
    tw, th = image_size * ar[0], image_size * ar[1]
    blocks = ar[0] * ar[1]
    resized = image.resize((tw, th))
    tiles = [resized.crop(((i % (tw // image_size)) * image_size, (i // (tw // image_size)) * image_size,
                           ((i % (tw // image_size)) + 1) * image_size, ((i // (tw // image_size)) + 1) * image_size))
             for i in range(blocks)]
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def decode_image(data_uri: str) -> Image.Image:
    """Decode a base64 image or ``data:`` URI to an RGB image."""
    b64 = data_uri.split(",", 1)[1] if "," in data_uri else data_uri
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


# ---------------------------------------------------------------------------- model
class QoldaEngine:
    """Qolda (InternVL architecture) loaded in bfloat16 on one GPU; answers one chat request at a time.

    Args:
        model_path: Hugging Face id or local path of the checkpoint.
        device: Torch device.
    """
    def __init__(self, model_path: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                               trust_remote_code=True).eval().to(device)
        self.device = device
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.transform = build_transform(448)

    def _prompt(self, messages: List[dict], enable_thinking: bool, n_images: int) -> str:
        rendered = []
        for m in messages:
            c = m["content"]
            text = " ".join(p.get("text", "") for p in c if p.get("type") == "text") if isinstance(c, list) else c
            if n_images and m is messages[-1]:
                text = "<image>\n" * n_images + text
            rendered.append({"role": m["role"], "content": text})
        return self.tokenizer.apply_chat_template(rendered, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=enable_thinking)

    @torch.no_grad()
    def chat(self, messages: List[dict], *, enable_thinking: bool, temperature: float, max_tokens: int,
             want_logprobs: bool, top_logprobs: int, max_num_tiles: int) -> Dict[str, Any]:
        """Generate one answer to OpenAI-style messages.

        Args:
            messages: OpenAI chat messages (text and ``image_url`` parts).
            enable_thinking: Use the model's thinking mode.
            temperature: Sampling temperature; 0 decodes greedily.
            max_tokens: Maximum number of new tokens.
            want_logprobs: Also return the top log-probabilities of the first generated token.
            top_logprobs: How many of them.
            max_num_tiles: Maximum number of image tiles per image (D-007).

        Returns:
            The text, latency, token counts, first-token top log-probabilities and the tiles used per image.
        """
        images = [decode_image(p["image_url"]["url"]) for m in messages if isinstance(m["content"], list)
                  for p in m["content"] if p.get("type") == "image_url"]
        prompt = self._prompt(messages, enable_thinking, len(images))
        pixel_values, tiles_used = None, []
        if images:
            per_image = [torch.stack([self.transform(t) for t in dynamic_preprocess(img, max_num_tiles)]) for img in images]
            tiles_used = [int(x.shape[0]) for x in per_image]
            pixel_values = torch.cat(per_image).to(torch.bfloat16).to(self.device)
            for n in tiles_used:
                prompt = prompt.replace("<image>", IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * n
                                        + IMG_END_TOKEN, 1)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        gen: Dict[str, Any] = dict(max_new_tokens=max_tokens, do_sample=temperature > 0,
                                   pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        if temperature > 0:
            gen["temperature"] = temperature
        if want_logprobs:
            gen.update(output_scores=True, return_dict_in_generate=True)
        out = None
        try:
            t0 = time.perf_counter()
            out = self.model.generate(pixel_values=pixel_values, input_ids=inputs["input_ids"],
                                      attention_mask=inputs["attention_mask"], **gen)
            latency = time.perf_counter() - t0
            top: List[Dict[str, Any]] = []
            seqs = out.sequences if want_logprobs else out
            if want_logprobs:
                lp = torch.log_softmax(out.scores[0][0].float(), dim=-1)
                k = torch.topk(lp, k=top_logprobs)
                top = [{"token": self.tokenizer.decode([i]), "logprob": v} for i, v in zip(k.indices.tolist(), k.values.tolist())]
            text = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)[0].strip()
            return {"text": text, "latency_s": latency, "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                    "completion_tokens": int(seqs.shape[-1]), "top_logprobs": top, "tiles": tiles_used}
        finally:
            del pixel_values, inputs, out
            torch.cuda.empty_cache()


class ChatRequest(BaseModel):
    """Body of ``/v1/chat/completions``: the OpenAI fields the bench uses plus thinking mode and the tile limit."""
    model_config = ConfigDict(extra="allow")
    model: str
    messages: List[dict]
    temperature: float = 0.0
    max_tokens: int = 512
    logprobs: bool = False
    top_logprobs: int = 5
    enable_thinking: bool = False
    max_num_tiles: Optional[int] = None


def create_app(engine: QoldaEngine, served_name: str, default_tiles: int) -> FastAPI:
    """FastAPI app with the OpenAI chat endpoint plus ``/kznav/stats`` and ``/kznav/reset_peak`` (GPU memory)."""
    app = FastAPI(title="kznav Qolda server")

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": served_name, "object": "model", "owned_by": "local"}]}

    @app.get("/kznav/stats")
    def stats():
        g = 2 ** 30
        return {"allocated_gb": torch.cuda.memory_allocated() / g, "reserved_gb": torch.cuda.memory_reserved() / g,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / g, "default_max_tiles": default_tiles,
                "dtype": "bf16", "torch": torch.__version__}

    @app.post("/kznav/reset_peak")
    def reset_peak():
        """Start a new peak-memory window (measure one configuration's peak, D-006)."""
        torch.cuda.reset_peak_memory_stats()
        return stats()

    @app.post("/v1/chat/completions")
    def completions(req: ChatRequest):
        tiles = int(req.max_num_tiles or default_tiles)
        r = engine.chat(req.messages, enable_thinking=req.enable_thinking, temperature=req.temperature,
                        max_tokens=req.max_tokens, want_logprobs=req.logprobs, top_logprobs=req.top_logprobs,
                        max_num_tiles=tiles)
        lp = None
        if req.logprobs and r["top_logprobs"]:
            first = r["top_logprobs"][0]
            lp = {"content": [{"token": first["token"], "logprob": first["logprob"], "bytes": None,
                               "top_logprobs": [{"token": t["token"], "logprob": t["logprob"], "bytes": None}
                                                for t in r["top_logprobs"]]}], "refusal": None}
        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion", "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": r["text"]}, "finish_reason": "stop",
                         "logprobs": lp}],
            "usage": {"prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                      "total_tokens": r["prompt_tokens"] + r["completion_tokens"]},
            "kznav": {"image_tiles": r["tiles"], "max_num_tiles": tiles, "server_latency_s": r["latency_s"],
                      "enable_thinking": req.enable_thinking},
        })

    return app


def main() -> None:
    """Parse the command line, load the model and serve it."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="issai/Qolda")
    ap.add_argument("--served-name", default="qolda-4b")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=23333)
    ap.add_argument("--max-tiles", type=int, default=4, help="default InternVL max_num tiles (D-007)")
    args = ap.parse_args()
    engine = QoldaEngine(args.model)
    uvicorn.run(create_app(engine, args.served_name, args.max_tiles), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
