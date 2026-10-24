"""D-075: the local SigLIP verifier backend (no language model) answers the verifier like a VLM would."""
import math

import numpy as np
import pytest
import torch

from kznav.language.schemas import VerifyQuery
from kznav.policy.local_verifier import SiglipVerifierBackend
from kznav.policy.verification import Verifier, VerifyParams

D = 4


class FakeModel(torch.nn.Module):
    """Image features: the mean colour, padded; logit scale 10, bias -2 (SigLIP-like)."""

    def __init__(self):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(10.0)))
        self.logit_bias = torch.nn.Parameter(torch.tensor(-2.0))

    def get_image_features(self, pixel_values):
        return torch.cat([pixel_values.mean(dim=(2, 3)), torch.ones(pixel_values.shape[0], D - 3)], dim=1)


class FakeProcessor:
    def __call__(self, images=None, return_tensors=None):
        a = torch.tensor(np.asarray(images, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        return _Batch(pixel_values=a)


class _Batch(dict):
    def to(self, device):
        return self


class FakeEncoder:
    model_id, template, device, dtype = "fake/siglip", "a photo of a {}.", "cpu", torch.float32

    def __init__(self):
        self.torch, self.model, self.processor = torch, FakeModel(), FakeProcessor()
        self.texts = {"bed": [1.0, 0.0, 0.0, 1.0], "door": [0.0, 1.0, 0.0, 1.0]}

    def _text_embeddings(self, texts):
        t = torch.tensor([self.texts[x] for x in texts])
        return t / t.norm(dim=-1, keepdim=True)


RED = np.zeros((32, 32, 3), dtype=np.uint8)
RED[..., 0] = 255


def head_file(tmp_path, model_id="fake/siglip", template="a photo of a {}."):
    head = torch.nn.Sequential(torch.nn.Linear(3 * D, 8), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(8, 1))
    f = tmp_path / "head.pt"
    torch.save({"state_dict": head.state_dict(), "in_dim": 3 * D, "hidden": 8, "model_id": model_id,
                "template": template}, f)
    return f, head


def test_zero_shot_is_siglips_own_sigmoid_and_prefers_the_matching_name():
    b = SiglipVerifierBackend(FakeEncoder(), "siglip_zero")
    img = torch.tensor([1.0, 0.0, 0.0, 1.0]) / math.sqrt(2)
    expected = 1 / (1 + math.exp(-(10 * float(img @ (torch.tensor([1.0, 0, 0, 1]) / math.sqrt(2))) - 2)))
    r = b.verify_candidate(RED, VerifyQuery("bed", "en"))
    assert r.p_target == pytest.approx(expected, rel=1e-5) and r.source == "siglip_zero"
    assert r.p_target > b.verify_candidate(RED, VerifyQuery("door", "en")).p_target


def test_head_mode_applies_the_trained_head(tmp_path):
    f, head = head_file(tmp_path)
    b = SiglipVerifierBackend(FakeEncoder(), "siglip_head", str(f))
    img = torch.tensor([1.0, 0.0, 0.0, 1.0]) / math.sqrt(2)
    txt = torch.tensor([1.0, 0.0, 0.0, 1.0]) / math.sqrt(2)
    head.eval()
    expected = float(torch.sigmoid(head(torch.cat([img, txt, img * txt]).unsqueeze(0)))[0, 0])
    assert b.verify_candidate(RED, VerifyQuery("bed", "en")).p_target == pytest.approx(expected, rel=1e-5)


def test_a_head_trained_on_another_encoder_is_refused(tmp_path):
    f, _ = head_file(tmp_path, model_id="other/siglip")
    with pytest.raises(ValueError, match="trained on"):
        SiglipVerifierBackend(FakeEncoder(), "siglip_head", str(f))
    with pytest.raises(ValueError):
        SiglipVerifierBackend(None, "siglip_zero")


def test_the_verifier_uses_the_local_backend_like_a_vlm():
    from kznav.perception.objects import ObjectCandidate

    params = VerifyParams(stop_threshold=0.6, reject_below=0.25, base_rate=0.5, max_calls=3, regrow_factor=2.0,
                          max_wait_steps=2, crop_padding=0.25, min_crop_px=16, calibrated=False)
    v = Verifier(SiglipVerifierBackend(FakeEncoder(), "siglip_zero"), VerifyQuery("bed", "en"), params, None,
                 view_discount=0.5, first_view_full=True)
    c = ObjectCandidate(0, 0, np.zeros(3), np.zeros((1, 2)), 0.6, 1, 0, 0, scores={0: 0.6}, last_box=(4, 4, 28, 28))
    ev = v.check(c, RED, 0, 1)
    assert ev.called and ev.source == "siglip_zero" and ev.decision == "stop"
