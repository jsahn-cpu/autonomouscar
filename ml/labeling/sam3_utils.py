"""Shared SAM3 model loading + VRAM workaround, used by both
score_probe.py and label_with_sam3.py so the tricky bits below only exist
in one place.

Run only inside the `sam3` conda env (Python 3.12) -- see ml/README.md.

All of this was worked out by hand against a 4GB RTX 3050 Ti and is *not*
obvious from SAM3's own docs, which just show a plain `build_sam3_image_model()`
+ `Sam3Processor(model)` with no precision handling at all:

- Loading the whole model in fp32 leaves ~3.57GB of the ~3.68GB usable VRAM
  taken by weights alone, which OOMs by a small margin (as little as ~80MB)
  during the vision encoder's own forward pass.
- Casting the *whole* model to bf16 fixes that, but then breaks: the
  decoder's FFN block (sam3/model/decoder.py forward_ffn) wraps itself in
  `torch.amp.autocast(enabled=False)`, deliberately forcing true fp32 math
  for numerical stability -- with bf16 weights there, the input (still fp32
  by that point due to residual-add dtype promotion) no longer matches the
  Linear layer's weight dtype and it errors out.
- The fix that actually works: cast only `model.backbone` (the vision +
  text encoder -- the dominant memory cost, and the thing that was actually
  OOMing) to bf16, leave `model.transformer` (the decoder, including that
  forced-fp32 FFN) at its native fp32, and wrap the whole forward pass in
  `torch.autocast(dtype=torch.bfloat16)` so the small fp32 decoder can still
  consume the bf16 backbone features (autocast promotes fp32+bf16 ops
  transparently; only the *stored weight* dtype had to be split this way).

- `Sam3Processor`'s `resolution` must stay at its default 1008 (the
  checkpoint's native size) -- passing any other value makes the model
  interpolate its position embeddings, which hits a separate, unrelated
  dtype bug (BFloat16 vs Float) inside sam3/model/vitdet.py. Don't change it.
"""
import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def load_model() -> torch.nn.Module:
    """Build SAM3, download the (already access-approved) facebook/sam3
    checkpoint if needed, and apply the backbone-only bf16 cast."""
    model = build_sam3_image_model()
    if torch.cuda.is_available():
        model.backbone = model.backbone.to(dtype=torch.bfloat16)
        torch.cuda.empty_cache()
    return model


def make_processor(model: torch.nn.Module, confidence_threshold: float = 0.3) -> Sam3Processor:
    """confidence_threshold default is 0.3, not SAM3's own default of 0.5 --
    empirically, this indoor mock-track scene scores generally lower than
    SAM3's real-photo training distribution (e.g. "white lane line" topped
    out at 0.379, "person" barely cleared 0.5 at 0.5117). 0.3 was the
    highest threshold that still surfaced real dashed-line fragments in
    testing; much lower (e.g. 0.1) lets through enough candidates that the
    full-resolution mask interpolation step OOMs. Re-check with
    score_probe.py on any new session before trusting this value -- scene/
    lighting changes shift the whole score distribution.
    """
    return Sam3Processor(model, confidence_threshold=confidence_threshold)


def autocast_ctx():
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    import contextlib
    return contextlib.nullcontext()


def raw_scores_for_prompt(model, processor, inference_state, prompt: str) -> torch.Tensor:
    """All candidate query scores for `prompt`, BEFORE confidence
    thresholding and BEFORE the (expensive) full-resolution mask
    interpolation -- lets score_probe.py cheaply see the real score
    distribution without risking the OOM that a low confidence_threshold
    triggers in the mask-producing path (Sam3Processor.set_text_prompt).
    """
    with torch.inference_mode():
        text_outputs = model.backbone.forward_text([prompt], device=processor.device)
        inference_state["backbone_out"].update(text_outputs)
        outputs = model.forward_grounding(
            backbone_out=inference_state["backbone_out"],
            find_input=processor.find_stage,
            geometric_prompt=model._get_dummy_prompt(),
            find_target=None,
        )
    out_probs = outputs["pred_logits"].sigmoid()
    presence = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
    return (out_probs * presence).squeeze(-1).float()
