"""Give lingbot-vla a working attention path on GPUs without flash-attn.

lingbot-vla hardcodes ``use_flash_attention_2=True`` at every model
construction site, and its vendored copy of Qwen2.5-VL only registers ``eager``
and ``flash_attention_2``. ROCm has no flash-attn wheel for RDNA3 (gfx1100), and
the eager fallback is unusable for two separate reasons:

  * ``apply_rotary_pos_emb_vision`` calls ``rotate_half`` without defining or
    importing it, so the eager path raises ``NameError`` on the first forward.
  * The eager vision attention builds a ``[1, S, S]`` mask and a full softmax
    over the *packed* patch sequence. At the S~24k this model produces that is
    a 36 GiB allocation, which OOMs a 48 GB card.

So this rewrite:
  1. imports the missing ``rotate_half``;
  2. adds an SDPA vision attention that attends within each ``cu_seqlens``
     segment -- images never attend across segment boundaries, so this needs no
     mask and lets SDPA choose a memory-efficient kernel. It is the same
     softmax attention as the eager path, just not materialised;
  3. points the vision tower at ``sdpa`` and the language towers at ``sdpa``
     (its expert uses transformers' own attention interface) instead of
     flash_attention_2.

Nothing here is ROCm-specific: it is equally correct on a CUDA box that simply
has no flash-attn built. Pass --require-flash-attn to skip the rewrite when
flash-attn is present and preferred.

Usage: python patch_lingbotvla_attn.py <path to lingbot-vla checkout>
"""

import argparse
import io
import os
import sys

VISION_SDPA_CLASS = '''class Qwen2_5_VLVisionSdpaAttention(nn.Module):
    """SDPA counterpart of Qwen2_5_VLVisionAttention.

    Patches never attend across cu_seqlens boundaries, so attention runs per
    segment. That avoids the eager path's [1, S, S] mask and full softmax,
    which is tens of GiB once the packed sequence reaches ~24k tokens.
    """

    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos, sin = emb.cos(), emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        bounds = cu_seqlens.tolist()
        outputs = []
        for start, end in zip(bounds[:-1], bounds[1:]):
            if end <= start:
                continue
            qi = q[start:end].transpose(0, 1).unsqueeze(0)
            ki = k[start:end].transpose(0, 1).unsqueeze(0)
            vi = v[start:end].transpose(0, 1).unsqueeze(0)
            oi = nn.functional.scaled_dot_product_attention(qi, ki, vi)
            outputs.append(oi.squeeze(0).transpose(0, 1))
        attn_output = torch.cat(outputs, dim=0).reshape(seq_length, -1)
        return self.proj(attn_output)


'''

OLD_VISION_REGISTRY = """QWEN2_5_VL_VISION_ATTENTION_CLASSES = {
    "eager": Qwen2_5_VLVisionAttention,
    "flash_attention_2": Qwen2_5_VLVisionFlashAttention2,
}"""

NEW_VISION_REGISTRY = """QWEN2_5_VL_VISION_ATTENTION_CLASSES = {
    "eager": Qwen2_5_VLVisionAttention,
    "flash_attention_2": Qwen2_5_VLVisionFlashAttention2,
    "sdpa": Qwen2_5_VLVisionSdpaAttention,
}"""

ROTATE_HALF_IMPORT = (
    "from transformers.models.qwen2_vl.modeling_qwen2_vl import rotate_half\n\n\n"
)


def patch_qwenvl(path: str) -> None:
    src = io.open(path, encoding="utf-8").read()
    original = src

    if "def rotate_half" not in src and "import rotate_half" not in src:
        anchor = "def apply_rotary_pos_emb_vision("
        if anchor not in src:
            raise SystemExit(
                f"[patch_lingbotvla_attn] {path}: apply_rotary_pos_emb_vision not "
                "found; lingbot-vla upstream has changed."
            )
        src = src.replace(anchor, ROTATE_HALF_IMPORT + anchor, 1)

    if "Qwen2_5_VLVisionSdpaAttention" not in src:
        anchor = "class Qwen2_5_VLVisionFlashAttention2(nn.Module):"
        if anchor not in src:
            raise SystemExit(
                f"[patch_lingbotvla_attn] {path}: vision attention classes not "
                "found; lingbot-vla upstream has changed."
            )
        src = src.replace(anchor, VISION_SDPA_CLASS + anchor, 1)

        if OLD_VISION_REGISTRY not in src:
            raise SystemExit(
                f"[patch_lingbotvla_attn] {path}: vision attention registry not "
                "found; lingbot-vla upstream has changed."
            )
        src = src.replace(OLD_VISION_REGISTRY, NEW_VISION_REGISTRY, 1)

    src = src.replace(
        '_from_config(config.vision_config, use_flash_attention_2=True)',
        '_from_config(config.vision_config, attn_implementation="sdpa")',
    )

    if src != original:
        io.open(path, "w", encoding="utf-8").write(src)
        print(f"[patch_lingbotvla_attn] patched {path}")
    else:
        print(f"[patch_lingbotvla_attn] already patched: {path}")


def patch_modeling(path: str) -> None:
    src = io.open(path, encoding="utf-8").read()
    original = src

    # vlm_config drives the vendored *text* decoder, whose registry has only
    # eager and flash_attention_2. Its sequences are short (~prompt length), so
    # eager costs little there; the vision tower is the one that blows up, and
    # it takes sdpa explicitly via its own _from_config call. The expert is
    # built on transformers' attention interface, which has sdpa already.
    src = src.replace(
        "_from_config(vlm_config, use_flash_attention_2=True)",
        '_from_config(vlm_config, attn_implementation="eager")',
    )
    src = src.replace(
        "_from_config(self.config.qwen_expert_config, use_flash_attention_2=True, eval=eval)",
        '_from_config(self.config.qwen_expert_config, attn_implementation="sdpa", eval=eval)',
    )

    if src != original:
        io.open(path, "w", encoding="utf-8").write(src)
        print(f"[patch_lingbotvla_attn] patched {path}")
    else:
        print(f"[patch_lingbotvla_attn] already patched: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="path to the lingbot-vla checkout")
    parser.add_argument(
        "--require-flash-attn",
        action="store_true",
        help="leave the sources alone if flash-attn is importable",
    )
    args = parser.parse_args()

    if args.require_flash_attn:
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            pass
        else:
            print("[patch_lingbotvla_attn] flash-attn present; leaving sources as-is.")
            return 0

    pi0 = os.path.join(args.root, "lingbotvla", "models", "vla", "pi0")
    qwenvl = os.path.join(pi0, "qwenvl_in_vla.py")
    modeling = os.path.join(pi0, "modeling_lingbot_vla.py")
    for path in (qwenvl, modeling):
        if not os.path.isfile(path):
            raise SystemExit(f"[patch_lingbotvla_attn] not found: {path}")

    patch_qwenvl(qwenvl)
    patch_modeling(modeling)
    return 0


if __name__ == "__main__":
    sys.exit(main())
