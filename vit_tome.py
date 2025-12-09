

from typing import Tuple

import torch
from dynamic_vit_viz import vit_register_dynamic_viz
from timm.models.vision_transformer import Attention, Block
from dynamic_vit_viz import Block, Layer_scale_init_Block, Block_paralx2, Attention
from merge import bipartite_soft_matching, merge_source, merge_wavg
from utils import parse_r


class ToMeBlock(Block):

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        # DEBUG 1: Kích thước token TRƯỚC Attention/Gộp
        T_start = x.shape[1]
        print(f"--- ToMeBlock Debug Start ---")
        print(f"Input Token Count (T_start): {T_start}")

        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(x_attn)

        r = self._tome_info["r"].pop(0)
        print(f"R (tokens to merge) for this block: {r}")
        if r > 0:
            P = (1 if self._tome_info["class_token"] else 0) + self._tome_info["num_register_tokens"]
            print(f"Protected Tokens (P, CLS+REG): {P}")
            print(f"Patch Tokens count: {T_start - P}")

            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["num_register_tokens"]
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])
            # DEBUG 3: Kích thước token SAU Gộp
            T_end = x.shape[1]
            print(f"Output Token Count (T_end): {T_end}")
            print(f"Tokens Reduced: {T_start - T_end}")
            print(f"Expected T_end: {T_start - r}") # Lý tưởng (nếu r không bị giới hạn)

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        print(f"--- ToMeBlock Debug End ---\n")
        return x


class ToMeAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return x, k.mean(1)


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.blocks), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None
            return super().forward(*args, **kwdargs)

    return ToMeVisionTransformer


def apply_patch(
    model: vit_register_dynamic_viz, trace_source: bool = False, prop_attn: bool = True
):
 
    ToMeVisionTransformer = make_tome_class(model.__class__)

    num_register_tokens = 0
    if hasattr(model, "register_tokens") and model.register_tokens is not None:
        num_register_tokens = model.register_tokens.shape[1]
    
   # 2. KIỂM TRA SỰ HIỆN DIỆN VÀ TÍNH TOÁN CHỈ SỐ
    has_cls_token = model.cls_token is not None
    cls_idx = 0 if has_cls_token else -1 # CLS token luôn ở index 0 nếu có
    
    # Register tokens bắt đầu ngay sau CLS token (nếu có)
    reg_start_idx = 1 if has_cls_token else 0
    reg_indices = list(range(reg_start_idx, reg_start_idx + num_register_tokens))


    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "num_register_tokens": num_register_tokens,
        "register_token": num_register_tokens > 0
    }

    if hasattr(model, "register_token") and model.register_tokens is not None:
        model._tome_info["register_token"] = True

    custom_blocks_to_patch = (Block, Layer_scale_init_Block, Block_paralx2) # <-- Dùng tuple chứa TẤT CẢ các lớp Block
    block_count = 0
    attention_count = 0
    for module in model.modules():
        if isinstance(module, custom_blocks_to_patch):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info

        elif isinstance(module, Attention): 
            module.__class__ = ToMeAttention
    

 
