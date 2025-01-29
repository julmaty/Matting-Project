# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import torch
from torch import nn
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

import sam2
from sam2.lora import LoRALayer, LoRALinearLayer, MLPWithLoRA, AttentionWithLoRA

# Check if the user is running Python from the parent directory of the sam2 repo
# (i.e. the directory where this repo is cloned into) -- this is not supported since
# it could shadow the sam2 package and cause issues.
if os.path.isdir(os.path.join(sam2.__path__[0], "sam2")):
    # If the user has "sam2/sam2" in their path, they are likey importing the repo itself
    # as "sam2" rather than importing the "sam2" python package (i.e. "sam2/sam2" directory).
    # This typically happens because the user is running Python from the parent directory
    # that contains the sam2 repo they cloned.
    raise RuntimeError(
        "You're likely running Python from the parent directory of the sam2 repository "
        "(i.e. the directory where https://github.com/facebookresearch/sam2 is cloned into). "
        "This is not supported since the `sam2` Python package could be shadowed by the "
        "repository name (the repository is also named `sam2` and contains the Python package "
        "in `sam2/sam2`). Please run Python from another directory (e.g. from the repo dir "
        "rather than its parent dir, or from your home directory) after installing SAM 2."
    )


def build_sam2_matting(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    
    transformer_dim = model.sam_mask_decoder.transformer_dim
    num_mask_tokens = model.sam_mask_decoder.num_mask_tokens

    model.sam_mask_decoder.output_hypernetworks_mlps = nn.ModuleList(
                [
                    MLPWithLoRA(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                    for i in range(num_mask_tokens)
                ]
            )
            
    embedding_dim = model.sam_mask_decoder.transformer.embedding_dim
    num_heads = model.sam_mask_decoder.transformer.num_heads
    downsample_rate= 2
    
    model.sam_mask_decoder.transformer.final_attn_token_to_image = AttentionWithLoRA(embedding_dim, num_heads, downsample_rate)
            
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor_matting(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    vos_optimized=False,
    train=False,
    **kwargs,
):
    if train:
        hydra_overrides = [
            "++model._target_=sam2.sam2_video_predictor_train.SAM2VideoPredictorTrain",
        ]
    else:
        hydra_overrides = [
            "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    
    if vos_optimized:
        hydra_overrides = [
            "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictorVOS",
            "++model.compile_image_encoder=True",  # Let sam2_base handle this
        ]

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)

    transformer_dim = model.sam_mask_decoder.transformer_dim
    num_mask_tokens = model.sam_mask_decoder.num_mask_tokens

    model.sam_mask_decoder.output_hypernetworks_mlps = nn.ModuleList(
                [
                    MLPWithLoRA(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                    for i in range(num_mask_tokens)
                ]
            )
            
    embedding_dim = model.sam_mask_decoder.transformer.embedding_dim
    num_heads = model.sam_mask_decoder.transformer.num_heads
    downsample_rate= 2
    
    model.sam_mask_decoder.transformer.final_attn_token_to_image = AttentionWithLoRA(embedding_dim, num_heads, downsample_rate)

    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")
