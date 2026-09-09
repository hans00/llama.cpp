# Copyright (c) 2026 codec.cpp contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import torch

from .base import ModelBase, MmprojModel, gguf


@ModelBase.register("SnacModel")
class SnacModel(MmprojModel):
    has_vision_encoder = False
    has_audio_encoder = False

    def __init__(self, *args, **kwargs):
        hp = json.loads((Path(args[0]) / "config.json").read_text())
        hp["num_hidden_layers"] = 4
        kwargs["hparams"] = {"hidden_size": 3072, "audio_config": hp, "architectures": ["SnacModel"]}
        super().__init__(*args, **kwargs)

    def set_gguf_parameters(self):
        required = {"sampling_rate": 24000, "encoder_dim": 48, "decoder_dim": 1024,
                    "encoder_rates": [2, 4, 8, 8], "decoder_rates": [8, 8, 4, 2],
                    "vq_strides": [4, 2, 1], "codebook_size": 4096, "codebook_dim": 8,
                    "attn_window_size": None, "depthwise": True, "noise": True}
        if any(self.hparams.get(k) != v for k, v in required.items()):
            raise ValueError("Only the SNAC 24 kHz architecture used by Orpheus is supported")
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_clip_has_gen_audio_encoder(True)
        self.gguf_writer.add_clip_gen_audio_projector_type(gguf.VisionProjectorType.SNAC)
        self.gguf_writer.add_gen_audio_projection_dim(3072)
        self.gguf_writer.add_gen_audio_embedding_length(768)
        self.gguf_writer.add_gen_audio_feed_forward_length(1024)
        self.gguf_writer.add_gen_audio_block_count(4)
        self.gguf_writer.add_gen_audio_head_count(1)
        self.gguf_writer.add_gen_audio_attention_layernorm_eps(1e-5)
        self.gguf_writer.add_string("clip.gen.audio.model_variant", "snac_24khz")

    def get_tensors(self):
        sd = torch.load(self.dir_model / "pytorch_model.bin", map_location="cpu", weights_only=True)
        for name, value in sd.items():
            if name.endswith(".parametrizations.weight.original0"):
                continue
            if name.endswith(".parametrizations.weight.original1"):
                prefix = name.removesuffix(".parametrizations.weight.original1")
                g = sd[prefix + ".parametrizations.weight.original0"]
                value = value.float()
                value = value * (g.float() / value.norm(dim=tuple(range(1, value.ndim)), keepdim=True))
                name = prefix + ".weight"
            if name.endswith(".alpha") or name.endswith(".bias"):
                value = value.reshape(-1, 1)
            elif name.endswith(".weight") and value.ndim == 3 and value.shape[-1] == 1:
                value = value.squeeze(-1)
            if name.startswith("encoder.") and name.endswith(".weight") and value.ndim == 3 and (value.shape[1] != 1 or name == "encoder.block.0.weight"):
                value = value.flatten(1)
            yield name, value
            if name.endswith("codebook.weight"):
                yield name.replace("codebook.weight", "codebook_norm.weight"), torch.nn.functional.normalize(value.float(), dim=1)

    def modify_tensors(self, data_torch, name, bid):
        name = name.replace("encoder.block.", "enc.").replace("decoder.model.", "dec.").replace("quantizer.quantizers.", "q.")
        name = name.replace(".block.", ".")
        yield "a.gen.wav." + name, data_torch

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if name.startswith(("encoder.", "quantizer.")):
            return gguf.GGMLQuantizationType.F32
        if name.endswith((".alpha", ".bias", "codebook.weight", "codebook_norm.weight")):
            return gguf.GGMLQuantizationType.F32
        if n_dims >= 2:
            return gguf.GGMLQuantizationType.F32 if self.ftype == gguf.LlamaFileType.ALL_F32 else gguf.GGMLQuantizationType.F16
        return super().tensor_force_quant(name, new_name, bid, n_dims)
