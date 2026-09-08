# Copyright (c) 2026 codec.cpp contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import torch
from safetensors.torch import load_file

from .base import ModelBase, MmprojModel, gguf

def _materialize_weight_norm(weight_g: np.ndarray, weight_v: np.ndarray) -> np.ndarray:
    g = np.asarray(weight_g, dtype=np.float32)
    v = np.asarray(weight_v, dtype=np.float32)
    axes = tuple(range(1, v.ndim))
    norm = np.linalg.norm(v, axis=axes, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return (v * (g / norm)).astype(np.float32, copy=False)


def _materialize_state_dict(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Bake `parametrizations.weight.original{0,1}` into a single effective weight."""
    out: dict[str, np.ndarray] = {}
    pending: dict[str, dict[str, np.ndarray]] = {}
    for key, value in state.items():
        if key.endswith(".parametrizations.weight.original0"):
            base = key[: -len(".parametrizations.weight.original0")]
            pending.setdefault(base, {})["g"] = value
        elif key.endswith(".parametrizations.weight.original1"):
            base = key[: -len(".parametrizations.weight.original1")]
            pending.setdefault(base, {})["v"] = value
        else:
            out[key] = value
    for base, gv in pending.items():
        if "g" not in gv or "v" not in gv:
            raise RuntimeError(f"incomplete weight_norm parametrization at {base}")
        out[base + ".weight"] = _materialize_weight_norm(gv["g"], gv["v"])
    return out


_S3G_FLOW_NUM_DOWN_BLOCKS = 6
_S3G_FLOW_NUM_UP_BLOCKS = 4
_S3G_CFM_NUM_DOWN_BLOCKS = 1
_S3G_CFM_NUM_MID_BLOCKS = 12
_S3G_CFM_NUM_UP_BLOCKS = 1
_S3G_CFM_TRANSFORMERS_PER_BLOCK = 4
_S3G_HIFT_F0_NUM_LAYERS = 5
_S3G_HIFT_NUM_UPS = 3


def _take(state: dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in state:
        raise KeyError(f"missing S3G tensor: {key}")
    return state.pop(key)


def _emit_flow_attn_block(
    out: list[tuple[str, np.ndarray]],
    state: dict[str, np.ndarray],
    src_prefix: str,
    dst_prefix: str,
) -> None:
    a = src_prefix + ".self_attn"
    f = src_prefix + ".feed_forward"
    out.append((dst_prefix + ".norm_mha.w", _take(state, src_prefix + ".norm_mha.weight")))
    out.append((dst_prefix + ".norm_mha.b", _take(state, src_prefix + ".norm_mha.bias")))
    out.append((dst_prefix + ".norm_ff.w",  _take(state, src_prefix + ".norm_ff.weight")))
    out.append((dst_prefix + ".norm_ff.b",  _take(state, src_prefix + ".norm_ff.bias")))
    out.append((dst_prefix + ".attn.q.w",   _take(state, a + ".linear_q.weight")))
    out.append((dst_prefix + ".attn.q.b",   _take(state, a + ".linear_q.bias")))
    out.append((dst_prefix + ".attn.k.w",   _take(state, a + ".linear_k.weight")))
    out.append((dst_prefix + ".attn.k.b",   _take(state, a + ".linear_k.bias")))
    out.append((dst_prefix + ".attn.v.w",   _take(state, a + ".linear_v.weight")))
    out.append((dst_prefix + ".attn.v.b",   _take(state, a + ".linear_v.bias")))
    out.append((dst_prefix + ".attn.o.w",   _take(state, a + ".linear_out.weight")))
    out.append((dst_prefix + ".attn.o.b",   _take(state, a + ".linear_out.bias")))
    out.append((dst_prefix + ".attn.pos.w", _take(state, a + ".linear_pos.weight")))
    out.append((dst_prefix + ".attn.pbu",   _take(state, a + ".pos_bias_u")))
    out.append((dst_prefix + ".attn.pbv",   _take(state, a + ".pos_bias_v")))
    out.append((dst_prefix + ".ff.w1.w",    _take(state, f + ".w_1.weight")))
    out.append((dst_prefix + ".ff.w1.b",    _take(state, f + ".w_1.bias")))
    out.append((dst_prefix + ".ff.w2.w",    _take(state, f + ".w_2.weight")))
    out.append((dst_prefix + ".ff.w2.b",    _take(state, f + ".w_2.bias")))


def _emit_cfm_resnet(
    out: list[tuple[str, np.ndarray]],
    state: dict[str, np.ndarray],
    src_prefix: str,
    dst_prefix: str,
) -> None:
    out.append((dst_prefix + ".b1.cv.w", _take(state, src_prefix + ".block1.block.0.weight")))
    out.append((dst_prefix + ".b1.cv.b", _take(state, src_prefix + ".block1.block.0.bias")))
    out.append((dst_prefix + ".b1.ln.w", _take(state, src_prefix + ".block1.block.2.weight")))
    out.append((dst_prefix + ".b1.ln.b", _take(state, src_prefix + ".block1.block.2.bias")))
    out.append((dst_prefix + ".b2.cv.w", _take(state, src_prefix + ".block2.block.0.weight")))
    out.append((dst_prefix + ".b2.cv.b", _take(state, src_prefix + ".block2.block.0.bias")))
    out.append((dst_prefix + ".b2.ln.w", _take(state, src_prefix + ".block2.block.2.weight")))
    out.append((dst_prefix + ".b2.ln.b", _take(state, src_prefix + ".block2.block.2.bias")))
    out.append((dst_prefix + ".mlp.w",   _take(state, src_prefix + ".mlp.1.weight")))
    out.append((dst_prefix + ".mlp.b",   _take(state, src_prefix + ".mlp.1.bias")))
    out.append((dst_prefix + ".res.w",   _take(state, src_prefix + ".res_conv.weight")))
    out.append((dst_prefix + ".res.b",   _take(state, src_prefix + ".res_conv.bias")))


def _emit_cfm_transformer(
    out: list[tuple[str, np.ndarray]],
    state: dict[str, np.ndarray],
    src_prefix: str,
    dst_prefix: str,
) -> None:
    a = src_prefix + ".attn1"
    out.append((dst_prefix + ".norm1.w", _take(state, src_prefix + ".norm1.weight")))
    out.append((dst_prefix + ".norm1.b", _take(state, src_prefix + ".norm1.bias")))
    out.append((dst_prefix + ".norm3.w", _take(state, src_prefix + ".norm3.weight")))
    out.append((dst_prefix + ".norm3.b", _take(state, src_prefix + ".norm3.bias")))
    out.append((dst_prefix + ".attn.q.w", _take(state, a + ".to_q.weight")))
    out.append((dst_prefix + ".attn.k.w", _take(state, a + ".to_k.weight")))
    out.append((dst_prefix + ".attn.v.w", _take(state, a + ".to_v.weight")))
    out.append((dst_prefix + ".attn.o.w", _take(state, a + ".to_out.0.weight")))
    out.append((dst_prefix + ".attn.o.b", _take(state, a + ".to_out.0.bias")))
    out.append((dst_prefix + ".ff.w1.w",  _take(state, src_prefix + ".ff.net.0.proj.weight")))
    out.append((dst_prefix + ".ff.w1.b",  _take(state, src_prefix + ".ff.net.0.proj.bias")))
    out.append((dst_prefix + ".ff.w2.w",  _take(state, src_prefix + ".ff.net.2.weight")))
    out.append((dst_prefix + ".ff.w2.b",  _take(state, src_prefix + ".ff.net.2.bias")))


def _emit_resblock(
    out: list[tuple[str, np.ndarray]],
    state: dict[str, np.ndarray],
    src_prefix: str,
    dst_prefix: str,
) -> None:
    for k in range(3):
        out.append((dst_prefix + f".cv1.{k}.w", _take(state, src_prefix + f".convs1.{k}.weight")))
        out.append((dst_prefix + f".cv1.{k}.b", _take(state, src_prefix + f".convs1.{k}.bias")))
        out.append((dst_prefix + f".cv2.{k}.w", _take(state, src_prefix + f".convs2.{k}.weight")))
        out.append((dst_prefix + f".cv2.{k}.b", _take(state, src_prefix + f".convs2.{k}.bias")))
        out.append((dst_prefix + f".a1.{k}",    _take(state, src_prefix + f".activations1.{k}.alpha")))
        out.append((dst_prefix + f".a2.{k}",    _take(state, src_prefix + f".activations2.{k}.alpha")))


def _build_s3g_tensor_map(
    state: dict[str, np.ndarray],
    *,
    meanflow: bool,
) -> list[tuple[str, np.ndarray]]:
    state = dict(state)  # take pop ownership
    out: list[tuple[str, np.ndarray]] = []

    for key in list(state.keys()):
        if key.startswith("tokenizer.") or key.startswith("speaker_encoder."):
            del state[key]

    out.append(("s3g.flow.input_emb.w", _take(state, "flow.input_embedding.weight")))
    out.append(("s3g.flow.spk_aff.w",   _take(state, "flow.spk_embed_affine_layer.weight")))
    out.append(("s3g.flow.spk_aff.b",   _take(state, "flow.spk_embed_affine_layer.bias")))
    out.append(("s3g.flow.proj.w",      _take(state, "flow.encoder_proj.weight")))
    out.append(("s3g.flow.proj.b",      _take(state, "flow.encoder_proj.bias")))

    out.append(("s3g.flow.enc.embed.lin.w", _take(state, "flow.encoder.embed.out.0.weight")))
    out.append(("s3g.flow.enc.embed.lin.b", _take(state, "flow.encoder.embed.out.0.bias")))
    out.append(("s3g.flow.enc.embed.ln.w",  _take(state, "flow.encoder.embed.out.1.weight")))
    out.append(("s3g.flow.enc.embed.ln.b",  _take(state, "flow.encoder.embed.out.1.bias")))
    out.append(("s3g.flow.enc.up_embed.lin.w", _take(state, "flow.encoder.up_embed.out.0.weight")))
    out.append(("s3g.flow.enc.up_embed.lin.b", _take(state, "flow.encoder.up_embed.out.0.bias")))
    out.append(("s3g.flow.enc.up_embed.ln.w",  _take(state, "flow.encoder.up_embed.out.1.weight")))
    out.append(("s3g.flow.enc.up_embed.ln.b",  _take(state, "flow.encoder.up_embed.out.1.bias")))
    out.append(("s3g.flow.enc.after_norm.w", _take(state, "flow.encoder.after_norm.weight")))
    out.append(("s3g.flow.enc.after_norm.b", _take(state, "flow.encoder.after_norm.bias")))
    out.append(("s3g.flow.enc.pre.cv1.w", _take(state, "flow.encoder.pre_lookahead_layer.conv1.weight")))
    out.append(("s3g.flow.enc.pre.cv1.b", _take(state, "flow.encoder.pre_lookahead_layer.conv1.bias")))
    out.append(("s3g.flow.enc.pre.cv2.w", _take(state, "flow.encoder.pre_lookahead_layer.conv2.weight")))
    out.append(("s3g.flow.enc.pre.cv2.b", _take(state, "flow.encoder.pre_lookahead_layer.conv2.bias")))
    out.append(("s3g.flow.enc.up.w", _take(state, "flow.encoder.up_layer.conv.weight")))
    out.append(("s3g.flow.enc.up.b", _take(state, "flow.encoder.up_layer.conv.bias")))
    for li in range(_S3G_FLOW_NUM_DOWN_BLOCKS):
        _emit_flow_attn_block(out, state, f"flow.encoder.encoders.{li}", f"s3g.flow.enc.blk.{li}")
    for li in range(_S3G_FLOW_NUM_UP_BLOCKS):
        _emit_flow_attn_block(out, state, f"flow.encoder.up_encoders.{li}", f"s3g.flow.enc.up_blk.{li}")

    out.append(("s3g.cfm.t.l1.w", _take(state, "flow.decoder.estimator.time_mlp.linear_1.weight")))
    out.append(("s3g.cfm.t.l1.b", _take(state, "flow.decoder.estimator.time_mlp.linear_1.bias")))
    out.append(("s3g.cfm.t.l2.w", _take(state, "flow.decoder.estimator.time_mlp.linear_2.weight")))
    out.append(("s3g.cfm.t.l2.b", _take(state, "flow.decoder.estimator.time_mlp.linear_2.bias")))
    if meanflow:
        out.append(("s3g.cfm.t_mix.w", _take(state, "flow.decoder.estimator.time_embed_mixer.weight")))

    def _emit_cfm_section(group: str, n_blocks: int, has_trailing: bool) -> None:
        for bi in range(n_blocks):
            src_b = f"flow.decoder.estimator.{group}.{bi}"
            dst_b = f"s3g.cfm.{ {'down_blocks':'dn','mid_blocks':'md','up_blocks':'up'}[group] }.{bi}"
            _emit_cfm_resnet(out, state, src_b + ".0", dst_b + ".r")
            for ti in range(_S3G_CFM_TRANSFORMERS_PER_BLOCK):
                _emit_cfm_transformer(out, state, f"{src_b}.1.{ti}", f"{dst_b}.t.{ti}")
            if has_trailing:
                out.append((dst_b + ".x.w", _take(state, src_b + ".2.weight")))
                out.append((dst_b + ".x.b", _take(state, src_b + ".2.bias")))

    _emit_cfm_section("down_blocks", _S3G_CFM_NUM_DOWN_BLOCKS, has_trailing=True)
    _emit_cfm_section("mid_blocks",  _S3G_CFM_NUM_MID_BLOCKS,  has_trailing=False)
    _emit_cfm_section("up_blocks",   _S3G_CFM_NUM_UP_BLOCKS,   has_trailing=True)

    out.append(("s3g.cfm.final.cv.w", _take(state, "flow.decoder.estimator.final_block.block.0.weight")))
    out.append(("s3g.cfm.final.cv.b", _take(state, "flow.decoder.estimator.final_block.block.0.bias")))
    out.append(("s3g.cfm.final.ln.w", _take(state, "flow.decoder.estimator.final_block.block.2.weight")))
    out.append(("s3g.cfm.final.ln.b", _take(state, "flow.decoder.estimator.final_block.block.2.bias")))
    out.append(("s3g.cfm.proj.w",     _take(state, "flow.decoder.estimator.final_proj.weight")))
    out.append(("s3g.cfm.proj.b",     _take(state, "flow.decoder.estimator.final_proj.bias")))

    for li in range(_S3G_HIFT_F0_NUM_LAYERS):
        src_idx = li * 2
        out.append((f"s3g.hift.f0.cn.{li}.w", _take(state, f"mel2wav.f0_predictor.condnet.{src_idx}.weight")))
        out.append((f"s3g.hift.f0.cn.{li}.b", _take(state, f"mel2wav.f0_predictor.condnet.{src_idx}.bias")))
    out.append(("s3g.hift.f0.cls.w", _take(state, "mel2wav.f0_predictor.classifier.weight")))
    out.append(("s3g.hift.f0.cls.b", _take(state, "mel2wav.f0_predictor.classifier.bias")))
    out.append(("s3g.hift.src.lin.w", _take(state, "mel2wav.m_source.l_linear.weight")))
    out.append(("s3g.hift.src.lin.b", _take(state, "mel2wav.m_source.l_linear.bias")))
    out.append(("s3g.hift.conv_pre.w", _take(state, "mel2wav.conv_pre.weight")))
    out.append(("s3g.hift.conv_pre.b", _take(state, "mel2wav.conv_pre.bias")))
    out.append(("s3g.hift.conv_post.w", _take(state, "mel2wav.conv_post.weight")))
    out.append(("s3g.hift.conv_post.b", _take(state, "mel2wav.conv_post.bias")))
    for ui in range(_S3G_HIFT_NUM_UPS):
        out.append((f"s3g.hift.up.{ui}.w", _take(state, f"mel2wav.ups.{ui}.weight")))
        out.append((f"s3g.hift.up.{ui}.b", _take(state, f"mel2wav.ups.{ui}.bias")))
        out.append((f"s3g.hift.src_dn.{ui}.w", _take(state, f"mel2wav.source_downs.{ui}.weight")))
        out.append((f"s3g.hift.src_dn.{ui}.b", _take(state, f"mel2wav.source_downs.{ui}.bias")))
        _emit_resblock(out, state, f"mel2wav.source_resblocks.{ui}", f"s3g.hift.src_rb.{ui}")
        for ki in range(3):  # 3 parallel resblocks per upsample stage
            _emit_resblock(out, state, f"mel2wav.resblocks.{ui * 3 + ki}", f"s3g.hift.rb.{ui * 3 + ki}")

    leftovers = sorted(state.keys())
    if leftovers:
        raise RuntimeError(f"unmapped S3G tensors after conversion: {leftovers[:20]} (+{len(leftovers)-20} more)" if len(leftovers) > 20 else f"unmapped S3G tensors after conversion: {leftovers}")
    return out


@ModelBase.register("ChatterboxT3Model", "ChatterboxTurboT3Model")
class ChatterboxMmprojModel(MmprojModel):
    has_vision_encoder = False
    has_audio_encoder = False

    def set_gguf_parameters(self):
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_clip_has_gen_audio_encoder(True)
        self.gguf_writer.add_clip_gen_audio_projector_type(gguf.VisionProjectorType.CHATTERBOX_GEN)
        self.gguf_writer.add_gen_audio_projection_dim(1024)
        self.gguf_writer.add_gen_audio_embedding_length(512)
        self.gguf_writer.add_gen_audio_feed_forward_length(2048)
        self.gguf_writer.add_gen_audio_block_count(10)
        self.gguf_writer.add_gen_audio_head_count(8)
        self.gguf_writer.add_gen_audio_attention_layernorm_eps(1e-5)
        self.gguf_writer.add_uint32("clip.chatterbox.text_vocab_size", self.hparams["chatterbox_text_vocab"])
        self.gguf_writer.add_string("clip.gen.audio.model_variant", self.hparams["chatterbox_checkpoint"])
        self.gguf_writer.add_bool("clip.chatterbox.meanflow", self.hparams["chatterbox_text_vocab"] == 50276)

    def get_tensors(self):
        meanflow = self.hparams["chatterbox_text_vocab"] == 50276
        file = self.dir_model / ("s3gen_meanflow.safetensors" if meanflow else "s3gen.safetensors")
        if file.is_file():
            state = load_file(file)
        else:
            state = torch.load(self.dir_model / "s3gen.pt", map_location="cpu", weights_only=True)
        state = {k: v.float().numpy() for k, v in state.items()}
        # Reference conditioning: S3 tokenizer, CAMPPlus and the T3 voice encoder.
        for name, value in state.items():
            if name.startswith("tokenizer."):
                short = name.removeprefix("tokenizer.").replace("encoder.blocks.", "enc.blk.")
                short = short.replace("encoder.", "enc.").replace("quantizer._codebook.project_down", "proj")
                yield "s3g.ref.tok." + short, torch.from_numpy(np.ascontiguousarray(value))
        camp = {k.removeprefix("speaker_encoder."): v for k, v in state.items() if k.startswith("speaker_encoder.")}
        def camp_name(name):
            return (name.replace("xvector.", "x.").replace("out_nonlinear.batchnorm", "out_n")
                    .replace("nonlinear1.batchnorm", "n1").replace("nonlinear2.batchnorm", "n2")
                    .replace("nonlinear.batchnorm", "n").replace("cam_layer", "cam").replace("tdnnd", "d"))
        for name in list(camp):
            if not name.endswith(".running_mean"):
                continue
            prefix = name.removesuffix(".running_mean")
            mean = camp.pop(name)
            variance = camp.pop(prefix + ".running_var")
            gamma = camp.pop(prefix + ".weight", np.ones_like(mean))
            beta = camp.pop(prefix + ".bias", np.zeros_like(mean))
            scale = gamma / np.sqrt(variance + 1e-5)
            camp[prefix + ".weight"] = scale
            camp[prefix + ".bias"] = beta - mean * scale
            camp.pop(prefix + ".num_batches_tracked", None)
        for name, value in camp.items():
            yield "s3g.ref.camp." + camp_name(name), torch.from_numpy(np.ascontiguousarray(value))
        ve_file = self.dir_model / "ve.safetensors"
        ve = load_file(ve_file) if ve_file.is_file() else torch.load(self.dir_model / "ve.pt", weights_only=True, map_location="cpu")
        for i in range(3):
            yield f"s3g.ref.ve.blk.{i}.input.weight", ve[f"lstm.weight_ih_l{i}"]
            yield f"s3g.ref.ve.blk.{i}.recurrent.weight", ve[f"lstm.weight_hh_l{i}"]
            yield f"s3g.ref.ve.blk.{i}.bias", ve[f"lstm.bias_ih_l{i}"] + ve[f"lstm.bias_hh_l{i}"]
        yield "s3g.ref.ve.proj.weight", ve["proj.weight"]
        yield "s3g.ref.ve.proj.bias", ve["proj.bias"]

        for name, value in _build_s3g_tensor_map(_materialize_state_dict(state), meanflow=meanflow):
            yield name, torch.from_numpy(np.ascontiguousarray(value))
        conds = torch.load(self.dir_model / "conds.pt", map_location="cpu", weights_only=True)["gen"]
        for key in ("prompt_token", "prompt_feat", "embedding"):
            yield "s3g.cond." + key, conds[key].float().squeeze(0)

        # The T3 wrapper is evaluated by MTMD; only its transformer lives in the backbone GGUF.
        t3 = load_file(self.dir_model / self.hparams["chatterbox_checkpoint"])
        for name, value in t3.items():
            if name.startswith(("cond_enc.", "text_pos_emb.", "speech_pos_emb.")) or name in ("text_emb.weight", "speech_emb.weight"):
                yield "s3g.t3." + name, value.squeeze(0) if name.endswith("pre_attention_query") else value
        t3_cond = torch.load(self.dir_model / "conds.pt", map_location="cpu", weights_only=True)["t3"]
        for key in ("speaker_emb", "cond_prompt_speech_tokens", "emotion_adv"):
            yield "s3g.t3.cond." + key, t3_cond[key].float().reshape(-1)

        # HiFT uses a 16-point periodic Hann STFT and matching overlap-add synthesis.
        n = np.arange(16, dtype=np.float64)
        k = np.arange(9, dtype=np.float64)
        hann = 0.5 - 0.5 * np.cos(2 * np.pi * n / 16)
        angle = 2 * np.pi * k[:, None] * n[None, :] / 16
        re = np.cos(angle) * hann
        im = -np.sin(angle) * hann
        factors = np.full(9, 2.0)
        factors[[0, -1]] = 1.0
        constants = {
            "stft_basis_re_k": re[:, None, :],
            "stft_basis_im_k": im[:, None, :],
            "istft_basis_re": (re * factors[:, None]).T,
            "istft_basis_im": (-im * factors[:, None]).T,
            "hann": hann,
            "ola_w": np.eye(16)[:, None, :],
        }
        for name, data in constants.items():
            yield "s3g.hift." + name, torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))

    def modify_tensors(self, data_torch, name, bid):
        name = name.replace("s3g.", "a.gen.wav.", 1)
        if name.endswith(".w"):
            name = name[:-2] + ".weight"
        elif name.endswith(".b"):
            name = name[:-2] + ".bias"
        yield name, data_torch

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if name.endswith("._mel_filters"):
            return gguf.GGMLQuantizationType.F32
        if any(name.endswith(suffix) for suffix in ("_basis_re_k", "_basis_im_k", "_basis_re", "_basis_im", ".hann", ".ola_w")):
            return gguf.GGMLQuantizationType.F32
        if n_dims == 3:
            return gguf.GGMLQuantizationType.F16
        if name.startswith(("s3g.cond.", "s3g.t3.cond.")) or ".a1." in name or ".a2." in name:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)
