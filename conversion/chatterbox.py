from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from .base import ModelBase, SentencePieceTokenTypes, gguf
from .gpt2 import GPT2Model
from .llama import LlamaModel


_CHECKPOINTS = ("t3_cfg.safetensors", "t3_23lang.safetensors", "t3_mtl23ls_v2.safetensors",
                "t3_mtl23ls_v3.safetensors", "t3_turbo_v1.safetensors")


def _checkpoint(path: Path) -> Path:
    files = [path / name for name in _CHECKPOINTS if (path / name).is_file()]
    if len(files) != 1:
        raise ValueError("Place exactly one Chatterbox T3 checkpoint in the conversion directory")
    return files[0]


@ModelBase.register_hparams_loader(lambda path: any((path / name).is_file() for name in _CHECKPOINTS))
def _load_hparams(path: Path) -> dict:
    file = _checkpoint(path)
    with safe_open(file, framework="pt") as f:
        shapes = {name: f.get_slice(name).get_shape() for name in f.keys()}
    n_text, hidden = shapes["text_emb.weight"]
    n_speech, speech_hidden = shapes["speech_emb.weight"]
    turbo = "tfmr.wpe.weight" in shapes
    if hidden != 1024 or speech_hidden != hidden or (n_text, n_speech) not in ((704, 8194), (2454, 8194), (50276, 6563)):
        raise ValueError("Unsupported Chatterbox T3 dimensions")
    return {
        "architectures": ["ChatterboxTurboT3Model" if turbo else "ChatterboxT3Model"],
        "model_type": "chatterbox", "chatterbox_checkpoint": file.name,
        "chatterbox_text_vocab": n_text, "chatterbox_speech_vocab": n_speech,
        "hidden_size": hidden, "n_embd": hidden, "intermediate_size": 4096,
        "num_hidden_layers": 24 if turbo else 30, "num_attention_heads": 16,
        "num_key_value_heads": 16, "n_head": 16, "head_dim": 64,
        "max_position_embeddings": 8196 if turbo else 131072, "n_ctx": 8196,
        "rms_norm_eps": 1e-5, "layer_norm_epsilon": 1e-5,
        "vocab_size": n_text + n_speech, "rope_theta": 500000.0,
        "rope_scaling": {} if turbo else {"rope_type": "llama3", "factor": 8.0,
            "high_freq_factor": 4.0, "low_freq_factor": 1.0, "original_max_position_embeddings": 8192},
    }


class ChatterboxT3Mixin:
    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_string(f"{self.gguf_writer.arch}.tts.model", "chatterbox")
        self.gguf_writer.add_string(f"{self.gguf_writer.arch}.tts.checkpoint", self.hparams["chatterbox_checkpoint"])
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.tts.text_vocab_size", self.hparams["chatterbox_text_vocab"])
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.tts.speech_vocab_size", self.hparams["chatterbox_speech_vocab"])

        turbo = self.hparams["chatterbox_text_vocab"] == 50276
        self.gguf_writer.add_sampling_sequence("temperature;top_k;top_p;penalties" if turbo else "penalties;temperature;min_p;top_p")
        self.gguf_writer.add_sampling_temp(0.8)
        self.gguf_writer.add_sampling_top_k(1000 if turbo else 0)
        self.gguf_writer.add_sampling_top_p(0.95 if turbo else 1.0)
        self.gguf_writer.add_sampling_min_p(0.0 if turbo else 0.05)
        self.gguf_writer.add_sampling_penalty_repeat(1.2)
        self.gguf_writer.add_sampling_penalty_last_n(4096)

        if self.hparams["chatterbox_text_vocab"] == 2454:
            import unicodedata as ud
            from collections import defaultdict
            substitutions, combining, case = {}, {}, {}
            for cp in range(0x110000):
                char = chr(cp)
                normalized = ud.normalize("NFKD", char.lower())
                if char != normalized:
                    substitutions[char] = normalized
                if ud.combining(char):
                    combining[char] = ud.combining(char)
                flags = int(char.lower() != char.upper())
                flags |= 2 * int(ud.category(char) in ("Mn", "Me", "Cf", "Lm", "Sk") or char in "'’·:")
                if flags:
                    case[char] = flags
            entries = json.loads((self.dir_model / "Cangjie5_TC.json").read_text())
            by_code, by_char = defaultdict(list), {}
            for entry in entries:
                char, code = entry.split("\t")[:2]
                by_code[code].append(char)
                by_char[char] = code
            cangjie = {}
            for char, code in by_char.items():
                if len(char) == 1 and ud.category(char) == "Lo":
                    index = by_code[code].index(char)
                    code += str(index) if index else ""
                    cangjie[char] = "".join(f"[cj_{c}]" for c in code) + "[cj_.]"
            rules = {"map": substitutions, "ccc": combining, "case": case, "cangjie": cangjie}
            self.gguf_writer.add_string(f"{self.gguf_writer.arch}.tts.normalization", json.dumps(rules, ensure_ascii=False, separators=(",", ":")))

    def set_vocab(self):
        n_text = self.hparams["chatterbox_text_vocab"]
        turbo = n_text == 50276
        if turbo:
            vocab = json.loads((self.dir_model / "vocab.json").read_text())
            config = json.loads((self.dir_model / "tokenizer_config.json").read_text())
            added = config["added_tokens_decoder"]
            for index, token in added.items():
                vocab[token["content"]] = int(index)
            added_ids = {int(index) for index in added}
            specials = {int(index) for index, token in added.items() if token.get("special")}
            merges = [line for line in (self.dir_model / "merges.txt").read_text().splitlines() if line and not line.startswith("#")]
        else:
            filename = "tokenizer.json" if n_text == 704 else "grapheme_mtl_merged_expanded_v1.json"
            tokenizer = json.loads((self.dir_model / filename).read_text())
            if tokenizer["model"]["type"] != "BPE" or tokenizer["pre_tokenizer"] != {"type": "Whitespace"}:
                raise ValueError("Unsupported Chatterbox tokenizer")
            vocab = tokenizer["model"]["vocab"]
            added_ids = {token["id"] for token in tokenizer["added_tokens"]}
            specials = {token["id"] for token in tokenizer["added_tokens"] if token["special"]}
            merges = [" ".join(part for part in (merge.split(" ") if isinstance(merge, str) else merge)) for merge in tokenizer["model"]["merges"]]
        tokens = [f"[chatterbox_unused_{i}]" for i in range(n_text)]
        types = [SentencePieceTokenTypes.UNUSED] * n_text
        for token, index in vocab.items():
            if index < 0 or index >= n_text:
                raise ValueError("Tokenizer exceeds the text embedding table")
            tokens[index] = token
            types[index] = (SentencePieceTokenTypes.CONTROL if index in specials else
                            SentencePieceTokenTypes.USER_DEFINED if index in added_ids else SentencePieceTokenTypes.NORMAL)
        for i in range(self.hparams["chatterbox_speech_vocab"]):
            tokens.append(f"<|chatterbox_speech_{i}|>")
            types.append(SentencePieceTokenTypes.CONTROL)
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre("gpt-2" if turbo else "chatterbox")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(types)
        self.gguf_writer.add_token_merges(merges)
        self.gguf_writer.add_bos_token_id(n_text + 6561)
        self.gguf_writer.add_eos_token_id(n_text + 6562)
        self.gguf_writer.add_add_bos_token(False)
        self.gguf_writer.add_add_eos_token(False)
        if not turbo:
            self.gguf_writer.add_unk_token_id(1)

    def get_tensors(self):
        n_text = self.hparams["chatterbox_text_vocab"]
        turbo = n_text == 50276
        with safe_open(_checkpoint(self.dir_model), framework="pt") as f:
            yield "token_embd.weight", torch.cat([f.get_tensor("text_emb.weight"), f.get_tensor("speech_emb.weight")])
            head = f.get_tensor("speech_head.weight")
            yield "output.weight", torch.cat([torch.zeros((n_text, head.shape[1]), dtype=head.dtype), head])
            if turbo:
                yield "output.bias", torch.cat([torch.full((n_text,), -1e9), f.get_tensor("speech_head.bias").float()])
            for name in f.keys():
                if not name.startswith("tfmr.") or name in ("tfmr.embed_tokens.weight", "tfmr.wte.weight"):
                    continue
                prefix = "transformer." if turbo else "model."
                yield name.replace("tfmr.", prefix, 1), f.get_tensor(name)


@ModelBase.register("ChatterboxT3Model")
class ChatterboxT3Model(ChatterboxT3Mixin, LlamaModel):
    model_arch = gguf.MODEL_ARCH.LLAMA


@ModelBase.register("ChatterboxTurboT3Model")
class ChatterboxTurboT3Model(ChatterboxT3Mixin, GPT2Model):
    model_arch = gguf.MODEL_ARCH.GPT2
