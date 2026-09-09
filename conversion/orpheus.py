from __future__ import annotations

import json

from .base import ModelBase, gguf
from .llama import LlamaModel


@ModelBase.register("OrpheusForCausalLM")
class OrpheusModel(LlamaModel):
    model_arch = gguf.MODEL_ARCH.LLAMA

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_string("llama.tts.model", "orpheus")
        self.gguf_writer.add_sampling_sequence("penalties;temperature;top_k;top_p")
        self.gguf_writer.add_sampling_temp(0.6)
        self.gguf_writer.add_sampling_top_k(50)
        self.gguf_writer.add_sampling_top_p(0.9)
        self.gguf_writer.add_sampling_min_p(0.0)
        self.gguf_writer.add_sampling_penalty_repeat(1.1)
        self.gguf_writer.add_sampling_penalty_last_n(8192)

    def get_vocab_base(self):
        # The pretrained tokenizer includes an unused <|audio|> token beyond the embedding table.
        tokenizer = json.loads((self.dir_model / "tokenizer.json").read_text())
        size = self.hparams["vocab_size"]
        extra = [t for t in tokenizer["added_tokens"] if t["id"] >= size]
        if any(t["content"] != "<|audio|>" or t["id"] != size for t in extra):
            raise ValueError("Orpheus tokenizer exceeds the embedding table")
        try:
            self.hparams["vocab_size"] = size + len(extra)
            tokens, types, pre = super().get_vocab_base()
        finally:
            self.hparams["vocab_size"] = size
        return tokens[:size], types[:size], pre

    def set_vocab(self):
        tokenizer = json.loads((self.dir_model / "tokenizer.json").read_text())
        tokens = {t["content"]: t["id"] for t in tokenizer["added_tokens"]}
        if self.hparams["vocab_size"] < 156938 or self.hparams["hidden_size"] != 3072 or any(
            tokens.get(f"<custom_token_{i}>") != 128256 + i for i in range(28682)
        ):
            raise ValueError("Unsupported Orpheus SNAC token layout")
        super().set_vocab()
        self.gguf_writer.add_eos_token_id(128258)
        self.gguf_writer.add_add_eos_token(False)
