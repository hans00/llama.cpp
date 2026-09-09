# llama.cpp TTS

This is a tool to demonstrate audio generation capability in llama.cpp via `libmtmd`. It was added via PR [#26254](https://github.com/ggml-org/llama.cpp/pull/26254)

Note: this tool used to serve as a demo for OuteTTS, but it was converted to a more model-agnostic tool.

## Common usage

Simple usage:

```sh
llama-tts -hf ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF -p "Hello world" --output out.wav
```

Common params:
- Sampling params such as `--top-k`, `--top-p`, `--temp`, etc.
- `-n <number_of_frames>` limits the output length, e.g. `-n 500`. Note that how many milliseconds each frame represents varies by model
- Core inference params such as `-ngl`, `-b`, `-ub`, etc.

## Qwen3-TTS

Available params:
- `--tts-lang` can be `zh`, `en`, `de`, `it`, `pt`, `es`, `ja`, `ko`, `fr`, `ru` (default: `en`)
- `--tts-speaker-file` should point to a speaker reference audio file (wav, mp3)

Example usage:

```sh
llama-tts -hf ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF \
    -p "Hello world" \
    --tts-lang english \
    --tts-speaker-file speaker.mp3 \
    --output out.wav
```

## Pocket TTS

Available params:
- `--tts-speaker-file` should point to a speaker reference audio file (wav, mp3). It is required, the model produces almost no audio without it
- Note: `lang` is not used, the language is a property of the weights

Example usage:

```sh
llama-tts -m pocket-tts.gguf \
    -mm mmproj-pocket-tts.gguf \
    -p "Hello world" \
    --tts-speaker-file speaker.mp3 \
    --output out.wav
```

**Note for GGUF conversion:**

The [upstream repository](https://huggingface.co/kyutai/pocket-tts) holds one complete model per language under `languages/`, next to a set of shared files at the root. Convert one of the `languages/<name>` directories, **not** the root directory:

```sh
python convert_hf_to_gguf.py path/to/pocket-tts/languages/english --outfile pocket-tts.gguf
python convert_hf_to_gguf.py path/to/pocket-tts/languages/english --mmproj --outfile mmproj-pocket-tts.gguf
```

## Orpheus TTS

Available params:
- `--tts-voice`: `tara`, `leah`, `jess`, `leo`, `dan`, `mia`, `zac`, `zoe` for the English finetuned model
- `--tts-speaker-file` and `--tts-speaker-text`: reference audio (up to 30 seconds) and its transcript, used together instead of `--tts-voice` for reference conditioning
- Language and available voices depend on the backbone checkpoint

Example usage:

```sh
llama-tts -m orpheus.gguf --mmproj mmproj-snac.gguf \
    -p "Hello world" --tts-voice tara --output out.wav
```

**Note for GGUF conversion:**

Convert the [Orpheus backbone](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) and [SNAC 24 kHz](https://huggingface.co/hubertsiuzdak/snac_24khz) separately:

```sh
python convert_hf_to_gguf.py path/to/orpheus --model-architecture OrpheusForCausalLM --outfile orpheus.gguf
python convert_hf_to_gguf.py path/to/snac_24khz --mmproj --mmproj-architecture SnacModel --outfile mmproj-snac.gguf
```
