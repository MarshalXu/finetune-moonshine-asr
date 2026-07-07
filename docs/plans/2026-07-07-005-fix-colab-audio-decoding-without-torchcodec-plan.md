# Colab Audio Decoding Without TorchCodec Fix Plan

## Summary

Fix the Colab notebook/runtime path by avoiding `torchcodec` as a hard dependency. Store audio paths in the converted HF dataset and let the training data loader decode paths with `soundfile`/`librosa`.

## Problem Frame

Colab reports two failures: a shell heredoc warning from a notebook install/verification cell, and a hard `torchcodec` import failure caused by Colab PyTorch `2.11.0+cu128` not matching the installed TorchCodec binary. Since `datasets.Audio` now routes decoding through TorchCodec, Colab should avoid `datasets.Audio` for this workflow.

## Evidence

- The notebook cell `!python - <<'PYCODE'` produces `here-document ... wanted PYCODE`, because Jupyter `!` shell lines do not handle the multi-line heredoc as intended.
- `import torchcodec` fails in Colab with `Could not load libtorchcodec` and missing `libnvrtc.so.13`.
- `moonshine_ft/data_loader.py` currently assumes `batch["audio"]` is decoded into `array` and `sampling_rate`.
- The converter can instead save plain path strings, avoiding `datasets.Audio` decode.

## Scope

In scope:

- Add path-string audio support to `MoonshineDataLoader.prepare_dataset()`.
- Add converter options to avoid casting audio to `datasets.Audio` and validate with `soundfile` instead of TorchCodec.
- Update the Colab notebook to remove shell heredoc, avoid importing TorchCodec, and pass no-cast/soundfile validation flags.
- Keep local TorchCodec-compatible behavior available for existing workflows.

Out of scope:

- Changing LoRA hyperparameters.
- Rewriting all dataset loading paths.
- Installing alternate Colab PyTorch/CUDA builds.

## Key Decisions

- Decode audio paths in `data_loader.py` with `soundfile`, resample with `librosa` only when needed, and downmix stereo to mono.
- Preserve `datasets.Audio` support for already-decoded local datasets.
- Make converter no-cast behavior opt-in via `--no-cast-audio` so existing local smoke datasets still work.
- In Colab, use `--validate-audio --validation-backend soundfile --no-cast-audio`.

## Implementation Plan

1. Update `scripts/prepare_manifest_dataset.py` with `--no-cast-audio` and `--validation-backend`.
2. Update `moonshine_ft/data_loader.py` to handle audio as a path string or `{path: ...}` dict.
3. Update `examples/arabic_lora_colab.ipynb` to remove heredoc and TorchCodec dependency.
4. Adjust requirements so `torchcodec` is optional/commented rather than installed unconditionally.
5. Run syntax checks and a small no-cast dataset conversion/training smoke.

## Verification

- Notebook JSON parses.
- `python -m py_compile train.py scripts/prepare_manifest_dataset.py moonshine_ft/data_loader.py` succeeds.
- A no-cast validation conversion creates a dataset with string audio paths.
- A tiny LoRA smoke run against the no-cast dataset completes.

## Risks

- `soundfile` may not decode every format TorchCodec can decode, but this Arabic dataset is WAV-based.
- Path-string datasets are less portable if moved away from the audio folder; Colab keeps them in one workspace for training.
