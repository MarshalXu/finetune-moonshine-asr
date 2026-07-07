# Arabic CPU Moonshine Fine-Tune Smoke Plan

## Summary

Run the repository's existing Moonshine fine-tuning path end to end on the prepared Arabic manifest dataset, using `UsefulSensors/moonshine-base-ar` and CPU-safe settings first. The goal is a small smoke training run that proves data loading, preprocessing, model loading, loss computation, checkpointing, and evaluation wiring before any LoRA work.

## Problem Frame

The dataset at `datasets/whisper_ar_manifist0508` is ready, but it is not directly compatible with the repository's existing `csv` loader because the manifests are JSONL files with nested `audio.path` objects. The current training script already supports `local` Hugging Face datasets via `load_from_disk()`, so the lowest-risk path is to convert the JSONL manifests into a saved `DatasetDict` with `train` and `test` splits.

The target machine has no usable CUDA in the current environment. The first run must use CPU-safe settings and a tiny subset, not the GPU-oriented example config.

## Evidence

- `datasets/whisper_ar_manifist0508/train.jsonl` and `test.jsonl` contain `audio.path`, `sentence`, `duration`, `language`, `uid`, and `channel`.
- `train.jsonl` has 28,948 rows and `test.jsonl` has 7,237 rows.
- Manifest paths are relative to `datasets/`; for example `whisper_ar_manifist0508/audio/train/...wav` exists under `datasets/whisper_ar_manifist0508/audio/train/...`.
- `moonshine_ft/data_loader.py` supports `dataset.type: local` and expects a saved Hugging Face `DatasetDict` with `train` and `test`.
- `moonshine_ft/data_loader.py` can rename a configured text column to `sentence`, and `train.py` uses `sentence` during curriculum filtering and tokenization.
- `train.py` filters by `duration` or `audio_duration` before feature extraction, so preserving `duration` in the converted dataset is required.
- `configs/mls_french_no_curriculum.yaml` uses GPU-oriented settings such as `fp16: true`; CPU smoke training should disable FP16 and use very small batches and steps.
- `moonshine-ft` resolves to `/root/miniconda3/envs/moonshine-ft/bin/python`; installed dependencies include Torch, Transformers, Datasets, Accelerate, Evaluate, Jiwer, Schedulefree, Torchaudio, Librosa, and Soundfile.
- In `moonshine-ft`, `torch.cuda.is_available()` and `torch.xpu.is_available()` are both false.

## Scope

In scope:

- Create a small repo-local conversion utility for this JSONL manifest shape.
- Convert the prepared Arabic manifests into a saved Hugging Face `DatasetDict`.
- Create a CPU smoke-training config using `UsefulSensors/moonshine-base-ar`.
- Run a tiny smoke training job through the existing `train.py` path.
- Report exact commands, generated artifact paths, and any blocker.

Out of scope:

- Full dataset training.
- LoRA/PEFT implementation.
- NPU training.
- ONNX/OpenVINO export.
- Dataset quality cleanup beyond schema/path validation needed to start training.
- Uploading model outputs or pushing to Hugging Face Hub.

## Key Decisions

- Use Hugging Face `local` dataset format because it matches the existing code path with the fewest changes.
- Keep generated training artifacts under ignored output-style paths where possible.
- Use the Hugging Face model ID `UsefulSensors/moonshine-base-ar` directly. Let Transformers cache the baseline model instead of vendoring model files into the repo.
- Use a small subset and low `max_steps` for the first run because CPU training on `moonshine-base-ar` can be slow.
- Preserve `duration` and `sentence` columns in the converted dataset so current duration filtering and tokenizer paths work unchanged.

## Implementation Plan

1. Add a conversion script under `scripts/` that:
   - reads `train.jsonl` and `test.jsonl`,
   - resolves `audio.path` relative to `datasets/`,
   - validates that referenced audio files exist,
   - writes a `DatasetDict` with `audio`, `sentence`, `duration`, `language`, `uid`, and `channel`,
   - casts `audio` to `datasets.Audio(sampling_rate=16000)`,
   - supports `--max-train-samples` and `--max-test-samples` for smoke datasets.

2. Create a CPU smoke config under `configs/`:
   - `model.name: UsefulSensors/moonshine-base-ar`,
   - `dataset.type: local`,
   - `dataset.path` pointing to the converted smoke dataset,
   - `audio.min_duration` low enough to include short Arabic utterances for smoke validation,
   - `training.fp16: false`,
   - tiny train/eval batch sizes,
   - small `max_steps`,
   - frequent logging and no Hub push.

3. Run conversion on a bounded sample first.

4. Run `train.py` with the CPU smoke config.

5. If smoke training fails due to a project compatibility issue, make the smallest targeted fix and rerun the focused command.

## Verification

- Conversion command exits successfully and writes a loadable dataset.
- A read-only dataset check confirms `train` and `test` split sizes, columns, first audio sampling rate, and first transcription.
- `train.py --config <cpu-smoke-config>` reaches training and completes the configured small `max_steps`.
- Final model/checkpoint output exists under the configured result directory.

## Risks

- CPU training may be very slow even for the base model; mitigate with tiny sample counts and very low `max_steps`.
- Some manifest rows may reference missing audio files; conversion should fail clearly or allow diagnosing missing paths.
- Short utterances are common in the dataset and may be filtered out if `min_duration` is too high; CPU smoke config should use a permissive duration range.
- The current trainer may expose compatibility issues with the installed Transformers version only once training starts; fixes should be narrowly scoped.
- The first smoke result is only a pipeline validation, not a meaningful model-quality result.
