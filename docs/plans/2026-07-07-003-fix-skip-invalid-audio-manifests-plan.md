# Skip Invalid Audio Manifests Fix Plan

## Summary

Fix full Arabic dataset preparation so training can skip manifest rows whose audio files cannot be decoded, instead of failing during `train.py` preprocessing.

## Problem Frame

The LoRA training command reaches dataset preprocessing and fails inside `datasets.Audio`/`torchcodec` with `No audio frames were decoded`. This means at least one manifest row points to an audio file that exists but has no decodable frames, or a header/container that `torchcodec` cannot decode.

## Evidence

- The failing command loads `datasets/whisper_ar_manifist0508_hf`, filters 28,948 train rows to 27,951, then fails during `MoonshineDataLoader.prepare_dataset()`.
- The traceback points to `audio["array"]` in `moonshine_ft/data_loader.py` and `torchcodec` raising `No audio frames were decoded`.
- The converter currently validates only that referenced audio paths exist; it does not validate decodability.

## Scope

In scope:

- Add optional audio validation to `scripts/prepare_manifest_dataset.py`.
- Skip invalid rows with a warning/count instead of saving them to the HF dataset.
- Keep existing conversion behavior usable for fast smoke datasets.
- Verify the bad sample can be detected before training.

Out of scope:

- Repairing corrupted audio files.
- Changing training hyperparameters.
- Changing LoRA implementation.
- Adding expensive waveform normalization or resampling outside `datasets.Audio`.

## Key Decisions

- Validate during manifest conversion rather than during `train.py` preprocessing, so bad rows are removed once and future training runs are stable.
- Use `torchcodec.decoders.AudioDecoder` for validation because that is the decoder path currently failing under `datasets.Audio`.
- Make validation opt-in with `--validate-audio` to avoid slowing down quick smoke conversion unless needed.

## Implementation Plan

1. Add `--validate-audio` and `--skip-invalid-audio` options to `scripts/prepare_manifest_dataset.py`.
2. When validation is enabled, decode the candidate audio with `torchcodec` and require at least one sample.
3. If `--skip-invalid-audio` is set, print a warning and skip invalid rows; otherwise fail fast.
4. Print loaded/skipped counts for train and test manifests.
5. Run a bounded/full validation command until the known failure is caught.

## Verification

- `python -m py_compile scripts/prepare_manifest_dataset.py train.py` succeeds.
- A validation run over the manifest reports skipped invalid audio rows instead of crashing.
- The regenerated HF dataset can be used by the same training command.

## Risks

- Full audio validation reads every audio file and will take longer than path-only conversion.
- Some files may decode in another backend but not in `torchcodec`; since training uses `torchcodec`, those rows should still be excluded for this pipeline.
