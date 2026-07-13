# Reuse Encoded Dataset Cache Plan

## Summary

Stop regenerating the feature-extracted/tokenized training dataset on every training run by adding a reusable encoded dataset cache path with metadata validation.

## Problem Frame

`train.py` currently preprocesses raw/local datasets every run and saves them to `{output_dir}_encoded`. Because each LoRA experiment uses a different `output_dir`, the same audio is repeatedly decoded, feature-extracted, tokenized, and saved again.

## Evidence

- `train.py` always calls `data_loader.prepare_dataset()` for train and test.
- `train.py` always saves to `encoded_path = f'{output_dir}_encoded'`.
- LoRA target modules, learning rate, batch size, and output directory do not change encoded audio features or labels.

## Scope

In scope:

- Add a reusable encoded cache path in config.
- Reuse the encoded dataset when metadata matches.
- Add a CLI escape hatch to force cache rebuild.
- Update C-plan configs to use a stable dataset-level encoded cache.

Out of scope:

- Changing raw manifest conversion.
- Changing model training behavior.
- Deleting existing generated cache directories.

## Key Decisions

- Put the stable cache path under `dataset.encoded_cache_dir`.
- Default to legacy `{output_dir}_encoded` when no cache path is configured.
- Store `_metadata.json` next to the encoded dataset and compare preprocessing-relevant fields.
- Allow `--rebuild-encoded-cache` to force regeneration.

## Implementation Plan

1. Add cache metadata helpers to `train.py`.
2. Load encoded cache before preprocessing when metadata matches.
3. Save metadata after preprocessing.
4. Add `dataset.encoded_cache_dir` and reuse flags to C CPU/GPU configs.
5. Validate syntax and YAML parsing.

## Verification

- Python syntax check passes.
- C configs parse and include the shared encoded cache path.
- Existing dirty notebook and smoke config remain untouched.

## Risks

- If metadata cannot detect a relevant preprocessing change, stale cache could be reused. Include all known preprocessing-relevant settings and provide `--rebuild-encoded-cache`.
