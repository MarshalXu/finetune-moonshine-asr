# Arabic LoRA C Config Plan

## Summary

Add a separate C-plan LoRA training config for Moonshine Arabic fine-tuning, expanding LoRA target modules from attention projections only to attention plus FFN layers.

## Problem Frame

The B-plan experiment using `q_proj/k_proj/v_proj/o_proj` has completed with only average results. The next experiment should test whether adding FFN adaptation (`fc1/fc2`) improves telephone-recording domain adaptation without overwriting the existing baseline config.

## Evidence

- Existing Colab notebook derives runtime config from `configs/moonshine_base_ar_lora_cpu_train.yaml`.
- Existing base config uses a conservative LoRA setup.
- Moonshine uses attention projections named `q_proj`, `k_proj`, `v_proj`, `o_proj`, and FFN layers named `fc1`, `fc2`.

## Scope

In scope:

- Add a new YAML config for C-plan training.
- Use a distinct output/log directory.
- Keep epoch checkpointing and WER-based best model selection.

Out of scope:

- Changing the Colab notebook default.
- Changing training code.
- Changing dataset preparation.

## Key Decisions

- Use `q_proj/k_proj/v_proj/o_proj/fc1/fc2` as LoRA targets.
- Use `r=16`, `lora_alpha=32`, and `dropout=0.05` as a moderate-capacity C plan.
- Lower learning rate to `5e-5` because C trains more adapter parameters than B and may overfit or oscillate more easily.

## Implementation Plan

1. Add `configs/moonshine_base_ar_lora_c_cpu_train.yaml`.
2. Keep dataset path and train/eval/save policy aligned with the existing Arabic LoRA config.
3. Validate the YAML parses successfully.

## Verification

- YAML parses with `yaml.safe_load`.
- Config includes all intended target modules.

## Risks

- C may improve capacity but also increase overfitting risk.
- Static config is CPU-shaped; Colab should still override batch size, fp16, output, and dataset path through the notebook runtime config cell.
