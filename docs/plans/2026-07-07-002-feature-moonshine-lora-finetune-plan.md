# Moonshine LoRA Fine-Tune Plan

## Summary

Add an optional LoRA/PEFT training path to the existing Moonshine fine-tuning script so Arabic adaptation can train a small number of parameters instead of full 61M-parameter fine-tuning. The default full fine-tune behavior must remain unchanged unless `peft.enabled: true` is present in the config.

## Problem Frame

The current project fine-tunes all model parameters. With roughly 50 hours of Arabic data and a CPU-only environment, full fine-tuning is higher-risk: it is slower, more likely to overfit, and produces larger checkpoints. LoRA should adapt attention projection layers while freezing the base model weights, making the experiment cheaper and easier to iterate.

## Evidence

- `train.py` currently loads `MoonshineForConditionalGeneration.from_pretrained(...)` and passes the full model to `Seq2SeqTrainer`.
- The successful smoke run proved the existing full fine-tune path works on `UsefulSensors/moonshine-base-ar` with the converted Arabic dataset.
- `peft` is currently missing from `moonshine-ft`.
- Moonshine linear module names include encoder and decoder attention projections such as `q_proj`, `k_proj`, `v_proj`, and `o_proj`, plus MLP layers `fc1` and `fc2`.
- PEFT target modules can use suffix names such as `q_proj` and `v_proj`, which match the inspected Moonshine module names.

## Scope

In scope:

- Add `peft` to the project dependency list and install it into `moonshine-ft` for verification.
- Add optional `peft:` config support in `train.py`.
- Apply LoRA only when `peft.enabled: true`.
- Print trainable parameter counts for LoRA runs.
- Add a CPU LoRA smoke config that reuses the existing converted Arabic smoke dataset.
- Run a tiny LoRA smoke training command and verify adapter artifacts are saved.

Out of scope:

- Full 50-hour training run.
- Hyperparameter search.
- Quantized LoRA or QLoRA.
- OpenVINO/NPU export or inference.
- Merging LoRA adapters into the base model.
- Refactoring trainer/data loader structure beyond the smallest PEFT integration.

## Key Decisions

- Keep LoRA optional and config-driven so existing configs keep full fine-tune semantics.
- Start with attention projections only: `q_proj`, `v_proj`, `k_proj`, and `o_proj`. This is a conservative ASR adaptation target and avoids adapting every MLP layer on the first pass.
- Use small rank defaults for CPU smoke, e.g. `r: 8`, `lora_alpha: 16`, and `lora_dropout: 0.05`.
- Use PEFT generic wrapping by default because the `SEQ_2_SEQ_LM` wrapper duplicates decoder `input_ids` for Moonshine. Keep `peft.task_type` optional for future experiments.
- Save processor/tokenizer alongside adapter output as the existing script already does.

## Implementation Plan

1. Install and verify `peft` in `moonshine-ft`.
2. Update `requirements.txt` with `peft>=0.10.0` near training dependencies.
3. Update `train.py`:
   - import PEFT lazily or with a clear error only when enabled,
   - read `peft` config after base model load/token setup,
   - construct `LoraConfig`,
   - call `get_peft_model(model, lora_config)`,
   - print trainable parameter counts,
   - leave full fine-tune path unchanged when disabled.
4. Add `configs/moonshine_base_ar_lora_cpu_smoke.yaml` based on the existing CPU smoke config.
5. Run syntax checks.
6. Run tiny LoRA smoke training with a separate output directory.
7. Inspect output artifacts to confirm adapter files are saved.

## Verification

- `conda run -n moonshine-ft python -c "import peft"` succeeds.
- `python -m py_compile train.py scripts/prepare_manifest_dataset.py` succeeds.
- Full fine-tune config remains parseable.
- LoRA smoke training reaches `STARTING TRAINING` and completes configured small `max_steps`.
- Logs show trainable parameter count is much smaller than total parameters.
- Final output contains adapter artifacts such as `adapter_config.json` and adapter weights.

## Risks

- PEFT task-specific wrappers may not recognize Moonshine generation/decoder behavior; use generic wrapping unless a future version handles Moonshine-specific `input_values` cleanly.
- Applying LoRA to all attention projections may still be slower than desired on CPU; if needed, reduce targets to `q_proj` and `v_proj`.
- Adapter-only save behavior differs from full model save behavior; downstream inference will need base model plus adapter loading.
- LoRA smoke WER will not be meaningful; it only validates training mechanics.
