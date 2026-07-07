# Colab LoRA Notebook Plan

## Summary

Add a Google Colab-friendly notebook for Arabic Moonshine LoRA training so the project can be run on Colab GPUs without hand-entering long terminal commands.

## Problem Frame

The shell workflow works locally but is slow on CPU. Colab users usually prefer `.ipynb` files with sequential setup, Drive mount, data conversion, training, resume, and artifact copy cells.

## Evidence

- The repo already has an examples notebook, but it is a broad curriculum tutorial rather than the Arabic LoRA workflow.
- The current Arabic LoRA path needs `scripts/prepare_manifest_dataset.py`, `configs/moonshine_base_ar_lora_cpu_train.yaml`, `peft`, and `torchcodec`.
- The training data is excluded from git and must come from Google Drive or manual upload in Colab.

## Scope

In scope:

- Add a Colab notebook under `examples/` for Arabic LoRA training.
- Add `torchcodec` to `requirements.txt` for Colab audio decoding.
- Include cells for clone/pull branch, install, mount Drive, unzip/copy data, validate/convert dataset, train, resume, and inspect checkpoints.

Out of scope:

- Uploading data to Drive.
- Running Colab training from this local environment.
- Reworking the training code.

## Key Decisions

- Keep the notebook command-driven so it mirrors the tested CLI workflow.
- Use Google Drive as the durable storage location for source data and copied training outputs.
- Default to the pushed branch `shawnx/peft_ft` but make repo/branch variables editable.
- Use `--no-capture-output` and `python -u` in training cells so Colab shows logs live.

## Implementation Plan

1. Create `examples/arabic_lora_colab.ipynb`.
2. Add `torchcodec>=0.14.0` to `requirements.txt`.
3. Validate the notebook JSON loads.
4. Provide the exact file path and next push command.

## Verification

- Notebook JSON parses successfully.
- `python -m py_compile train.py scripts/prepare_manifest_dataset.py` succeeds.
- `requirements.txt` contains both `peft` and `torchcodec`.

## Risks

- Colab package versions may change; install cell pins only the project requirements and leaves PyTorch to Colab unless the user chooses otherwise.
- Full validation conversion reads every audio file and can take several minutes.
- Drive paths need user adjustment if the zip is stored elsewhere.
