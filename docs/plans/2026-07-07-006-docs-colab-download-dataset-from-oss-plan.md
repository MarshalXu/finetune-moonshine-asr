# Colab Download Dataset From OSS Plan

## Summary

Update the Colab notebook so training data can be downloaded directly from an OSS URL inside Colab, avoiding slow manual Google Drive upload.

## Problem Frame

Uploading the 5GB+ Arabic dataset zip to Google Drive is too slow. Colab should download the zip from OSS to the local Colab disk, unzip locally, and keep Google Drive only for durable training outputs and logs.

## Evidence

- The existing notebook assumes `DATA_ZIP` under `/content/drive/MyDrive/moonshine`.
- The dataset zip is large, while Colab local disk is faster for unzip/training staging.
- The notebook already copies final artifacts back to Drive.

## Scope

In scope:

- Add notebook variables for `DATA_SOURCE`, `OSS_ZIP_URL`, and `LOCAL_ZIP`.
- Make raw dataset preparation support OSS download or Drive zip fallback.
- Use `curl` with retry/follow-redirect options to handle public or signed OSS URLs.
- Keep Drive mount for outputs/logs.

Out of scope:

- Managing private OSS credentials inside the repo.
- Uploading data to OSS.
- Changing training hyperparameters.

## Key Decisions

- Default `DATA_SOURCE` to `oss` because the user wants to avoid Drive upload.
- Require the user to paste the OSS URL into `OSS_ZIP_URL`.
- Quote the URL in shell commands so signed URLs with `&` work.
- Keep a `drive` data source option for fallback.

## Implementation Plan

1. Edit `examples/arabic_lora_colab.ipynb` path configuration cell.
2. Replace the raw dataset preparation cell with OSS/Drive branching.
3. Validate notebook JSON parses and no stale `DATA_ZIP` assumptions remain.
4. Commit the notebook and plan changes.

## Verification

- Notebook JSON parses.
- Inspection shows `OSS_ZIP_URL`, `DATA_SOURCE`, and local zip logic are present.
- Existing Python scripts still compile.

## Risks

- Signed OSS URLs may expire; the notebook should clearly require refreshing the URL.
- Some OSS URLs may require headers/cookies; this notebook handles normal public or signed URL downloads.
