"""
Convert nested JSONL ASR manifests into a Hugging Face DatasetDict.

Expected input rows:
    {
      "audio": {"path": "relative/or/absolute.wav"},
      "sentence": "...",
      "duration": 1.23,
      ...
    }
"""

import argparse
import json
import shutil
from pathlib import Path

from datasets import Audio, Dataset, DatasetDict
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a local Hugging Face DatasetDict from JSONL ASR manifests."
    )
    parser.add_argument(
        "--manifest-dir",
        required=True,
        help="Directory containing train.jsonl and test.jsonl.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for DatasetDict.save_to_disk().",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Root used to resolve relative audio paths. Defaults to manifest-dir parent.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        help="Target sampling rate for datasets.Audio.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional maximum number of valid train rows to convert.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional maximum number of valid test rows to convert.",
    )
    parser.add_argument(
        "--validate-audio",
        action="store_true",
        help="Decode every audio file before saving it.",
    )
    parser.add_argument(
        "--validation-backend",
        choices=["soundfile", "torchcodec"],
        default="soundfile",
        help="Audio backend for --validate-audio.",
    )
    parser.add_argument(
        "--no-cast-audio",
        action="store_true",
        help="Store audio paths as plain strings instead of datasets.Audio.",
    )
    parser.add_argument(
        "--skip-invalid-audio",
        action="store_true",
        help="Skip rows that fail audio validation instead of failing fast.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory first if it exists.",
    )
    return parser.parse_args()


def resolve_audio_path(raw_path, manifest_dir, data_root):
    audio_path = Path(raw_path)
    if audio_path.is_absolute():
        return audio_path

    candidates = [
        data_root / audio_path,
        manifest_dir / audio_path,
        Path.cwd() / audio_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Audio file not found for {raw_path!r}. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def validate_audio(path, backend="soundfile"):
    if backend == "torchcodec":
        from torchcodec.decoders import AudioDecoder

        samples = AudioDecoder(str(path)).get_all_samples()
        if samples.data.numel() == 0:
            raise ValueError("decoded audio has zero samples")
        return

    import soundfile as sf

    info = sf.info(str(path))
    if info.frames <= 0:
        raise ValueError("decoded audio has zero frames")


def load_manifest(
    path,
    manifest_dir,
    data_root,
    max_samples,
    sampling_rate,
    validate_audio_files=False,
    skip_invalid_audio=False,
    validation_backend="soundfile",
    no_cast_audio=False,
):
    rows = []
    skipped = 0
    total = sum(1 for _ in path.open("r", encoding="utf-8"))

    with path.open("r", encoding="utf-8") as handle:
        iterator = tqdm(handle, total=total, desc=f"Loading {path.name}")
        for index, line in enumerate(iterator):
            if max_samples is not None and len(rows) >= max_samples:
                break

            record = json.loads(line)
            audio = record.get("audio")
            if not isinstance(audio, dict) or not audio.get("path"):
                raise ValueError(f"{path}:{index + 1} is missing audio.path")

            sentence = record.get("sentence")
            if sentence is None:
                raise ValueError(f"{path}:{index + 1} is missing sentence")

            duration = record.get("duration")
            if duration is None:
                raise ValueError(f"{path}:{index + 1} is missing duration")

            resolved_audio = resolve_audio_path(audio["path"], manifest_dir, data_root)
            if validate_audio_files:
                try:
                    validate_audio(resolved_audio, backend=validation_backend)
                except Exception as exc:
                    message = f"{path}:{index + 1} invalid audio {resolved_audio}: {exc}"
                    if skip_invalid_audio:
                        skipped += 1
                        print(f"[WARNING] Skipping {message}")
                        continue
                    raise RuntimeError(message) from exc

            rows.append(
                {
                    "audio": str(resolved_audio),
                    "sentence": str(sentence),
                    "duration": float(duration),
                    "language": record.get("language"),
                    "uid": record.get("uid"),
                    "channel": record.get("channel"),
                }
            )

    if not rows:
        raise ValueError(f"No rows loaded from {path}")

    print(f"{path.name}: loaded {len(rows):,} rows, skipped {skipped:,} invalid rows")
    dataset = Dataset.from_list(rows)
    if no_cast_audio:
        return dataset
    return dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))


def main():
    args = parse_args()
    manifest_dir = Path(args.manifest_dir).expanduser().resolve()
    data_root = (
        Path(args.data_root).expanduser().resolve()
        if args.data_root
        else manifest_dir.parent
    )
    output = Path(args.output).expanduser().resolve()

    train_manifest = manifest_dir / "train.jsonl"
    test_manifest = manifest_dir / "test.jsonl"
    if not train_manifest.exists():
        raise FileNotFoundError(train_manifest)
    if not test_manifest.exists():
        raise FileNotFoundError(test_manifest)

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} already exists. Use --overwrite to replace it.")
        shutil.rmtree(output)

    dataset = DatasetDict(
        {
            "train": load_manifest(
                train_manifest,
                manifest_dir,
                data_root,
                args.max_train_samples,
                args.sampling_rate,
                validate_audio_files=args.validate_audio,
                skip_invalid_audio=args.skip_invalid_audio,
                validation_backend=args.validation_backend,
                no_cast_audio=args.no_cast_audio,
            ),
            "test": load_manifest(
                test_manifest,
                manifest_dir,
                data_root,
                args.max_test_samples,
                args.sampling_rate,
                validate_audio_files=args.validate_audio,
                skip_invalid_audio=args.skip_invalid_audio,
                validation_backend=args.validation_backend,
                no_cast_audio=args.no_cast_audio,
            ),
        }
    )
    dataset.save_to_disk(str(output))

    print(f"Saved dataset to {output}")
    print(f"Train rows: {len(dataset['train'])}")
    print(f"Test rows: {len(dataset['test'])}")
    print(f"Columns: {dataset['train'].column_names}")


if __name__ == "__main__":
    main()
