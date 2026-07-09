#!/usr/bin/env python3
"""
Slice MS ASR dual-channel audio files based on Microsoft ASR results (jsonls),
resample them to 16kHz mono, and prepare train/test JSONL manifests.
Then merge them with the human-annotated whisper_ar_manifist0508 dataset.
"""

import os
import json
import random
import shutil
import warnings
from pathlib import Path
import soundfile as sf
import librosa
from tqdm import tqdm

warnings.filterwarnings('ignore')

def main():
    # ----------------------------------------------------
    # Configuration and Paths
    # ----------------------------------------------------
    random.seed(42)  # For reproducible train/test split
    
    project_root = Path(__file__).resolve().parent.parent
    datasets_dir = project_root / "datasets"
    
    # MS ASR source paths
    ms_dir = datasets_dir / "ms_asr_data"
    association_file = ms_dir / "association.jsonl"
    audios_src_dir = ms_dir / "audios"
    jsonls_src_dir = ms_dir / "jsonls"
    
    # Sliced output paths
    segmented_dir = ms_dir / "segmented_audio"
    ms_train_manifest = ms_dir / "train.jsonl"
    ms_test_manifest = ms_dir / "test.jsonl"
    
    # Target dataset paths
    whisper_dir = datasets_dir / "whisper_ar_manifist0508"
    whisper_train_manifest = whisper_dir / "train.jsonl"
    whisper_test_manifest = whisper_dir / "test.jsonl"
    
    # Merged dataset paths
    merged_dir = datasets_dir / "merged_ar_manifist"
    merged_train_manifest = merged_dir / "train.jsonl"
    merged_test_manifest = merged_dir / "test.jsonl"
    
    # Final HuggingFace local output directory
    final_hf_dir = datasets_dir / "whisper_ar_manifist0508_hf"
    
    # Create necessary output directories
    segmented_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STARTING DATASET PREPROCESSING AND SLICING WORKFLOW")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print(f"MS ASR directory: {ms_dir}")
    print(f"Whisper manifest directory: {whisper_dir}")
    print(f"Merged manifest directory: {merged_dir}")
    print(f"Final HF local directory: {final_hf_dir}")
    print("-" * 80)

    # ----------------------------------------------------
    # Step 1: Read MS ASR association.jsonl & Split IDs
    # ----------------------------------------------------
    if not association_file.exists():
        print(f"[ERROR] Association file not found at: {association_file}")
        return

    print("Step 1: Reading association.jsonl and performing session-level split...")
    sessions = []
    with open(association_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sessions.append(json.loads(line))
    
    total_sessions = len(sessions)
    print(f"[OK] Loaded {total_sessions} sessions from association.jsonl")

    # Shuffle sessions and split (90% train, 10% test)
    random.shuffle(sessions)
    split_idx = int(total_sessions * 0.9)
    train_sessions = sessions[:split_idx]
    test_sessions = sessions[split_idx:]
    
    train_uids = {s['unique_id'] for s in train_sessions}
    test_uids = {s['unique_id'] for s in test_sessions}
    
    print(f"  Train sessions: {len(train_sessions)} ({len(train_sessions)/total_sessions:.1%})")
    print(f"  Test sessions: {len(test_sessions)} ({len(test_sessions)/total_sessions:.1%})")
    print("-" * 80)

    # ----------------------------------------------------
    # Step 2: Slice and preprocess MS ASR audios
    # ----------------------------------------------------
    print("Step 2: Slicing MS ASR audios and creating MS JSONL manifests...")
    
    train_rows = []
    test_rows = []
    
    missing_audio_count = 0
    missing_jsonl_count = 0
    total_segments_processed = 0
    
    # We iterate over all sessions
    for session_idx, session in enumerate(tqdm(sessions, desc="Processing sessions")):
        uid = session['unique_id']
        wav_path = audios_src_dir / f"{uid}.wav"
        jsonl_path = jsonls_src_dir / f"{uid}.jsonl"
        
        if not wav_path.exists():
            missing_audio_count += 1
            continue
        if not jsonl_path.exists():
            missing_jsonl_count += 1
            continue
            
        # Determine whether this session goes to train or test
        is_train = uid in train_uids
        rows_list = train_rows if is_train else test_rows
        
        # Load the audio at original sample rate
        try:
            audio_data, original_sr = sf.read(str(wav_path))
        except Exception as e:
            print(f"\n[WARNING] Failed to read audio {wav_path.name}: {e}")
            continue
            
        # Parse MS ASR results from JSONL
        segments = []
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        segments.append(json.loads(line))
        except Exception as e:
            print(f"\n[WARNING] Failed to parse jsonl {jsonl_path.name}: {e}")
            continue
            
        # Create output directory for this session's slices
        session_slice_dir = segmented_dir / uid
        session_slice_dir.mkdir(parents=True, exist_ok=True)
        
        for seg_idx, seg in enumerate(segments):
            channel_str = seg.get('channel')
            start_time_str = seg.get('start_time')
            end_time_str = seg.get('end_time')
            text = seg.get('text', '').strip()
            
            if not text or channel_str is None or start_time_str is None or end_time_str is None:
                continue
                
            ch = int(channel_str)
            start_ms = float(start_time_str)
            end_ms = float(end_time_str)
            
            start_s = start_ms / 1000.0
            end_s = end_ms / 1000.0
            duration = end_s - start_s
            
            if duration <= 0.2:  # Skip extremely short slices (e.g. <= 0.2s)
                continue
                
            start_frame = int(start_s * original_sr)
            end_frame = int(end_s * original_sr)
            
            # Clamp frames to audio length
            start_frame = max(0, min(start_frame, len(audio_data)))
            end_frame = max(0, min(end_frame, len(audio_data)))
            
            if start_frame >= end_frame:
                continue
                
            # Extract mono channel slice
            try:
                slice_data = audio_data[start_frame:end_frame, ch]
            except IndexError:
                # In case the channel doesn't exist
                continue
                
            # Skip if slice is empty or contains only zeros
            if len(slice_data) == 0 or not slice_data.any():
                continue
                
            # Resample to 16000 Hz using soxr_hq
            try:
                resampled_slice = librosa.resample(slice_data, orig_sr=original_sr, target_sr=16000, res_type='soxr_hq')
            except Exception as e:
                print(f"\n[WARNING] Failed to resample slice {uid}_seg{seg_idx}: {e}")
                continue
                
            # Save resampled slice as mono 16kHz WAV
            slice_filename = f"slice_{seg_idx:04d}.wav"
            slice_file_path = session_slice_dir / slice_filename
            
            try:
                sf.write(str(slice_file_path), resampled_slice, 16000)
            except Exception as e:
                print(f"\n[WARNING] Failed to write slice {slice_file_path}: {e}")
                continue
                
            # Construct relative path from datasets/ directory
            relative_audio_path = f"ms_asr_data/segmented_audio/{uid}/{slice_filename}"
            
            row = {
                "audio": {"path": relative_audio_path},
                "sentence": text,
                "language": "ar",
                "sentences": [{"start": 0.0, "end": float(duration), "text": text}],
                "duration": float(duration),
                "uid": uid,
                "channel": "left" if ch == 0 else "right"
            }
            rows_list.append(row)
            total_segments_processed += 1

    # Save MS train/test manifests
    with open(ms_train_manifest, 'w', encoding='utf-8') as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
    with open(ms_test_manifest, 'w', encoding='utf-8') as f:
        for r in test_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"\n[OK] Completed MS ASR slicing:")
    print(f"  Missing audios: {missing_audio_count}, Missing jsonls: {missing_jsonl_count}")
    print(f"  Total sliced segments processed: {total_segments_processed:,}")
    print(f"  MS Train rows: {len(train_rows):,}")
    print(f"  MS Test rows: {len(test_rows):,}")
    print(f"  Saved MS Train manifest: {ms_train_manifest}")
    print(f"  Saved MS Test manifest: {ms_test_manifest}")
    print("-" * 80)

    # ----------------------------------------------------
    # Step 3: Merge MS ASR slices with whisper_ar_manifist0508
    # ----------------------------------------------------
    print("Step 3: Merging datasets (MS ASR + human-annotated)...")
    
    merged_train_rows = []
    merged_test_rows = []
    
    # Read whisper train manifest
    whisper_train_count = 0
    if whisper_train_manifest.exists():
        with open(whisper_train_manifest, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    merged_train_rows.append(json.loads(line))
                    whisper_train_count += 1
    else:
        print(f"[WARNING] Whisper train manifest not found at: {whisper_train_manifest}")

    # Read whisper test manifest
    whisper_test_count = 0
    if whisper_test_manifest.exists():
        with open(whisper_test_manifest, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    merged_test_rows.append(json.loads(line))
                    whisper_test_count += 1
    else:
        print(f"[WARNING] Whisper test manifest not found at: {whisper_test_manifest}")

    # Add MS Train rows
    merged_train_rows.extend(train_rows)
    # Add MS Test rows
    merged_test_rows.extend(test_rows)
    
    # Save merged manifests
    with open(merged_train_manifest, 'w', encoding='utf-8') as f:
        for r in merged_train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
    with open(merged_test_manifest, 'w', encoding='utf-8') as f:
        for r in merged_test_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"[OK] Completed dataset merging:")
    print(f"  Whisper train rows: {whisper_train_count:,}, Test rows: {whisper_test_count:,}")
    print(f"  Merged Train rows: {len(merged_train_rows):,}")
    print(f"  Merged Test rows: {len(merged_test_rows):,}")
    print(f"  Saved merged train manifest: {merged_train_manifest}")
    print(f"  Saved merged test manifest: {merged_test_manifest}")
    print("-" * 80)


if __name__ == '__main__':
    main()
