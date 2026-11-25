#!/usr/bin/env python3
"""
Extract text using Omi Med STT via uv, using the correct CLI syntax.
"""
import argparse, json, subprocess
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-path", required=True, help="Path to omi-med-stt-v1-q8_0.gguf")
    ap.add_argument("--runtime", default="cpp", choices=["auto", "nemo", "mlx", "cpp"], 
                    help="Use 'cpp' for GGUF files")
    a = ap.parse_args()

    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    wavs = sorted(Path(a.audio_dir).glob("*.wav"))

    failed = []
    for i, wav in enumerate(wavs, 1):
        out_path = Path(a.out_dir) / f"{wav.stem}.json"
        if out_path.exists():
            continue

        # CORRECTED COMMAND: uses 'transcribe' subcommand, 'cpp' runtime, and '--gguf-file'
        cmd = [
            "uv", "run", "--python", "3.10", "--with", "omi-med-stt",
            "omi-med-stt", "transcribe", str(wav),
            "--runtime", a.runtime,
            "--gguf-file", a.model_path,
            "--json"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
        except subprocess.CalledProcessError as e:
            print(f"[{i}/{len(wavs)}] FAILED on {wav.name}: {e.stderr[:200]}")
            failed.append(wav.name)
            with open(out_path, "w") as f:
                json.dump({"text": "", "words": []}, f)

        if i % 50 == 0:
            print(f"Processed {i}/{len(wavs)}")

    print(f"Done. Failed: {len(failed)}")
    if failed:
        with open(f"{a.out_dir}/_failed_omi.txt", "w") as f:
            f.write("\n".join(failed))

if __name__ == "__main__":
    main()