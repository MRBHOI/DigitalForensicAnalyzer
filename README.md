Digital Audio Forensics Helper

This repository contains a small utility for basic digital audio forensics tasks:
- SHA256 file integrity checks
- Basic file and audio metadata extraction
- Signal-level analysis and spectrogram generation

Files added
- `fgcb.py` — main script (functions exported for testing): compute SHA256, extract metadata, create spectrogram.
- `requirements.txt` — Python dependencies.
- `tests/test_fgcb.py` — small pytest that generates a tiny WAV file and validates core functions (see note).

Quick setup (Windows cmd.exe)

1) Create a virtual environment (recommended):

    python -m venv .venv
    .venv\Scripts\activate

2) Install dependencies:

    pip install -r requirements.txt

Notes and platform requirements
- `pydub` requires `ffmpeg` available on PATH to read many compressed formats (MP3, AAC, etc.).
  Download from https://ffmpeg.org/ and add the binary directory to your PATH.
- `soundfile` (used by librosa) depends on the system `libsndfile` library; on Windows installing the `soundfile` wheel will usually include the needed binaries, but if you hit errors consider installing via conda or from the binary distributions.

Usage examples

Analyze a file and save spectrogram:

    python fgcb.py -i suspect_recording.mp3 -o spectrogram.png

Run tests (after installing dependencies):

    pytest -q

If you don't want to install system libraries, you can still use the SHA256 function or the helper metadata functions for files already readable by the Python standard library.

Caveats
- This tool is for helping with exploratory audio forensics only. It does not provide court-ready provenance or cryptographic acquisition workflows.
- For robust forensic workflows, record original acquisition hashes and chain-of-custody metadata at capture time.

