# 🔊 Audio Digital Forensics Analyzer

A comprehensive Python-based tool for performing digital forensic analysis on audio files. This project provides both a command-line interface and an interactive web-based GUI using Streamlit.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Analysis Capabilities](#analysis-capabilities)
- [Output Files](#output-files)
- [Screenshots](#screenshots)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **File Integrity Verification**: SHA-256 and MD5 hash calculation for tampering detection
- **Metadata Extraction**: Extract detailed audio file information
- **Waveform Analysis**: Detect clipping, silence, and amplitude anomalies
- **Edit Detection**: Identify potential splice points and audio manipulations
- **Spectral Analysis**: Frequency content analysis across multiple bands
- **Noise Analysis**: Noise floor calculation and SNR estimation
- **Visual Reports**: Generate comprehensive charts and spectrograms
- **Multiple Interfaces**: Command-line tool and interactive web GUI
- **Export Capabilities**: JSON and text report generation

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone or Download the Project

```bash
git clone https://github.com/yourusername/audio-forensics.git
cd audio-forensics
```

### Step 2: Install Required Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy scipy matplotlib streamlit
```

### Step 3: Verify Installation

```bash
python fgcb.py
```

## 💻 Usage

### Option 1: Streamlit Web Interface (Recommended)

1. **Start the web application:**
   ```bash
   python -m streamlit run app.py
   ```

2. **Open your browser** (usually auto-opens at `http://localhost:8501`)

3. **Upload a WAV file** using the file uploader

4. **Click "Analyze Audio"** to start the forensic analysis

5. **Review results** in the interactive dashboard

6. **Download reports** in JSON or text format

### Option 2: Command-Line Interface

1. **Edit the audio file path** in `fgcb.py`:
   ```python
   audio_file = "your_audio_file.wav"
   ```

2. **Run the analysis:**
   ```bash
   python fgcb.py
   ```

3. **View results** in the `forensics_output/` folder

## 📁 Project Structure

```
audio-forensics/
│
├── app.py                      # Streamlit web interface
├── fgcb.py                     # Command-line forensics tool
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── forensics_output/           # Generated analysis results
│   ├── forensic_analysis.png   # Visual analysis plots
│   ├── forensic_report.json    # Machine-readable report
│   └── forensic_report.txt     # Human-readable report
│
└── sample_audio/               # Sample audio files (optional)
    └── jackhammer.wav
```

## 🔍 Analysis Capabilities

### 1. File Integrity
- **SHA-256 Hash**: Cryptographic hash for file verification
- **MD5 Hash**: Additional hash for integrity checking

### 2. Waveform Analysis
- Mean, max, and RMS amplitude calculations
- Clipping detection (distortion identification)
- Silence detection and percentage calculation

### 3. Edit Detection
- Energy envelope analysis
- Sudden change detection
- Potential splice point identification
- Timestamp tracking of edits

### 4. Spectral Analysis
- Frequency band energy distribution:
  - Sub-bass (20-60 Hz)
  - Bass (60-250 Hz)
  - Low-mid (250-500 Hz)
  - Mid (500-2000 Hz)
  - High-mid (2000-4000 Hz)
  - Presence (4000-6000 Hz)
  - Brilliance (6000-20000 Hz)
- Dominant frequency identification

### 5. Noise Analysis
- Noise floor calculation
- Signal-to-noise ratio (SNR) estimation
- Quality assessment

### 6. Visual Analysis
- Waveform plot
- Spectrogram visualization
- Frequency spectrum graph

## 📊 Output Files

### JSON Report (`forensic_report.json`)
```json
{
    "file_hash": {
        "sha256": "...",
        "md5": "..."
    },
    "metadata": {...},
    "waveform_analysis": {...},
    "edit_detection": {...},
    "spectral_analysis": {...},
    "noise_analysis": {...}
}
```

### Text Report (`forensic_report.txt`)
Human-readable report with all analysis results formatted for easy reading.

### Visual Report (`forensic_analysis.png`)
High-resolution image containing:
- Waveform visualization
- Spectrogram
- Frequency spectrum

## 📸 Screenshots

### Web Interface
![Main Dashboard](screenshots/dashboard.png)
*Upload and analyze audio files through an intuitive interface*

### Analysis Results
![Analysis Results](screenshots/results.png)
*Comprehensive forensic analysis with interactive tabs*

### Visual Reports
![Visual Analysis](screenshots/charts.png)
*Detailed waveform, spectrogram, and frequency analysis*

## 📦 Requirements

Create a `requirements.txt` file with:

```txt
numpy>=1.23.0
scipy>=1.9.0
matplotlib>=3.5.0
streamlit>=1.20.0
```

## 🔧 Troubleshooting

### Issue: "streamlit is not recognized"
**Solution:** Use `python -m streamlit run app.py` instead of `streamlit run app.py`

### Issue: "Error loading audio"
**Solution:** Ensure your audio file is in WAV format. Convert other formats using:
```bash
ffmpeg -i input.mp3 output.wav
```

### Issue: Unicode encoding errors on Windows
**Solution:** The code has been updated to use ASCII-safe characters. Make sure you're using the latest version.

### Issue: Module not found errors
**Solution:** Install all dependencies:
```bash
pip install numpy scipy matplotlib streamlit
```

## 🎯 Use Cases

- **Legal Evidence**: Verify authenticity of audio recordings
- **Media Forensics**: Detect tampering in news or documentary audio
- **Quality Control**: Analyze audio recordings for technical issues
- **Research**: Study audio characteristics and patterns
- **Education**: Learn about digital audio forensics

## 🛠️ Advanced Features

### Batch Processing (Coming Soon)
Process multiple audio files at once.

### Real-time Analysis (Coming Soon)
Analyze audio streams in real-time.

### Machine Learning Detection (Coming Soon)
AI-powered deepfake and manipulation detection.

## 📚 Technical Details

### Audio Format Support
- **Primary**: WAV (PCM)
- **Sampling Rates**: All standard rates (8kHz - 96kHz)
- **Bit Depths**: 16-bit, 24-bit, 32-bit
- **Channels**: Mono and Stereo

### Analysis Algorithms
- **FFT**: Fast Fourier Transform for frequency analysis
- **STFT**: Short-Time Fourier Transform for spectrograms
- **Energy Envelope**: Windowed energy calculation
- **Statistical Analysis**: Mean, RMS, variance calculations

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/audio-forensics](https://github.com/yourusername/audio-forensics)

## 🙏 Acknowledgments

- NumPy and SciPy communities for scientific computing tools
- Matplotlib for visualization capabilities
- Streamlit for the amazing web framework
- Digital forensics research community

## 📖 References

- [Digital Audio Forensics Fundamentals](https://example.com)
- [Audio Signal Processing](https://example.com)
- [Forensic Audio Analysis Best Practices](https://example.com)

---

**Made with ❤️ for Digital Forensics**

*Last Updated: November 2025*
