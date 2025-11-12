"""
Audio Digital Forensics Analysis Tool - Streamlit Frontend
"""

import streamlit as st
import os
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import json
from datetime import datetime
import warnings
import io
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Audio Forensics Analyzer",
    page_icon="🔊",
    layout="wide"
)

class AudioForensics:
    def __init__(self, audio_file):
        """Initialize with audio file"""
        self.audio_file = audio_file
        self.sample_rate = None
        self.audio_data = None
        self.duration = None
        self.channels = None
        self.report = {}
        
    def load_audio(self):
        """Load audio file and extract basic information"""
        try:
            self.sample_rate, self.audio_data = wavfile.read(self.audio_file)
            
            # Handle stereo/mono
            if len(self.audio_data.shape) > 1:
                self.channels = self.audio_data.shape[1]
                self.audio_mono = np.mean(self.audio_data, axis=1)
            else:
                self.channels = 1
                self.audio_mono = self.audio_data
            
            self.duration = len(self.audio_mono) / self.sample_rate
            return True
        except Exception as e:
            st.error(f"Error loading audio: {e}")
            return False
    
    def calculate_hash(self):
        """Calculate file hash for integrity verification"""
        try:
            sha256_hash = hashlib.sha256()
            md5_hash = hashlib.md5()
            
            # Reset file pointer
            self.audio_file.seek(0)
            for byte_block in iter(lambda: self.audio_file.read(4096), b""):
                sha256_hash.update(byte_block)
                md5_hash.update(byte_block)
            
            self.report['file_hash'] = {
                'sha256': sha256_hash.hexdigest(),
                'md5': md5_hash.hexdigest()
            }
            return True
        except Exception as e:
            st.error(f"Error calculating hash: {e}")
            return False
    
    def extract_metadata(self):
        """Extract file metadata"""
        try:
            self.report['metadata'] = {
                'sample_rate': int(self.sample_rate),
                'duration': float(self.duration),
                'channels': int(self.channels),
                'total_samples': len(self.audio_mono)
            }
            return True
        except Exception as e:
            st.error(f"Error extracting metadata: {e}")
            return False
    
    def analyze_waveform(self):
        """Analyze audio waveform for anomalies"""
        try:
            # Normalize audio
            audio_norm = self.audio_mono / (np.max(np.abs(self.audio_mono)) + 1e-10)
            
            # Calculate statistics
            mean_amplitude = np.mean(np.abs(audio_norm))
            max_amplitude = np.max(np.abs(audio_norm))
            rms = np.sqrt(np.mean(audio_norm**2))
            
            # Detect clipping
            clipping_threshold = 0.99
            clipped_samples = np.sum(np.abs(audio_norm) > clipping_threshold)
            clipping_percentage = (clipped_samples / len(audio_norm)) * 100
            
            # Detect silence
            silence_threshold = 0.01
            silent_samples = np.sum(np.abs(audio_norm) < silence_threshold)
            silence_percentage = (silent_samples / len(audio_norm)) * 100
            
            self.report['waveform_analysis'] = {
                'mean_amplitude': float(mean_amplitude),
                'max_amplitude': float(max_amplitude),
                'rms': float(rms),
                'clipping_percentage': float(clipping_percentage),
                'silence_percentage': float(silence_percentage)
            }
            return True
        except Exception as e:
            st.error(f"Error analyzing waveform: {e}")
            return False
    
    def detect_edits(self):
        """Detect potential audio edits using spectral analysis"""
        try:
            # Calculate energy envelope
            window_size = int(self.sample_rate * 0.1)  # 100ms windows
            energy = []
            
            for i in range(0, len(self.audio_mono) - window_size, window_size):
                segment = self.audio_mono[i:i+window_size]
                energy.append(np.sum(segment**2))
            
            energy = np.array(energy)
            
            # Detect sudden changes in energy (potential edits)
            if len(energy) > 1:
                energy_diff = np.abs(np.diff(energy))
                threshold = np.mean(energy_diff) + 3 * np.std(energy_diff)
                edit_points = np.where(energy_diff > threshold)[0]
            else:
                edit_points = np.array([])
            
            self.report['edit_detection'] = {
                'potential_edits': int(len(edit_points)),
                'edit_timestamps': [float(ep * 0.1) for ep in edit_points[:10]]  # Limit to 10
            }
            return True
        except Exception as e:
            st.error(f"Error detecting edits: {e}")
            return False
    
    def spectral_analysis(self):
        """Perform spectral analysis"""
        try:
            # Calculate spectrogram
            frequencies, times, spectrogram = signal.spectrogram(
                self.audio_mono,
                self.sample_rate,
                nperseg=1024
            )
            
            # Analyze frequency content
            freq_bands = {
                'sub_bass': (20, 60),
                'bass': (60, 250),
                'low_mid': (250, 500),
                'mid': (500, 2000),
                'high_mid': (2000, 4000),
                'presence': (4000, 6000),
                'brilliance': (6000, 20000)
            }
            
            band_energy = {}
            for band_name, (low, high) in freq_bands.items():
                band_mask = (frequencies >= low) & (frequencies <= high)
                if np.any(band_mask):
                    band_energy[band_name] = float(np.mean(spectrogram[band_mask]))
                else:
                    band_energy[band_name] = 0.0
            
            self.report['spectral_analysis'] = {
                'frequency_bands': band_energy,
                'dominant_frequency': float(frequencies[np.argmax(np.mean(spectrogram, axis=1))])
            }
            return True
        except Exception as e:
            st.error(f"Error in spectral analysis: {e}")
            return False
    
    def noise_analysis(self):
        """Analyze noise characteristics"""
        try:
            # Calculate noise floor
            sorted_audio = np.sort(np.abs(self.audio_mono))
            noise_floor = np.mean(sorted_audio[:int(len(sorted_audio) * 0.1)])
            
            # Calculate SNR estimate
            signal_level = np.mean(np.abs(self.audio_mono))
            snr = 20 * np.log10(signal_level / (noise_floor + 1e-10))
            
            self.report['noise_analysis'] = {
                'noise_floor': float(noise_floor),
                'estimated_snr_db': float(snr)
            }
            return True
        except Exception as e:
            st.error(f"Error in noise analysis: {e}")
            return False
    
    def generate_visualizations(self):
        """Generate forensic visualization plots"""
        try:
            fig, axes = plt.subplots(3, 1, figsize=(15, 10))
            
            # Waveform
            time_axis = np.linspace(0, self.duration, len(self.audio_mono))
            axes[0].plot(time_axis, self.audio_mono, linewidth=0.5, color='#1f77b4')
            axes[0].set_title('Waveform Analysis', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Time (seconds)')
            axes[0].set_ylabel('Amplitude')
            axes[0].grid(True, alpha=0.3)
            
            # Spectrogram
            frequencies, times, spectrogram = signal.spectrogram(
                self.audio_mono,
                self.sample_rate,
                nperseg=1024
            )
            pcm = axes[1].pcolormesh(times, frequencies, 10 * np.log10(spectrogram + 1e-10), 
                              shading='gouraud', cmap='viridis')
            axes[1].set_title('Spectrogram', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Frequency (Hz)')
            axes[1].set_xlabel('Time (seconds)')
            axes[1].set_ylim([0, min(self.sample_rate/2, 8000)])
            plt.colorbar(pcm, ax=axes[1], label='Power (dB)')
            
            # Frequency spectrum
            fft = np.fft.fft(self.audio_mono)
            frequencies = np.fft.fftfreq(len(fft), 1/self.sample_rate)
            magnitude = np.abs(fft)
            
            positive_freq_idx = frequencies > 0
            axes[2].plot(frequencies[positive_freq_idx], 
                        20 * np.log10(magnitude[positive_freq_idx] + 1e-10),
                        color='#ff7f0e', linewidth=0.8)
            axes[2].set_title('Frequency Spectrum', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('Frequency (Hz)')
            axes[2].set_ylabel('Magnitude (dB)')
            axes[2].set_xlim([0, min(self.sample_rate/2, 8000)])
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            st.error(f"Error generating visualizations: {e}")
            return None


def main():
    # Header
    st.title("🔊 Audio Digital Forensics Analyzer")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
        This tool performs comprehensive forensic analysis on audio files including:
        
        - File integrity verification (hashing)
        - Metadata extraction
        - Waveform analysis
        - Edit detection
        - Spectral analysis
        - Noise analysis
        - Visual reports
        """)
        
        st.header("Instructions")
        st.markdown("""
        1. Upload a WAV audio file
        2. Click 'Analyze Audio'
        3. Review the forensic report
        4. Download results
        """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Audio File (WAV format)", type=['wav'])
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"✓ File uploaded: {uploaded_file.name}")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            analyze_button = st.button("🔍 Analyze Audio", type="primary", use_container_width=True)
        
        if analyze_button:
            with st.spinner("Analyzing audio file..."):
                # Save uploaded file temporarily
                temp_file = "temp_audio.wav"
                with open(temp_file, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Create analyzer
                analyzer = AudioForensics(uploaded_file)
                
                # Load audio
                if not analyzer.load_audio():
                    st.error("Failed to load audio file")
                    return
                
                # Perform analyses
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Calculating file hash...")
                analyzer.calculate_hash()
                progress_bar.progress(15)
                
                status_text.text("Extracting metadata...")
                analyzer.extract_metadata()
                progress_bar.progress(30)
                
                status_text.text("Analyzing waveform...")
                analyzer.analyze_waveform()
                progress_bar.progress(50)
                
                status_text.text("Detecting edits...")
                analyzer.detect_edits()
                progress_bar.progress(65)
                
                status_text.text("Performing spectral analysis...")
                analyzer.spectral_analysis()
                progress_bar.progress(80)
                
                status_text.text("Analyzing noise...")
                analyzer.noise_analysis()
                progress_bar.progress(90)
                
                status_text.text("Generating visualizations...")
                fig = analyzer.generate_visualizations()
                progress_bar.progress(100)
                
                status_text.empty()
                progress_bar.empty()
                
                st.success("✓ Analysis complete!")
                
                # Display results
                st.markdown("---")
                st.header("Forensic Analysis Report")
                
                # Tabs for different sections
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "📊 Overview", "🔐 File Integrity", "📈 Waveform", 
                    "✂️ Edit Detection", "🎵 Spectral Analysis", "📉 Noise Analysis"
                ])
                
                with tab1:
                    st.subheader("Audio File Overview")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Sample Rate", f"{analyzer.report['metadata']['sample_rate']:,} Hz")
                    with col2:
                        st.metric("Duration", f"{analyzer.report['metadata']['duration']:.2f} sec")
                    with col3:
                        st.metric("Channels", analyzer.report['metadata']['channels'])
                    with col4:
                        st.metric("Total Samples", f"{analyzer.report['metadata']['total_samples']:,}")
                
                with tab2:
                    st.subheader("File Integrity Hashes")
                    st.code(f"SHA-256: {analyzer.report['file_hash']['sha256']}", language="text")
                    st.code(f"MD5:     {analyzer.report['file_hash']['md5']}", language="text")
                    st.info("These hashes can be used to verify file integrity and detect tampering.")
                
                with tab3:
                    st.subheader("Waveform Analysis Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Mean Amplitude", f"{analyzer.report['waveform_analysis']['mean_amplitude']:.4f}")
                        st.metric("Max Amplitude", f"{analyzer.report['waveform_analysis']['max_amplitude']:.4f}")
                        st.metric("RMS Level", f"{analyzer.report['waveform_analysis']['rms']:.4f}")
                    
                    with col2:
                        clipping = analyzer.report['waveform_analysis']['clipping_percentage']
                        silence = analyzer.report['waveform_analysis']['silence_percentage']
                        
                        st.metric("Clipping", f"{clipping:.2f}%", 
                                 delta="Warning" if clipping > 1 else "Good",
                                 delta_color="inverse")
                        st.metric("Silence", f"{silence:.2f}%")
                    
                    if clipping > 1:
                        st.warning("⚠️ Clipping detected! This may indicate audio distortion or recording issues.")
                
                with tab4:
                    st.subheader("Edit Detection Results")
                    edits = analyzer.report['edit_detection']['potential_edits']
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.metric("Potential Edits Found", edits)
                    
                    if edits > 0:
                        st.warning(f"⚠️ {edits} potential edit point(s) detected!")
                        st.markdown("**Edit Timestamps (seconds):**")
                        timestamps = analyzer.report['edit_detection']['edit_timestamps']
                        for i, ts in enumerate(timestamps, 1):
                            st.write(f"{i}. {ts:.2f}s")
                        
                        if edits > 10:
                            st.info(f"Showing first 10 of {edits} detected edits")
                    else:
                        st.success("✓ No obvious edits detected")
                
                with tab5:
                    st.subheader("Spectral Analysis")
                    st.metric("Dominant Frequency", 
                             f"{analyzer.report['spectral_analysis']['dominant_frequency']:.2f} Hz")
                    
                    st.markdown("**Frequency Band Energy Distribution:**")
                    bands = analyzer.report['spectral_analysis']['frequency_bands']
                    
                    band_names = list(bands.keys())
                    band_values = list(bands.values())
                    
                    fig_bands = plt.figure(figsize=(10, 5))
                    plt.bar(band_names, band_values, color='skyblue', edgecolor='navy')
                    plt.xlabel('Frequency Bands')
                    plt.ylabel('Average Energy')
                    plt.title('Energy Distribution Across Frequency Bands')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig_bands)
                
                with tab6:
                    st.subheader("Noise Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Noise Floor", f"{analyzer.report['noise_analysis']['noise_floor']:.6f}")
                    with col2:
                        snr = analyzer.report['noise_analysis']['estimated_snr_db']
                        st.metric("Estimated SNR", f"{snr:.2f} dB",
                                 delta="Good" if snr > 40 else "Poor",
                                 delta_color="normal" if snr > 40 else "inverse")
                    
                    if snr < 30:
                        st.warning("⚠️ Low signal-to-noise ratio detected. Recording may be noisy.")
                    elif snr > 50:
                        st.success("✓ Excellent signal-to-noise ratio!")
                
                # Visualizations
                st.markdown("---")
                st.header("Visual Analysis")
                if fig:
                    st.pyplot(fig)
                
                # Download section
                st.markdown("---")
                st.header("Download Report")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # JSON report
                    analyzer.report['analysis_timestamp'] = datetime.now().isoformat()
                    analyzer.report['filename'] = uploaded_file.name
                    json_str = json.dumps(analyzer.report, indent=4)
                    st.download_button(
                        label="📥 Download JSON Report",
                        data=json_str,
                        file_name=f"forensic_report_{uploaded_file.name}.json",
                        mime="application/json"
                    )
                
                with col2:
                    # Text report
                    text_report = f"""
AUDIO DIGITAL FORENSICS REPORT
{'='*70}

Analysis Date: {analyzer.report['analysis_timestamp']}
File: {uploaded_file.name}

FILE INTEGRITY
{'-'*70}
SHA-256: {analyzer.report['file_hash']['sha256']}
MD5: {analyzer.report['file_hash']['md5']}

METADATA
{'-'*70}
Sample Rate: {analyzer.report['metadata']['sample_rate']} Hz
Duration: {analyzer.report['metadata']['duration']:.2f} seconds
Channels: {analyzer.report['metadata']['channels']}
Total Samples: {analyzer.report['metadata']['total_samples']}

WAVEFORM ANALYSIS
{'-'*70}
Mean Amplitude: {analyzer.report['waveform_analysis']['mean_amplitude']:.4f}
Max Amplitude: {analyzer.report['waveform_analysis']['max_amplitude']:.4f}
RMS Level: {analyzer.report['waveform_analysis']['rms']:.4f}
Clipping: {analyzer.report['waveform_analysis']['clipping_percentage']:.2f}%
Silence: {analyzer.report['waveform_analysis']['silence_percentage']:.2f}%

EDIT DETECTION
{'-'*70}
Potential Edits: {analyzer.report['edit_detection']['potential_edits']}

NOISE ANALYSIS
{'-'*70}
Noise Floor: {analyzer.report['noise_analysis']['noise_floor']:.6f}
Estimated SNR: {analyzer.report['noise_analysis']['estimated_snr_db']:.2f} dB
"""
                    st.download_button(
                        label="📥 Download Text Report",
                        data=text_report,
                        file_name=f"forensic_report_{uploaded_file.name}.txt",
                        mime="text/plain"
                    )
                
                # Clean up
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    
    else:
        st.info("👆 Please upload a WAV audio file to begin analysis")


if __name__ == "__main__":
    main()