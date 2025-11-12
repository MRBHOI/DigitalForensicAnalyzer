"""
Audio Digital Forensics Analysis Tool
Performs comprehensive forensic analysis on audio files including:
- Metadata extraction
- Waveform analysis
- Spectral analysis
- Noise detection
- Audio manipulation detection
- File integrity verification
"""

import os
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AudioForensics:
    def __init__(self, audio_file):
        """Initialize with audio file path"""
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
                # Convert to mono for analysis
                self.audio_mono = np.mean(self.audio_data, axis=1)
            else:
                self.channels = 1
                self.audio_mono = self.audio_data
            
            self.duration = len(self.audio_mono) / self.sample_rate
            
            print(f"[OK] Audio loaded successfully")
            print(f"  Sample Rate: {self.sample_rate} Hz")
            print(f"  Duration: {self.duration:.2f} seconds")
            print(f"  Channels: {self.channels}")
            return True
        except Exception as e:
            print(f"[ERROR] Error loading audio: {e}")
            return False
    
    def calculate_hash(self):
        """Calculate file hash for integrity verification"""
        try:
            sha256_hash = hashlib.sha256()
            md5_hash = hashlib.md5()
            
            with open(self.audio_file, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
                    md5_hash.update(byte_block)
            
            self.report['file_hash'] = {
                'sha256': sha256_hash.hexdigest(),
                'md5': md5_hash.hexdigest()
            }
            print(f"[OK] File hashes calculated")
            print(f"  SHA-256: {sha256_hash.hexdigest()}")
            print(f"  MD5: {md5_hash.hexdigest()}")
        except Exception as e:
            print(f"[ERROR] Error calculating hash: {e}")
    
    def extract_metadata(self):
        """Extract file metadata"""
        try:
            file_stats = os.stat(self.audio_file)
            
            self.report['metadata'] = {
                'filename': os.path.basename(self.audio_file),
                'file_size': file_stats.st_size,
                'created': datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                'sample_rate': int(self.sample_rate),
                'duration': float(self.duration),
                'channels': int(self.channels)
            }
            print(f"[OK] Metadata extracted")
        except Exception as e:
            print(f"[ERROR] Error extracting metadata: {e}")
    
    def analyze_waveform(self):
        """Analyze audio waveform for anomalies"""
        try:
            # Normalize audio
            audio_norm = self.audio_mono / np.max(np.abs(self.audio_mono))
            
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
            
            print(f"[OK] Waveform analyzed")
            print(f"  RMS Level: {rms:.4f}")
            print(f"  Clipping: {clipping_percentage:.2f}%")
            print(f"  Silence: {silence_percentage:.2f}%")
        except Exception as e:
            print(f"[ERROR] Error analyzing waveform: {e}")
    
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
            energy_diff = np.abs(np.diff(energy))
            threshold = np.mean(energy_diff) + 3 * np.std(energy_diff)
            edit_points = np.where(energy_diff > threshold)[0]
            
            self.report['edit_detection'] = {
                'potential_edits': int(len(edit_points)),
                'edit_timestamps': [float(ep * 0.1) for ep in edit_points]
            }
            
            print(f"[OK] Edit detection completed")
            print(f"  Potential edits found: {len(edit_points)}")
            if len(edit_points) > 0 and len(edit_points) <= 5:
                print(f"  At timestamps (seconds): {[f'{ep*0.1:.2f}' for ep in edit_points]}")
        except Exception as e:
            print(f"[ERROR] Error detecting edits: {e}")
    
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
                band_energy[band_name] = float(np.mean(spectrogram[band_mask]))
            
            self.report['spectral_analysis'] = {
                'frequency_bands': band_energy,
                'dominant_frequency': float(frequencies[np.argmax(np.mean(spectrogram, axis=1))])
            }
            
            print(f"[OK] Spectral analysis completed")
            print(f"  Dominant frequency: {self.report['spectral_analysis']['dominant_frequency']:.2f} Hz")
        except Exception as e:
            print(f"[ERROR] Error in spectral analysis: {e}")
    
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
            
            print(f"[OK] Noise analysis completed")
            print(f"  Estimated SNR: {snr:.2f} dB")
        except Exception as e:
            print(f"[ERROR] Error in noise analysis: {e}")
    
    def generate_visualizations(self, output_dir='forensics_output'):
        """Generate forensic visualization plots"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))
            
            # Waveform
            time_axis = np.linspace(0, self.duration, len(self.audio_mono))
            axes[0].plot(time_axis, self.audio_mono, linewidth=0.5)
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
            axes[1].pcolormesh(times, frequencies, 10 * np.log10(spectrogram + 1e-10), 
                              shading='gouraud', cmap='viridis')
            axes[1].set_title('Spectrogram', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Frequency (Hz)')
            axes[1].set_xlabel('Time (seconds)')
            axes[1].set_ylim([0, self.sample_rate/2])
            
            # Frequency spectrum
            fft = np.fft.fft(self.audio_mono)
            frequencies = np.fft.fftfreq(len(fft), 1/self.sample_rate)
            magnitude = np.abs(fft)
            
            positive_freq_idx = frequencies > 0
            axes[2].plot(frequencies[positive_freq_idx], 
                        20 * np.log10(magnitude[positive_freq_idx] + 1e-10))
            axes[2].set_title('Frequency Spectrum', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('Frequency (Hz)')
            axes[2].set_ylabel('Magnitude (dB)')
            axes[2].set_xlim([0, self.sample_rate/2])
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_file = os.path.join(output_dir, 'forensic_analysis.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[OK] Visualizations saved to: {output_file}")
        except Exception as e:
            print(f"[ERROR] Error generating visualizations: {e}")
    
    def generate_report(self, output_dir='forensics_output'):
        """Generate comprehensive forensic report"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Add analysis timestamp
            self.report['analysis_timestamp'] = datetime.now().isoformat()
            self.report['analyst'] = 'Audio Forensics Tool v1.0'
            
            # Save JSON report
            json_file = os.path.join(output_dir, 'forensic_report.json')
            with open(json_file, 'w') as f:
                json.dump(self.report, f, indent=4)
            
            # Generate text report
            text_file = os.path.join(output_dir, 'forensic_report.txt')
            with open(text_file, 'w') as f:
                f.write("=" * 70 + "\n")
                f.write("AUDIO DIGITAL FORENSICS REPORT\n")
                f.write("=" * 70 + "\n\n")
                
                f.write(f"Analysis Date: {self.report['analysis_timestamp']}\n")
                f.write(f"File: {self.report['metadata']['filename']}\n\n")
                
                f.write("-" * 70 + "\n")
                f.write("FILE INTEGRITY\n")
                f.write("-" * 70 + "\n")
                f.write(f"SHA-256: {self.report['file_hash']['sha256']}\n")
                f.write(f"MD5: {self.report['file_hash']['md5']}\n\n")
                
                f.write("-" * 70 + "\n")
                f.write("METADATA\n")
                f.write("-" * 70 + "\n")
                for key, value in self.report['metadata'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                
                f.write("-" * 70 + "\n")
                f.write("WAVEFORM ANALYSIS\n")
                f.write("-" * 70 + "\n")
                for key, value in self.report['waveform_analysis'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                
                f.write("-" * 70 + "\n")
                f.write("EDIT DETECTION\n")
                f.write("-" * 70 + "\n")
                f.write(f"Potential edits: {self.report['edit_detection']['potential_edits']}\n")
                if self.report['edit_detection']['edit_timestamps']:
                    f.write("Edit timestamps (seconds):\n")
                    for ts in self.report['edit_detection']['edit_timestamps']:
                        f.write(f"  - {ts:.2f}s\n")
                f.write("\n")
                
                f.write("-" * 70 + "\n")
                f.write("NOISE ANALYSIS\n")
                f.write("-" * 70 + "\n")
                for key, value in self.report['noise_analysis'].items():
                    f.write(f"{key}: {value}\n")
            
            print(f"[OK] Reports saved to:")
            print(f"  JSON: {json_file}")
            print(f"  Text: {text_file}")
        except Exception as e:
            print(f"[ERROR] Error generating report: {e}")
    
    def run_full_analysis(self, output_dir='forensics_output'):
        """Run complete forensic analysis"""
        print("\n" + "=" * 70)
        print("AUDIO DIGITAL FORENSICS ANALYSIS")
        print("=" * 70 + "\n")
        
        if not self.load_audio():
            return
        
        print("\nPerforming forensic analysis...\n")
        
        self.calculate_hash()
        self.extract_metadata()
        self.analyze_waveform()
        self.detect_edits()
        self.spectral_analysis()
        self.noise_analysis()
        self.generate_visualizations(output_dir)
        self.generate_report(output_dir)
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"\nAll results saved to: {output_dir}/")


def main():
    """Main function to run forensic analysis"""
    print("\nAudio Digital Forensics Tool")
    print("-" * 70)
    
    # Example usage - replace with your audio file
    audio_file = "jackhammer.wav"  # Change this to your audio file path
    
    if not os.path.exists(audio_file):
        print(f"\nWarning: Audio file not found: {audio_file}")
        print("\nTo use this tool:")
        print("1. Place your WAV audio file in the same directory")
        print("2. Update the 'audio_file' variable with your filename")
        print("3. Run the script again")
        return
    
    # Create forensics analyzer
    analyzer = AudioForensics(audio_file)
    
    # Run complete analysis
    analyzer.run_full_analysis(output_dir='forensics_output')


if __name__ == "__main__":
    main()