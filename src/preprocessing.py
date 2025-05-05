import librosa
import soundfile as sf
import numpy as np
from typing import Tuple, Optional
import noisereduce as nr

def load_audio(audio_path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Load audio file and return audio data and sample rate.
    
    Args:
        audio_path: Path to the audio file
        target_sr: Target sample rate. If None, uses original sample rate
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    audio_data, sample_rate = librosa.load(audio_path, sr=target_sr)
    return audio_data, sample_rate

def reduce_noise(audio_data: np.ndarray, sample_rate: int, 
                noise_duration: float = 0.5) -> np.ndarray:
    """
    Reduce noise in audio using noisereduce.
    
    Args:
        audio_data: Audio data array
        sample_rate: Sample rate of the audio
        noise_duration: Duration of noise sample to use (in seconds)
        
    Returns:
        Cleaned audio data
    """
    # Use the first noise_duration seconds as noise sample
    noise_sample = audio_data[:int(noise_duration * sample_rate)]
    
    # Apply noise reduction
    reduced_noise = nr.reduce_noise(
        y=audio_data,
        sr=sample_rate,
        y_noise=noise_sample,
        stationary=True
    )
    
    return reduced_noise

def detect_silence(audio_data: np.ndarray, top_db: float = 40) -> np.ndarray:
    """
    Detect non-silent intervals in audio.
    
    Args:
        audio_data: Audio data array
        top_db: Threshold in decibels below reference
        
    Returns:
        Array of intervals (start, end) in samples
    """
    non_silent_intervals = librosa.effects.split(audio_data, top_db=top_db)
    return non_silent_intervals

def save_audio(audio_data: np.ndarray, sample_rate: int, output_path: str):
    """
    Save audio data to file.
    
    Args:
        audio_data: Audio data array
        sample_rate: Sample rate
        output_path: Path to save the audio file
    """
    sf.write(output_path, audio_data, sample_rate) 