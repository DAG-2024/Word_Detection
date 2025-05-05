from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import torch
from typing import Dict, List, Tuple
import numpy as np
import librosa
import os

class WhisperASR:
    def __init__(self, model_name: str = "openai/whisper-large-v3", 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 use_cuda: bool = True):
        """
        Initialize Whisper ASR model.
        
        Args:
            model_name: Name of the Whisper model to use
            device: Device to run the model on ("cuda" or "cpu")
            use_cuda: Whether to use CUDA if available (overrides device if True and CUDA is available)
        """
        if use_cuda and torch.cuda.is_available():
            device = "cuda"
            print("Using CUDA for ASR")
        else:
            device = "cpu"
            print("Using CPU for ASR")
            
        # Initialize processor and model directly for more control
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        self.device = device
        self.target_sr = 16000  # Whisper expects 16kHz audio
        
    def transcribe(self, audio_data: np.ndarray, sample_rate: int,
                  language: str = "english") -> Dict:
        """
        Transcribe audio data using Whisper.
        
        Args:
            audio_data: Audio data array
            sample_rate: Sample rate of the audio
            language: Language of the audio
            
        Returns:
            Dictionary containing transcription results
        """
        # Convert audio data to float32 if needed
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            
        # Normalize audio
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Resample to 16kHz if needed
        if sample_rate != self.target_sr:
            print(f"Resampling audio from {sample_rate}Hz to {self.target_sr}Hz...")
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=sample_rate, 
                target_sr=self.target_sr
            )
            sample_rate = self.target_sr
        
        # Process audio
        input_features = self.processor(
            audio_data, 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        ).input_features.to(self.device)
        
        # Generate transcription with timestamps
        predicted_ids = self.model.generate(
            input_features,
            language=language,
            task="transcribe",
            return_timestamps=True,
            return_dict_in_generate=True
        )
        
        # Get the generated sequence and ensure it's the right shape
        if isinstance(predicted_ids, dict):
            sequence = predicted_ids["sequences"]
            # Get segments if available
            if "segments" in predicted_ids:
                segments = predicted_ids["segments"]
                # Process segments to get word-level timestamps
                chunks = []
                for segment in segments:
                    if "text" in segment and "timestamp" in segment:
                        chunks.append({
                            "text": segment["text"].strip(),
                            "timestamp": (segment["timestamp"][0], segment["timestamp"][1])
                        })
                return {
                    "text": self.processor.batch_decode(sequence, skip_special_tokens=True)[0],
                    "chunks": chunks
                }
        else:
            sequence = predicted_ids
            
        # Convert tensor to numpy array and then to list for processing
        sequence = sequence.cpu().numpy().tolist()
        
        # Ensure sequence is 2D (batch_size, sequence_length)
        if isinstance(sequence[0], int):
            sequence = [sequence]
        
        # Decode the transcription
        transcription = self.processor.batch_decode(
            sequence, 
            skip_special_tokens=False
        )[0]
        
        # Get timestamps with detailed decoding
        timestamps = self.processor.batch_decode(
            sequence, 
            skip_special_tokens=False,
            decode_with_timestamps=True,
            return_timestamps=True
        )[0]
        
        print("Raw transcription:", transcription)
        print("Raw timestamps:", timestamps)
        
        # Process the output to get word-level timestamps
        chunks = []
        current_text = ""
        current_start = None
        current_end = None
        
        # Split the timestamps string into tokens
        tokens = timestamps.split()
        
        for i, token in enumerate(tokens):
            if token.startswith("<|") and token.endswith("|>"):
                # If we have accumulated text and have a start time, save the chunk
                if current_text and current_start is not None:
                    chunks.append({
                        "text": current_text.strip(),
                        "timestamp": (current_start, current_end)
                    })
                    current_text = ""
                
                # Extract timestamp
                try:
                    time = float(token.strip("<|>"))
                    if current_start is None:
                        current_start = time
                    current_end = time
                except ValueError:
                    continue
            else:
                # Add the word to the current text
                current_text += " " + token
                
                # If this is the last token and have a start time, save the final chunk
                if i == len(tokens) - 1 and current_start is not None:
                    chunks.append({
                        "text": current_text.strip(),
                        "timestamp": (current_start, current_end)
                    })
        
        print("Processed chunks:", chunks)
        
        return {
            "text": transcription,
            "chunks": chunks
        }
    
    def get_word_segments(self, transcription_result: Dict) -> List[Dict]:
        """
        Extract word segments from transcription result.
        
        Args:
            transcription_result: Result from transcribe method
            
        Returns:
            List of word segments with text, start time, end time, and confidence
        """
        word_segments = []
        
        if "chunks" in transcription_result:
            for chunk in transcription_result["chunks"]:
                if "timestamp" in chunk:
                    start, end = chunk["timestamp"]
                    word_segments.append({
                        "text": chunk["text"].strip(),
                        "start": start,
                        "end": end,
                        "confidence": 1.0  # Whisper doesn't provide confidence scores
                    })
                    
        return word_segments 