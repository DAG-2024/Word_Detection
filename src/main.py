import argparse
import os
import json
from typing import List, Tuple
import numpy as np
import soundfile as sf
import torch
from preprocessing import load_audio, reduce_noise, detect_silence, save_audio
from asr import WhisperASR
from detection import ProblemSegmentDetector
from prediction import TextPredictor
from tts import VoiceCloner

class SpeechRestorer:
    def __init__(self, output_dir: str = "output", use_cuda: bool = True):
        """
        Initialize the speech restorer.
        
        Args:
            output_dir: Directory to save output files
            use_cuda: Whether to use CUDA if available
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for each step
        self.steps_dir = os.path.join(output_dir, "steps")
        os.makedirs(self.steps_dir, exist_ok=True)
        
        # Initialize components
        self.asr = WhisperASR(use_cuda=use_cuda)
        self.detector = ProblemSegmentDetector()
        self.predictor = TextPredictor()
        self.voice_cloner = VoiceCloner()
        
    def process_audio(self, input_path: str, language: str = "english"):
        """
        Process an audio file to restore problematic speech segments.
        
        Args:
            input_path: Path to input audio file
            language: Language of the audio
        """
        # Step 1: Load and preprocess audio
        print("Step 1: Loading and preprocessing audio...")
        audio_data, sample_rate = load_audio(input_path)
        save_audio(audio_data, sample_rate, os.path.join(self.steps_dir, "1.1_original.wav"))
        
        print("Reducing noise...")
        audio_data = reduce_noise(audio_data, sample_rate)
        save_audio(audio_data, sample_rate, os.path.join(self.steps_dir, "1.2_denoised.wav"))
        
        # Step 2: Perform ASR
        print("Step 2: Transcribing audio...")
        asr_result = self.asr.transcribe(audio_data, sample_rate, language)
        
        # Save transcription
        with open(os.path.join(self.steps_dir, "2.1_transcription.txt"), "w") as f:
            f.write(asr_result["text"])
        
        # Save word segments
        with open(os.path.join(self.steps_dir, "2.2_word_segments.json"), "w") as f:
            json.dump(asr_result["chunks"], f, indent=2)

        word_segments = self.asr.get_word_segments(asr_result)
        
        # Step 3: Detect problematic segments
        print("Step 3: Detecting problematic segments...")
        low_confidence_segments = self.detector.detect_low_confidence_segments(word_segments)
        grammar_error_segments = self.detector.detect_grammar_errors(asr_result["text"], word_segments)
        
        # Save detection results
        with open(os.path.join(self.steps_dir, "3.1_detection_results.json"), "w") as f:
            json.dump({
                "low_confidence_segments": low_confidence_segments,
                "grammar_error_segments": grammar_error_segments
            }, f, indent=2)
        
        # Train acoustic model on clean segments
        print("Training acoustic model...")
        clean_segments = [(s["start"], s["end"]) for s in word_segments 
                         if s["confidence"] > 0.9]
        self.detector.train_acoustic_model(audio_data, sample_rate, clean_segments)
        acoustic_anomaly_segments = self.detector.detect_acoustic_anomalies(audio_data, sample_rate)

        # Save acoustic detection results
        with open(os.path.join(self.steps_dir, "3.2_acoustic_detection.json"), "w") as f:
            json.dump(acoustic_anomaly_segments, f, indent=2)
        
        # Combine all problematic segments
        problematic_segments = list(set(low_confidence_segments + 
                                      grammar_error_segments + 
                                      acoustic_anomaly_segments))
        problematic_segments.sort(key=lambda x: x[0])
        
        # Save combined problematic segments
        with open(os.path.join(self.steps_dir, "3.3_problematic_segments.json"), "w") as f:
            json.dump(problematic_segments, f, indent=2)
        
        # Step 4: Extract reference audio for voice cloning
        print("Step 4: Extracting reference audio...")
        reference_files = self.voice_cloner.extract_reference_audio(
            audio_data, sample_rate, clean_segments,
            os.path.join(self.steps_dir, "references")
        )
        
        # Step 5: Process each problematic segment
        print("Step 5: Processing problematic segments...")
        final_audio = audio_data.copy()
        segment_predictions = []
        
        for start, end in problematic_segments:
            # Predict replacement text
            predicted_text = self.predictor.predict_for_segment(
                word_segments, start, end
            )
            
            # Save prediction
            segment_predictions.append({
                "start": start,
                "end": end,
                "predicted_text": predicted_text
            })
            
            # Synthesize speech
            synth_audio, synth_sr = self.voice_cloner.synthesize_speech(
                predicted_text, reference_files
            )
            
            # Save synthesized audio
            save_audio(synth_audio, synth_sr, 
                      os.path.join(self.steps_dir, f"5.1_synthesized_{start:.2f}_{end:.2f}.wav"))
            
            # Ensure sample rates match
            if synth_sr != sample_rate:
                import librosa
                synth_audio = librosa.resample(synth_audio, orig_sr=synth_sr, target_sr=sample_rate)
            
            # Replace the segment in the original audio
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            
            # Apply cross-fade
            fade_duration = 0.05  # 50ms fade
            fade_samples = int(fade_duration * sample_rate)
            
            # Create fade windows
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            
            # Apply fade to original audio
            final_audio[start_sample:start_sample + fade_samples] *= fade_out
            final_audio[end_sample - fade_samples:end_sample] *= fade_in
            
            # Apply fade to synthesized audio
            synth_audio[:fade_samples] *= fade_in
            synth_audio[-fade_samples:] *= fade_out
            
            # Insert synthesized audio
            final_audio[start_sample:end_sample] = synth_audio
        
        # Save predictions
        with open(os.path.join(self.steps_dir, "5.2_segment_predictions.json"), "w") as f:
            json.dump(segment_predictions, f, indent=2)
        
        # Save the restored audio
        output_path = os.path.join(self.output_dir, "restored_audio.wav")
        print(f"Saving restored audio to {output_path}...")
        save_audio(final_audio, sample_rate, output_path)
        
        return output_path

def main():
    parser = argparse.ArgumentParser(description="Speech Restoration Program")
    parser.add_argument("input_path", help="Path to input audio file")
    parser.add_argument("--language", default="english", help="Language of the audio")
    parser.add_argument("--output_dir", default="output", help="Output directory")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA even if available")
    
    args = parser.parse_args()
    
    restorer = SpeechRestorer(args.output_dir, use_cuda=not args.no_cuda)
    output_path = restorer.process_audio(args.input_path, args.language)
    
    print(f"Processing complete. Restored audio saved to: {output_path}")
    print(f"Intermediate outputs saved in: {restorer.steps_dir}")

if __name__ == "__main__":
    main() 