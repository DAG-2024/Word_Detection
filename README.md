# Speech Restoration Program

A Python-based program for restoring problematic speech segments in audio files using advanced machine learning techniques.

## Features

- Automatic Speech Recognition (ASR) using Whisper
- Problematic segment detection using multiple methods:
  - Low ASR confidence
  - Grammar errors
  - Acoustic anomalies
- Text prediction using language models
- Voice cloning and speech synthesis using Coqui TTS
- Seamless audio insertion with cross-fading

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for faster processing)
- See `requirements.txt` for Python dependencies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd speech-restoration
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python src/main.py path/to/input/audio.wav
```

Advanced options:
```bash
python src/main.py path/to/input/audio.wav --language english --output_dir output
```

### Arguments

- `input_path`: Path to the input audio file (required)
- `--language`: Language of the audio (default: "english")
- `--output_dir`: Directory to save output files (default: "output")

## Project Structure

```
speech-restoration/
├── src/
│   ├── preprocessing.py    # Audio loading and preprocessing
│   ├── asr.py             # Automatic Speech Recognition
│   ├── detection.py       # Problematic segment detection
│   ├── prediction.py      # Text prediction
│   ├── tts.py            # Voice cloning and synthesis
│   └── main.py           # Main program
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## How It Works

1. **Preprocessing**: Load and clean the input audio
2. **ASR**: Transcribe the audio and get word-level timestamps
3. **Detection**: Identify problematic segments using multiple methods
4. **Prediction**: Generate replacement text for problematic segments
5. **Synthesis**: Clone the speaker's voice and synthesize new speech
6. **Insertion**: Seamlessly insert the new speech with cross-fading

## Notes

- The program requires a significant amount of computational resources, especially for ASR and TTS
- Processing time depends on the length of the audio and the number of problematic segments
- Results may vary depending on the quality of the input audio and the speaker's voice characteristics

## License

[Your chosen license]

## Acknowledgments

- OpenAI for Whisper
- Hugging Face for Transformers
- Coqui for TTS
- And all other open-source libraries used in this project 