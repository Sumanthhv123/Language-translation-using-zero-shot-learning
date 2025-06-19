# Language-translation-using-zero-shot-learning

A multilingual speech and text translation web app built with Gradio, Hugging Face Transformers, Wav2Vec2, and gTTS. This app allows users to:

🎤 Transcribe English audio

🌐 Translate English text or speech into multiple languages using zero-shot translation

🔊 Listen to translated speech using Text-to-Speech (TTS)

🔧 Features Zero-Shot Translation using Facebook's M2M100 model

Audio Transcription using Wav2Vec2

Multilingual Text-to-Speech with Google TTS

Gradio Web Interface with tabs for:

Text Translation

Audio Transcription + Translation

🧰 Tech Stack transformers

torchaudio

torch

gradio

gtts

facebook/m2m100_418M for translation

facebook/wav2vec2-large-960h for transcription

📦 Installation Clone this repository

bash Copy Edit git clone https://github.com/yourusername/speech-text-translation-app.git cd speech-text-translation-app Install dependencies Make sure Python 3.7+ is installed.

bash Copy Edit pip install -r requirements.txt Run the app

bash Copy Edit python app.py 📄 Usage

Text Translation Input English text.
Select a target language.

Get translated text and hear it spoken via gTTS.

Audio Transcription & Translation Upload an audio file (.wav recommended).
App transcribes the speech using Wav2Vec2.

Translates the transcribed text into the selected target language.

🌐 Supported Languages French

German

Spanish

Italian

Russian

Chinese

Japanese

Hindi

Urdu
