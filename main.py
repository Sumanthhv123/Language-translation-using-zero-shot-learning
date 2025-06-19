!pip install gradio torchaudio transformers torch gtts
import gradio as gr
import torchaudio
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from gtts import gTTS
import os

# Suppress Hugging Face symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Define supported languages with their ISO 639-1 codes and gTTS language codes
SUPPORTED_LANGUAGES = {
    "French": {"code": "fr", "gtts_code": "fr"},
    "German": {"code": "de", "gtts_code": "de"},
    "Spanish": {"code": "es", "gtts_code": "es"},
    "Italian": {"code": "it", "gtts_code": "it"},
    "Russian": {"code": "ru", "gtts_code": "ru"},
    "Chinese": {"code": "zh", "gtts_code": "zh"},
    "Japanese": {"code": "ja", "gtts_code": "ja"},
    "Hindi": {"code": "hi", "gtts_code": "hi"},
    "Urdu": {"code": "ur", "gtts_code": "ur"},
}

# Load the zero-shot translation model (M2M-100)
try:
    translation_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    translation_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    print("Zero-shot translation model loaded successfully!")
except Exception as e:
    print(f"Error loading translation model: {e}")
    translation_model = None
    translation_tokenizer = None

# Load the transcription model (Wav2Vec 2.0)
try:
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to("cpu")
    print("Transcription model loaded successfully!")
except Exception as e:
    print(f"Error loading transcription model: {e}")

# Function to transcribe audio
def transcribe_audio(audio_file):
    try:
        # Load the audio file
        waveform, sample_rate = torchaudio.load(audio_file)

        # Resample if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # Preprocess the audio
        input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=sample_rate).input_values

        # Perform transcription
        with torch.no_grad():
            logits = model(input_values).logits

        # Decode the logits to text
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])

        return transcription
    except Exception as e:
        return f"Error during transcription: {e}"

# Zero-shot translation function
def zero_shot_translate(text, src_lang, tgt_lang):
    if not text.strip():
        return "Error: Input text is empty."

    try:
        # Set the source language
        translation_tokenizer.src_lang = src_lang

        # Tokenize the input text
        encoded_inputs = translation_tokenizer(text, return_tensors="pt")

        # Generate translation
        generated_tokens = translation_model.generate(
            **encoded_inputs,
            forced_bos_token_id=translation_tokenizer.get_lang_id(tgt_lang)
        )

        # Decode the translated text
        translated_text = translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translated_text
    except Exception as e:
        return f"Error during translation: {e}"

# Function to translate text and generate audio
def translate_gradio(text, target_language):
    if not text.strip():
        return "Error: Input text is empty.", None

    try:
        # Translate the text
        if target_language not in SUPPORTED_LANGUAGES:
            return f"Error: {target_language} is not supported.", None

        tgt_lang = SUPPORTED_LANGUAGES[target_language]["code"]
        translated_text = zero_shot_translate(text, "en", tgt_lang)

        # Check if translation was successful
        if translated_text.startswith("Error"):
            return translated_text, None

        # Generate audio using gTTS
        tts = gTTS(translated_text, lang=SUPPORTED_LANGUAGES[target_language]["gtts_code"])
        audio_file = "translated_audio.mp3"
        tts.save(audio_file)

        return translated_text, audio_file
    except Exception as e:
        return f"Error during translation or audio generation: {e}", None

# Transcription and translation function
def transcribe_and_translate(audio_file, target_language):
    if not audio_file:
        return "Error: No audio file uploaded.", None

    try:
        # Transcribe the audio
        transcription = transcribe_audio(audio_file)
        if transcription.startswith("Error"):
            return transcription, None

        # Translate the transcribed text
        if target_language not in SUPPORTED_LANGUAGES:
            return f"Error: {target_language} is not supported.", None

        tgt_lang = SUPPORTED_LANGUAGES[target_language]["code"]
        translated_text = zero_shot_translate(transcription, "en", tgt_lang)

        # Check if translation was successful
        if translated_text.startswith("Error"):
            return translated_text, None

        return transcription, translated_text
    except Exception as e:
        return f"Error during transcription or translation: {e}", None

# Combine all interfaces into a single app
with gr.Blocks() as app:
    with gr.Tab("Text Translation"):
        gr.Interface(
            fn=translate_gradio,
            inputs=[
                gr.Textbox(label="Enter text"),
                gr.Dropdown(list(SUPPORTED_LANGUAGES.keys()), label="Target Language")
            ],
            outputs=[
                gr.Textbox(label="Translated text"),
                gr.Audio(label="Translated audio", type="filepath")
            ],
            title="Text Translation",
            description="Translate English text to multiple languages using zero-shot translation and hear the translation."
        )

    with gr.Tab("Audio Transcription and Translation"):
        gr.Interface(
            fn=transcribe_and_translate,
            inputs=[
                gr.Audio(label="Upload audio file", type="filepath"),
                gr.Dropdown(list(SUPPORTED_LANGUAGES.keys()), label="Target Language")
            ],
            outputs=[
                gr.Textbox(label="Transcribed text"),
                gr.Textbox(label="Translated text")
            ],
            title="Audio Transcription and Translation",
            description="Upload an audio file to transcribe it to text and translate it into another language."
        )

# Launch the app
app.launch(debug=True)
