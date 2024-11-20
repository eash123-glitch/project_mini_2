from flask import Flask, request, render_template, send_from_directory
import os
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
from gtts import gTTS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
from captions import greedy_algorithm  # Assuming this function exists

# Initialize Flask app
app = Flask(__name__)

# Configure the directory for audio files
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['AUDIO_FOLDER'] = 'static/audio'

# Ensure the static/audio folder exists
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

# Load the IndicTrans model for translations
ip = IndicProcessor(inference=True)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)

# Function to translate text into a target language
def translate_text(sentences, src_lang="eng_Latn", tgt_lang="hin_Deva"):
    batch = ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
    batch = tokenizer(batch, padding="longest", truncation=True, max_length=256, return_tensors="pt")

    with torch.inference_mode():
        outputs = model.generate(**batch, num_beams=5, num_return_sequences=1, max_length=256)

    with tokenizer.as_target_tokenizer():
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    outputs = ip.postprocess_batch(outputs, lang=tgt_lang)
    return outputs

# Flask route to handle image upload
@app.route('/', methods=['GET', 'POST'])
def index():
    audio_file = None
    if request.method == 'POST':
        # Get the uploaded image
        image_file = request.files['image']
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(image_path)

        # Generate captions for the uploaded image
        captions = greedy_algorithm(image_path)

        # Translate the captions into Hindi, Telugu, and Marathi
        captions_hindi = translate_text(captions, tgt_lang="hin_Deva")
        captions_telugu = translate_text(captions, tgt_lang="tel_Telu")
        captions_marathi = translate_text(captions, tgt_lang="mar_Deva")

        # Combine all the translated captions into a list
        captions_all = captions + captions_hindi + captions_telugu + captions_marathi
        languages = ['en', 'hi', 'te', 'mr']  # English, Hindi, Telugu, Marathi

        # Convert each caption to speech and save as .mp3 files
        for idx, caption in enumerate(captions_all):
            tts_lang = languages[idx % len(languages)]
            tts = gTTS(text=caption, lang=tts_lang)

            # Save the audio file with a unique name
            audio_filename = f"caption_{idx+1}.mp3"
            audio_path = os.path.join(app.config['AUDIO_FOLDER'], audio_filename)
            tts.save(audio_path)
            print(f"Saved {audio_filename}")

            audio_file = audio_filename  # Save the latest generated audio file to be returned

        return render_template('index.html', audio_file=audio_file)

    return render_template('index.html', audio_file=None)

# Serve static files like audio
@app.route('/static/audio/<filename>')
def download_audio(filename):
    return send_from_directory(app.config['AUDIO_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
