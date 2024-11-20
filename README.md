# Image Captioning and Audio Playback Flask App

This project is a Flask-based web application that takes an image as input, generates captions for the image using a trained model, translates the captions into Indian languages, and provides audio playback for each caption.

---

## Features

- Upload an image to generate captions using a pre-trained image captioning model.
- Translate captions into Hindi, Telugu, and Marathi using **IndicTrans**.
- Convert captions into audio using **gTTS (Google Text-to-Speech)**.
- Visualize the generated captions and listen to them directly from the web interface.

---

## Folder Structure

```
project/
├── app/
│   ├── captions.py          # Core logic for caption generation, translation, and audio synthesis
│   ├── app.py               # Flask server implementation
│   ├── static/              # Static files like CSS, JavaScript, images
│   │   └── styles.css       # Stylesheet for the web interface
│   └── templates/           # HTML templates
│       └── index.html       # Main UI for uploading images and interacting with the app
├── models/
│   └── image_caption_model/ # Folder containing the pre-trained image captioning model
├── requirements.txt         # List of Python dependencies
├── README.md                # Documentation
└── .gitignore               # Git ignore file
```

---

## Installation

### 1. Prerequisites
- Python 3.8 or higher
- `pip` package manager
- Git

### 2. Clone the Repository
```bash
git clone git@github.com:<your-username>/image-captioning-app.git
cd image-captioning-app
```

### 3. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Start the Flask App
Run the Flask server using the following command:
```bash
python app/app.py
```

### 2. Access the Web Application
Open your browser and go to:
```
http://127.0.0.1:5000
```

### 3. Upload an Image
- Select an image from your local system.
- Click the **"Upload"** button to generate captions.

### 4. Listen to Captions
- View captions in English, Hindi, Telugu, and Marathi.
- Click the play button next to each caption to listen to the audio.

---

## Key Components

### 1. **Caption Generation**
Captions are generated using a trained **image captioning model**. The `greedy_algorithm` function predicts captions from the input image.

### 2. **Translation**
Translations are powered by **IndicTrans**, a state-of-the-art transformer-based multilingual model.

### 3. **Text-to-Speech**
Audio for captions is synthesized using **Google Text-to-Speech (gTTS)**.

---

## Technologies Used

- **Flask**: Web framework
- **TensorFlow/Keras**: For the image captioning model
- **Hugging Face Transformers**: Translation via IndicTrans
- **gTTS**: Audio generation
- **Librosa**: Audio visualization

---

---


## Acknowledgments

- [ai4bharat/IndicTrans2](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M) for translation support.
- [Google Text-to-Speech](https://pypi.org/project/gTTS/) for enabling multilingual audio synthesis.
- TensorFlow and Keras for the image captioning model.

---

## Contribution
Feel free to fork this repository and submit pull requests for improvements or bug fixes. Contributions are welcome!
