from indicnlp import common
from indicnlp import loader
from indicnlp.tokenize import indic_tokenize
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from googletrans import Translator
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification
import tensorflow as tf

# Load Indic NLP resources
loader.load()

# Load tokenizer and model for emotion analysis
tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")
emotion_labels = model.config.id2label

def convert_audio_to_text(audio_file_path):
    # Load audio file
    audio = AudioSegment.from_file(audio_file_path)

    # Split audio on silence
    chunks = split_on_silence(audio, silence_thresh=-40)

    # Use SpeechRecognition to convert audio chunks to text
    recognizer = sr.Recognizer()
    text = ""

    for chunk in chunks:
        # Convert Pydub AudioSegment to bytes
        chunk_bytes = chunk.raw_data

        # Convert bytes to AudioData using SpeechRecognition's AudioData class
        chunk_audio_data = sr.AudioData(chunk_bytes, chunk.frame_rate, 2)

        try:
            chunk_text = recognizer.recognize_google(chunk_audio_data, language='ta-IN')
            text += chunk_text + " "
        except sr.UnknownValueError:
            print("Google Web Speech API could not understand audio.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Web Speech API; {e}")

    return text

def translate_tamil_to_english(text):
    translator = Translator()
    translation = ""

    # Split the text into smaller chunks (e.g., 200 characters) and translate each chunk
    chunk_size = 200
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunk_translation = translator.translate(chunk, src='ta', dest='en').text
        translation += chunk_translation

    return translation

def analyze_emotions(text_result):
    inputs = tokenizer(text_result, return_tensors="tf")
    outputs = model(**inputs)
    raw_output = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
    emotion_probabilities = list(zip(emotion_labels.values(), raw_output))
    sorted_emotions = sorted(emotion_probabilities, key=lambda x: x[1], reverse=True)
    return sorted_emotions

# Example usage
audio_file_path = r"D:\Hema\Sem 6\IP Lab\Final codes\tamil audio files\Vasu_test_tamil.mp4" # Replace with the actual file path

# Convert audio to text
tamil_text = convert_audio_to_text(audio_file_path)

# Translate Tamil text to English
english_translation = translate_tamil_to_english(tamil_text)

# Analyze emotions from the translated English text
emotion_analysis = analyze_emotions(english_translation)

# Save outputs to a file
with open("tamiloutput_vasu.txt", "w", encoding="utf-8") as f:
    f.write("Tamil Text:\n")
    f.write(tamil_text + "\n\n")
    f.write("English Translation:\n")
    f.write(english_translation + "\n\n")
    f.write("Emotion Analysis:\n")
    for emotion, probability in emotion_analysis:
        f.write(f"{emotion}: {probability:.4f}\n")
