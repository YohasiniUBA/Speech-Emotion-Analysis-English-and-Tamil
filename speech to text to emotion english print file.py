import subprocess
import sys
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

# Check if required libraries are installed
try:
    import nltk
    import speech_recognition as sr
    from pydub import AudioSegment
    from pydub.silence import split_on_silence  
    from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline
    import tensorflow as tf

except ImportError:
    print("Some dependencies are missing. Installing required libraries...")
    install_requirements()
    print("Libraries installed successfully.")

# Check if NLTK data is already downloaded, if not, download it
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_path):
    print("Downloading NLTK data...")
    nltk.download('punkt', download_dir=nltk_data_path)
    print("NLTK data downloaded successfully.")

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
            chunk_text = recognizer.recognize_google(chunk_audio_data, language='en')
            text += chunk_text + " "
        except sr.UnknownValueError:
            print("Google Web Speech API could not understand audio.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Web Speech API; {e}")

    return text

# Load tokenizer and model
tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")

# Example usage
audio_file_path = r"D:\Hema\Sem 6\IP Lab\Final codes\english audio files\STE_anxiety.mp3" # Replace with the actual file path

# Convert audio to text
text_result = convert_audio_to_text(audio_file_path)

# Tokenize text
inputs = tokenizer(text_result, return_tensors="tf")

# Get raw model output
outputs = model(**inputs)

# Fetch the emotion labels from the model config
emotion_labels = model.config.id2label

# Interpret the output probabilities for each emotion label
raw_output = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
emotion_probabilities = list(zip(emotion_labels.values(), raw_output))

# Sort emotions based on probabilities in descending order
sorted_emotions = sorted(emotion_probabilities, key=lambda x: x[1], reverse=True)

# Write the text and emotion labels with probabilities to an output file
output_file_path = "output_anxiety.txt"  # Change the file path as needed
with open(output_file_path, 'w') as output_file:
    output_file.write("Text: {}\n\n".format(text_result))
    for emotion, probability in sorted_emotions:
        output_file.write("{}: {:.9f}\n".format(emotion, probability))

