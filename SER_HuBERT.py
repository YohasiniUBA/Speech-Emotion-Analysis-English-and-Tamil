from transformers import pipeline
from pydub import AudioSegment
import os

# Initialize the speech emotion recognition pipeline
classifier = pipeline("audio-classification", model="superb/hubert-large-superb-er")

# Define the duration for each segment (in milliseconds)
segment_duration = 30000  # 30 seconds

# Provide the path to your audio file
audio_file_path = r"D:\Hema\Sem 6\IP Lab\Final codes\english audio files\STE_anxiety.mp3"  # Replace this with the actual path to your audio file

# Load the audio file
audio = AudioSegment.from_file(audio_file_path)

# Split the audio file into segments
segments = [audio[i:i+segment_duration] for i in range(0, len(audio), segment_duration)]

# Predict emotions for each segment
all_predictions = []
for i, segment in enumerate(segments):
    # Save the segment as a temporary file
    segment_path = f"segment_{i}.mp3"
    segment.export(segment_path, format="mp3")
    
    # Perform inference on the segment
    predictions = classifier(segment_path, top_k=28)
    all_predictions.extend(predictions)
    
    # Remove the temporary file
    os.remove(segment_path)

# Initialize a dictionary to store the sum of scores for each unique emotion
sum_scores = {}

# Initialize a dictionary to store the count of scores for each unique emotion
count_scores = {}

# Iterate over the predictions
for prediction in all_predictions:
    emotion = prediction['label']
    score = prediction['score']
    
    # If the emotion is not already in the dictionaries, initialize its sum and count to 0
    if emotion not in sum_scores:
        sum_scores[emotion] = 0
        count_scores[emotion] = 0
    
    # Add the score to the sum of scores for the emotion and increment the count
    sum_scores[emotion] += score
    count_scores[emotion] += 1

# Initialize an empty dictionary to store the mean scores for each unique emotion
mean_scores = {}

# Calculate the mean score for each unique emotion
for emotion, sum_score in sum_scores.items():
    count = count_scores[emotion]
    mean_scores[emotion] = sum_score / count

# Write the output to a file
output_file_path = "SERoutput_anxiety.txt"
with open(output_file_path, "w") as f:
    f.write("Unique predicted emotions with mean scores:\n")
    for emotion, score in mean_scores.items():
        f.write(f"Emotion: {emotion}, Mean Score: {score}\n")

print("Output written to:", output_file_path)
