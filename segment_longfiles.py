import os
import wave
import webrtcvad
from pydub import AudioSegment
from pydub.silence import detect_silence

# Function to split WAV files using VAD and silence detection
def split_wav_file(input_file, output_dir, min_segment_duration=10, max_segment_duration=20, final_min_duration=3):
    # Load the audio file
    audio = AudioSegment.from_wav(input_file)
    sample_rate = audio.frame_rate

    # Detect silence using pydub
    silences = detect_silence(audio, min_silence_len=300, silence_thresh=audio.dBFS - 14)
    
    # Convert silence intervals to milliseconds for indexing
    silence_intervals = [(start / 1000, end / 1000) for start, end in silences]

    segments = []
    current_start = 0.0

    while current_start < len(audio) / 1000:
        # Find the next silence point within the desired range
        segment_end = None
        for start, end in silence_intervals:
            if current_start + min_segment_duration <= start <= current_start + max_segment_duration:
                segment_end = start
                break

        # If no suitable silence point is found, use max_segment_duration
        if segment_end is None:
            segment_end = min(current_start + max_segment_duration, len(audio) / 1000)

        # Extract the segment
        segment = audio[current_start * 1000:segment_end * 1000]
        segments.append(segment)
        current_start = segment_end

    # Handle the last segment
    if len(segments) > 1 and len(segments[-1]) / 1000 < final_min_duration:
        segments[-2] += segments[-1]
        segments.pop()

    # Save the segments
    base_name = os.path.basename(input_file).split(".")[0]
    for i, segment in enumerate(segments):
        output_path = os.path.join(output_dir, f"{base_name}_part{i + 1}.wav")
        segment.export(output_path, format="wav")
        print(f"Saved segment: {output_path}")

# Main function to process all WAV files in a folder
def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        if file.endswith(".wav"):
            input_path = os.path.join(input_folder, file)
            print(f"Processing: {input_path}")
            split_wav_file(input_path, output_folder)

# Replace these paths with your input and output folder paths
input_folder = "farsdat_ctc_long"
output_folder = "farsdat_ctc_long2"

process_folder(input_folder, output_folder)

