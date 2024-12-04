import os
import librosa
import soundfile as sf
from tqdm import tqdm

root_directory = './Farsi_Eslahatnews_audio'
output_directory = '/home/rf/farsi_data_prepr/telegram_data/eslahat'



# First, count total number of MP3 files for progress tracking
def count_mp3_files(directory):
    total = 0
    for subdir, _, files in os.walk(directory):
        total += sum(1 for file in files if file.lower().endswith('.mp3'))
    return total

# Function to convert audio to 16kHz mono WAV
def convert_to_16khz_mono_wav(input_path, output_path):
    try:
        # Load audio file with librosa
        audio, original_sr = librosa.load(input_path, mono=True, sr=None)
        
        # Resample to 16kHz
        audio_16khz = librosa.resample(audio, orig_sr=original_sr, target_sr=16000)
        
        # Write to wav file (16-bit PCM)
        sf.write(output_path, audio_16khz, 16000, subtype='PCM_16')
        return True
    except Exception as e:
        print(f"\nError converting {input_path}: {e}")
        return False

# Create output directory
os.makedirs(output_directory, exist_ok=True)

# Count total files
total_files = count_mp3_files(root_directory)

# Use tqdm to show progress
with tqdm(total=total_files, desc="Converting MP3 to WAV", unit="file") as pbar:
    for subdir, _, files in os.walk(root_directory):
        for file in files:
            if file.lower().endswith('.mp3'):
                # Input file path
                mp3_path = os.path.join(subdir, file)
                
                # Create corresponding output subdirectory
                relative_path = os.path.relpath(subdir, root_directory)
                target_subdir = os.path.join(output_directory, relative_path)
                os.makedirs(target_subdir, exist_ok=True)
                
                # Output file path
                wav_filename = os.path.splitext(file)[0] + '.wav'
                wav_path = os.path.join(target_subdir, wav_filename)
                
                # Convert file
                success = convert_to_16khz_mono_wav(mp3_path, wav_path)
                
                # Update progress bar
                pbar.update(1)

print("\nConversion complete!")
