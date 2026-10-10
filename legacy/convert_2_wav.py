import os
from pydub import AudioSegment

# Path to the root directory containing subdirectories with MP3 files
root_directory = 'clips'
# Output directory where WAV files will be saved (mirroring the structure of root_directory)
output_directory = '/media/rf/My Passport/commonVoice_fa'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Process each MP3 file in the directory and its subdirectories
for subdir, _, files in os.walk(root_directory):
    for file in files:
        if file.lower().endswith('.mp3'):
            # Path to the original MP3 file
            mp3_path = os.path.join(subdir, file)
            
            # Create a corresponding subdirectory structure in the output directory
            relative_path = os.path.relpath(subdir, root_directory)
            target_subdir = os.path.join(output_directory, relative_path)
            os.makedirs(target_subdir, exist_ok=True)
            
            # Set the output WAV file path
            wav_filename = os.path.splitext(file)[0] + '.wav'
            wav_path = os.path.join(target_subdir, wav_filename)
            
            try:
                # Load the MP3 file
                audio = AudioSegment.from_file(mp3_path, format="mp3")
                
                # Export as WAV with 18,000 Hz sampling rate
                audio.export(wav_path, format="wav", parameters=["-ar", "18000"])
                
                print(f"Converted {mp3_path} to {wav_path}")
                
            except Exception as e:
                print(f"Error converting {mp3_path}: {e}")

