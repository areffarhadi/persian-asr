import pandas as pd

def update_audio_paths(input_path, output_path, directory_address):
    """
    Update audio file paths in the 'path' column by changing '.mp3' to '.wav'
    and prepending a specified directory address.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(input_path, encoding='utf-8')
        
        # Check if 'path' column exists
        if 'path' not in df.columns:
            raise ValueError("Column 'path' not found in CSV file")
        
        # Update 'path' column with directory address and change extension to .wav
        df['path'] = df['path'].apply(lambda x: f"{directory_address}/{x.replace('.mp3', '.wav')}")
        
        # Save the updated DataFrame to a new CSV file
        df.to_csv(output_path, encoding='utf-8', index=False)
        
        print("Audio paths updated successfully.")
        print(f"Updated file saved to: {output_path}")
        
        # Display a sample of the updated paths
        print("Sample of updated paths:")
        print(df['path'].head(10))
        
        return df
    
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    input_file = "test_filtered4_norm.csv"  # Replace with your input CSV file path
    output_file = "test_filtered5_norm.csv"  # Desired output CSV file path
    directory_address = "commonVoice_fa"  # Replace with your actual directory path
    
    # Update paths in the CSV file
    updated_df = update_audio_paths(input_file, output_file, directory_address)
