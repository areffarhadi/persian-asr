import pandas as pd
# remove rows that dont have text in the man
# Load the manifest file
manifest_path = "./filtered_train_file9.csv"  # Replace with your manifest file path
output_path = "./filtered_train_file10.csv"  # Path to save the cleaned file

# Read the manifest into a DataFrame
df = pd.read_csv(manifest_path)

# Remove rows with non-string or NaN values in the 'text' column
df = df[df['text'].apply(lambda x: isinstance(x, str))]

# Drop rows where 'text' is empty, only whitespace, or less than 5 characters
cleaned_df = df[df['text'].str.strip().apply(len) >= 5]

# Save the cleaned DataFrame back to a CSV file
cleaned_df.to_csv(output_path, index=False)

print(f"Cleaned manifest saved to {output_path}.")
