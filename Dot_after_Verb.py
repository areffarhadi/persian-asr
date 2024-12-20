import os
import re

def load_verbs(verbs_file_path):
    """
    Load a list of Persian verbs from a text file.

    :param verbs_file_path: Path to the text file containing verbs (one verb per line).
    :return: Set of Persian verbs with at least three characters.
    """
    with open(verbs_file_path, 'r', encoding='utf-8') as file:
        verbs = {line.strip() for line in file if len(line.strip()) >= 3}
    return verbs

def add_dots_after_verbs(source_dir, verbs, output_dir):
    """
    Add a dot after each Persian verb in all text files in the directory.

    :param source_dir: Path to the directory containing Persian text files.
    :param verbs: Set of Persian verbs.
    :param output_dir: Path to save the modified text files.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(source_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                # Replace verbs with the verb followed by a dot
                words = content.split()
                modified_words = []
                last_dot_index = -4  # To ensure at least 3 words between dots
                for i, word in enumerate(words):
                    if word in verbs and (i - last_dot_index) > 3:
                        # Check if a dot exists in the previous or next 3 words
                        surrounding_words = words[max(0, i-3):i+4]
                        if not any('.' in w for w in surrounding_words):
                            modified_words.append(word + '.')
                            last_dot_index = i
                        else:
                            modified_words.append(word)
                    else:
                        modified_words.append(word)

                modified_content = ' '.join(modified_words)

                # Ensure a dot at the end of the file if not present
                if not modified_content.endswith('.'): 
                    modified_content += '.'

                # Save the modified content to the output directory
                output_file_path = os.path.join(output_dir, filename)
                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(modified_content)

                print(f"Processed and saved: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


# Example usage
verbs_file = "Persian_verb"  # Path to the file containing Persian verbs
source_directory = "/home/rf/farsi_data_prepr/test_test/transcriptions_chunk"  # Directory with input text files
output_directory = "/home/rf/farsi_data_prepr/test_test/transcriptions_chunk2"  # Directory to save modified text files

# Load Persian verbs and process files
persian_verbs = load_verbs(verbs_file)
add_dots_after_verbs(source_directory, persian_verbs, output_directory)

