import os
import pandas as pd
import random
import os
import torch
import numpy as np
import pypianoroll


root_dir = '/content/drive/MyDrive/682 Project'
data_dir = root_dir + '/Lakh Piano Dataset/lpd_17/lpd_17_cleansed'
# Local path constants
DATA_PATH = 'data'
RESULTS_PATH = 'results'

# Utility functions for retrieving paths
def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

def msd_id_to_mp3(msd_id):
    """Given an MSD ID, return the path to the corresponding mp3"""
    return os.path.join(DATA_PATH, 'msd', 'mp3',
                        msd_id_to_dirs(msd_id) + '.mp3')

def msd_id_to_h5(h5):
    """Given an MSD ID, return the path to the corresponding h5"""
    return os.path.join(RESULTS_PATH, 'lmd_matched_h5',
                        msd_id_to_dirs(msd_id) + '.h5')

def get_midi_npz_path(msd_id, midi_md5):
    return os.path.join(data_dir,
                        msd_id_to_dirs(msd_id), midi_md5 + '.npz')

def get_midi_path(msd_id, midi_md5, kind):
    """Given an MSD ID and MIDI MD5, return path to a MIDI file.
    kind should be one of 'matched' or 'aligned'. """
    return os.path.join(RESULTS_PATH, 'lmd_{}'.format(kind),
                        msd_id_to_dirs(msd_id), midi_md5 + '.mid')

def get_random_song_ids(file_paths, num_songs=1000):
    """
    Generalized function to read song IDs from multiple files and randomly select a given number of IDs.

    :param file_paths: List of file paths to read song IDs from
    :param num_songs: Number of random song IDs to select (default is 1000)
    :return: A list of randomly selected song IDs
    """
    song_ids = []

    # Loop through the provided file paths and read song IDs
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as file:
                song_ids.extend([line.strip() for line in file.readlines()])
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
        except Exception as e:
            print(f"An error occurred while reading '{file_path}': {e}")

    # Randomly select the specified number of song IDs
    if len(song_ids) > 0:
        return random.sample(song_ids, min(num_songs, len(song_ids)))
    else:
        print("No song IDs were found in the provided files.")
        return []

def create_combined_pianorolls(train_ids, combined_pianorolls_path='rock1000_acoustic_combined_pianorolls.pt'):
    """
    Generate and save combined pianorolls to a file.

    :param train_ids: List of MSD file names (train IDs)
    :param combined_pianorolls_path: Path to the file where combined pianorolls are saved/loaded from
    :return: List of combined pianorolls tensors
    """

    combined_pianorolls = []
    
    # Initialize parts names
    track_names = [
        'Drums', 'Piano', 'Chromatic Percussion', 'Organ', 'Guitar', 'Bass',
        'Strings', 'Ensemble', 'Brass', 'Reed', 'Pipe', 'Synth Lead', 'Synth Pad',
        'Synth Effects', 'Ethnic', 'Percussive', 'Sound Effects'
    ]

    # Open the cleansed ids - cleansed file ids : msd ids
    cleansed_ids = pd.read_csv(os.path.join(root_dir, 'Lakh Piano Dataset', 'cleansed_ids.txt'), delimiter = '    ', header = None)
    lpd_to_msd_ids = {a:b for a, b in zip(cleansed_ids[0], cleansed_ids[1])}
    msd_to_lpd_ids = {a:b for a, b in zip(cleansed_ids[1], cleansed_ids[0])}
    
    # Generate the pianorolls
    for msd_file_name in train_ids:
        lpd_file_name = msd_to_lpd_ids[msd_file_name]
        
        # Get the NPZ path
        npz_path = get_midi_npz_path(msd_file_name, lpd_file_name)
        
        # Load the multitrack MIDI
        multitrack = pypianoroll.load(npz_path)
        multitrack.set_resolution(4).pad_to_same()
        
        # Initialize parts for all tracks
        parts = {track_name: None for track_name in track_names}
        
        # Initialize an empty array for missing parts
        empty_array = None
        
        # Process each track in the multitrack
        for track in multitrack.tracks:
            if track.pianoroll.shape[0] > 0 and empty_array is None:
                empty_array = np.zeros_like(track.pianoroll)
            parts[track.name] = track.pianoroll if track.pianoroll.shape[0] > 0 else empty_array.copy()
        
        # Stack all parts together
        combined_pianoroll = torch.stack([torch.tensor(parts[track_name], dtype=torch.uint8) for track_name in parts])
        
        # Append the combined pianoroll to the list
        combined_pianorolls.append(combined_pianoroll)
    
    # Save the combined pianorolls to a file
    torch.save(combined_pianorolls, combined_pianorolls_path)
    
    return combined_pianorolls


def generate_dataset(combined_pianorolls, file_path="dataset.txt"):
    # Mapping of track indices to instrument names
    track_mapping = {
        0: 'D',    # Drums
        1: 'P',    # Piano
        2: 'C',    # Chromatic Percussion
        3: 'O',    # Organ
        4: 'G',    # Guitar
        5: 'B',    # Bass
        6: 'S',    # Strings
        7: 'E',    # Ensemble
        8: 'R',    # Brass
        9: 'r',    # Reed
        10: 'Pp',  # Pipe
        11: 'L',   # Synth Lead
        12: 'Pad', # Synth Pad
        13: 'Fx',  # Synth Effects
        14: 'Eth', # Ethnic
        15: 'Per', # Percussive
        16: 'X',   # Sound Effects
    }

    # Open the file in write mode before processing the sequences
    with open(file_path, "w") as file:
        # Iterate over all combined pianorolls (first dimension)
        for pianoroll in combined_pianorolls:
            tokenized_sequences = []

            # Iterate over all sequences (second dimension)
            for i in range(pianoroll.shape[1]):  # Loop over all time steps
                tokens = []
                for track_idx in range(pianoroll.shape[0]):  # Loop over all tracks
                    track_label = track_mapping[track_idx]  # Get the instrument label for this track
                    notes = pianoroll[track_idx, i, :]  # Get the notes for this track at time step i

                    # Find indices of active notes
                    active_note_indices = torch.nonzero(notes).flatten().tolist()  # Get active note indices as a list

                    # Create tokens for each active note
                    for note_idx in active_note_indices:
                        token = f"{track_label}{note_idx}"
                        tokens.append(token)

                # Join tokens for the current time step into a single string (space-separated)
                tokenized_sequences.append(" ".join(tokens))

            # Join the time-step-based token sequences with <beat> separator
            sequence = " <beat> ".join(tokenized_sequences)

            # Add start and end of token markers
            sequence = "<SOT> " + sequence + " <EOT>"

            # Write the sequence to the file followed by a new line
            file.write(sequence + "\n")

    # Confirm that the dataset.txt was written successfully
    print(f"Tokenized sequences have been saved to {file_path}.")

def max_tokens_in_line(file_path='dataset.txt'):
    """
    Calculate the maximum number of tokens in a single line of the dataset file
    and print that line.

    :param file_path: Path to the dataset file (default is 'dataset.txt')
    :return: Maximum number of tokens in any single line, and print that line
    """
    max_tokens = 0  # To store the maximum number of tokens found
    line_with_max_tokens = ""  # To store the line with the maximum tokens

    try:
        with open(file_path, 'r') as f:
            for line in f:
                tokens = line.strip().split()  # Split line into tokens by whitespace
                num_tokens = len(tokens)  # Count the tokens in the current line
                if num_tokens > max_tokens:
                    max_tokens = num_tokens  # Update max_tokens
                    line_with_max_tokens = line.strip()  # Store the line with max tokens

        print(f"Maximum number of tokens in a single line: {max_tokens}")
        print(f"Line with maximum tokens: {line_with_max_tokens}")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def average_tokens_per_line(file_path='dataset.txt'):
    """
    Calculate and print the average number of tokens per line in the dataset file.

    :param file_path: Path to the dataset file (default is 'dataset.txt')
    :return: The average number of tokens per line
    """
    total_tokens = 0  # To store the total number of tokens across all lines
    total_lines = 0  # To count the number of lines in the file

    try:
        with open(file_path, 'r') as f:
            for line in f:
                tokens = line.strip().split()  # Split the line into tokens by whitespace
                total_tokens += len(tokens)  # Add the number of tokens in the current line
                total_lines += 1  # Increment the line count

        # Calculate average number of tokens per line
        if total_lines > 0:
            average_tokens = total_tokens / total_lines
            print(f"Average number of tokens per line: {average_tokens:.2f}")
        else:
            print("The file is empty.")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def view_dataset_sample(file_path='dataset.txt', num_lines=5):
    """
    Print the first few lines of the dataset file to inspect the data.

    :param file_path: Path to the dataset file (default is 'dataset.txt')
    :param num_lines: Number of lines to display (default is 5)
    """
    try:
        with open(file_path, 'r') as f:
            for i in range(num_lines):
                line = f.readline().strip()
                if not line:
                    break  # End of file reached
                print(f"Line {i+1}: {line}")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Function to convert tokenized sequence back to the original tensor
def detokenize_sequence(tokenized_sequences, num_time_steps, num_notes=128):
    # Mapping of track labels back to their indices
    reverse_track_mapping = {
        'D': 0,    # Drums
        'P': 1,    # Piano
        'C': 2,    # Chromatic Percussion
        'O': 3,    # Organ
        'G': 4,    # Guitar
        'B': 5,    # Bass
        'S': 6,    # Strings
        'E': 7,    # Ensemble
        'R': 8,    # Brass
        'r': 9,    # Reed
        'Pp': 10,  # Pipe
        'L': 11,   # Synth Lead
        'Pad': 12, # Synth Pad
        'Fx': 13,  # Synth Effects
        'Eth': 14, # Ethnic
        'Per': 15, # Percussive
        'X': 16,   # Sound Effects
    }
    max_note_values = {
        0: 115,
        1: 118,
        2: 126,
        3: 106,
        4: 127,
        5: 93,
        6: 106,
        7: 104,
        8: 90,
        9: 88,
        10: 127,
        11: 103,
        12: 98,
        13: 116,
        14: 117,
        15: 121,
        16: 123
    }
    # Initialize an empty tensor with the original shape: (17, num_time_steps, 72)
    detokenized_tensor = torch.zeros((17, num_time_steps, num_notes), dtype=torch.float32)

    # Iterate over each time step
    for time_step, token_string in enumerate(tokenized_sequences):
        # Split the token string into individual tokens
        tokens = token_string.strip().split()

        # Process each token
        for token in tokens:
            if token != "<PAD>":
                # Extract the track label and note index from the token
                # We are assuming that all track labels (like 'Pp', 'Pad') are distinct and without numbers,
                # so the track label is composed of letters, and the note index is the number at the end.
                track_label = ''.join(filter(str.isalpha, token))
                note_idx = int(''.join(filter(str.isdigit, token)))

                # Get the corresponding track index using reverse mapping
                track_idx = reverse_track_mapping[track_label]

                # Set the note in the detokenized tensor to 1 (indicating an active note)
                detokenized_tensor[track_idx, time_step, note_idx] = max_note_values[track_idx] * (1 - track_idx / (len(reverse_track_mapping) - 1))

    return detokenized_tensor
