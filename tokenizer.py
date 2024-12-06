import re
class PianoRollTokenizer:
    def __init__(self, dataset_path):
        # Define track mapping
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
        with open(dataset_path, 'r') as file:
            text = file.read()

        # Use a regular expression to split the text but keep the leading spaces
        tokens = re.split(r'(\s+)', text)  # This splits the text while preserving spaces

        # Add your special tokens at the end
        unique_tokens = set(tokens)
        vocab = sorted(list(unique_tokens)+['<PAD>'])
        # Create token-to-id mapping
        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        # Create id-to-token mapping
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def encode(self, text):
        tokens = re.split(r'(\s+)', text)  # This splits the text while preserving spaces
        encoded_tokens = [self.token_to_id[token] for token in tokens if token in self.token_to_id]
        return encoded_tokens

    def decode(self, token_ids):
        decoded_tokens = [self.id_to_token[token_id] for token_id in token_ids if token_id in self.id_to_token]
        return " ".join(decoded_tokens)