import re
import os
import json

class PianoRollTokenizer:
    def __init__(self, dataset_paths):
        self.token_to_id_path = "./tokenizer_jsons/token_to_id.json"
        self.id_to_token_path = "./tokenizer_jsons/id_to_token.json"
        # Check if the mappings already exist
        if os.path.exists(self.token_to_id_path) and os.path.exists(self.id_to_token_path):
            # Load the mappings from the JSON files
            with open(self.token_to_id_path, 'r') as token_to_id_file:
                self.token_to_id = json.load(token_to_id_file)
            
            with open(self.id_to_token_path, 'r') as id_to_token_file:
                self.id_to_token = {int(k): v for k, v in json.load(id_to_token_file).items()}  # Convert keys back to int

        else:
            text = ''
            unique_tokens = set()
            genre_tokens = set()

            for tag, dataset_path in dataset_paths:
                print("TAG: ", tag)
                print("DATASET: ", dataset_path)
                genre_tokens.add(f"<{tag}>")
                with open(dataset_path, 'r') as file:
                    for line_num, line in enumerate(file, start=1):
                        # Use a regular expression to split the line but keep the leading spaces
                        tokens = re.split(r'(\s+)', line.strip())  # Split line while preserving spaces
                        unique_tokens.update(tokens)

            # Add your special tokens at the end
            vocab = sorted(list(unique_tokens)+['<PAD>']+list(genre_tokens))

            # Create token-to-id mapping
            self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
            # Create id-to-token mapping
            self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

            # Save token_to_id mapping
            with open(self.token_to_id_path, 'w') as token_to_id_file:
                json.dump(self.token_to_id, token_to_id_file, indent=4)

            # Save id_to_token mapping
            with open(self.id_to_token_path, 'w') as id_to_token_file:
                json.dump(self.id_to_token, id_to_token_file, indent=4)

    def encode(self, text):
        tokens = re.split(r'(\s+)', text)  # This splits the text while preserving spaces
        encoded_tokens = [self.token_to_id[token] for token in tokens if token in self.token_to_id]
        return encoded_tokens

    def decode(self, token_ids):
        decoded_tokens = [self.id_to_token[token_id] for token_id in token_ids if token_id in self.id_to_token]
        return " ".join(decoded_tokens)