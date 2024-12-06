from tokenizer import PianoRollTokenizer
import torch
import random
import torch.nn.functional as F

class PianoRollDataLoader():
    def __init__(self, B, T, dataset_path="cleaned_dataset.txt") -> None:
        """
        Initialize the data loader with batch size B and sequence length T.
        :param B: Batch size
        :param T: Sequence length
        """
        self.B = B
        self.T = T
        
        # Load the text file with each line treated as an independent song
        with open(dataset_path, 'r') as f:
            lines = f.readlines()  # Each line is a song
        
        self.tokenizer = PianoRollTokenizer(dataset_path=dataset_path)
        # Tokenize each song separately
        self.song_tokens = [torch.tensor(self.tokenizer.encode(song.strip())) for song in lines]
        
        # Filter out empty songs (if any)
        self.song_tokens = [tokens for tokens in self.song_tokens if len(tokens) > 0]
        
        # Shuffle the song tokens
        random.shuffle(self.song_tokens)
        
        # Initialize the current song index and batch pointers
        self.current_song_idx = 0
        self.song_indices = list(range(len(self.song_tokens)))  # Indices of all songs
        self.song_position = [0] * len(self.song_tokens)  # Track current position in each song
    
    def next_batch(self):
        """
        Return the next batch of data (input and target sequences).
        Each batch is constructed from multiple songs without crossing their boundaries.
        :return: Tuple (x, y) where x is the input and y is the target.
        """
        B, T = self.B, self.T
        batch_x = []
        batch_y = []
        
        # Process B songs for the batch
        for _ in range(B):
            # Get the current song index and tokens
            song_idx = self.song_indices[self.current_song_idx]
            song_tokens = self.song_tokens[song_idx]
            pos = self.song_position[song_idx]
            
            # Calculate the remaining length of the song from the current position
            remaining_length = len(song_tokens) - pos
            if remaining_length < T + 1:
                # Pad the sequence if it is shorter than T+1
                x = song_tokens[pos:].view(1, remaining_length)  # Input tokens (1 sequence)
                y = song_tokens[pos+1:].view(1, remaining_length - 1)  # Target tokens
                # Padding the sequence to have length T
                x_padded = F.pad(x, (0, T - remaining_length), value=self.tokenizer.token_to_id['<PAD>'])
                y_padded = F.pad(y, (0, T - remaining_length + 1), value=self.tokenizer.token_to_id['<PAD>'])
                
                batch_x.append(x_padded)
                batch_y.append(y_padded)
                
                # Reset the song position and move to the next song
                self.song_position[song_idx] = 0
                self.current_song_idx += 1
                if self.current_song_idx >= len(self.song_tokens):
                    self.current_song_idx = 0
                    random.shuffle(self.song_indices)  # Shuffle the songs for the next epoch
            else:
                # If the sequence is long enough, take T tokens for x and T+1 tokens for y
                x = song_tokens[pos:pos+T].view(1, T)  # Input tokens (1 sequence)
                y = song_tokens[pos+1:pos+T+1].view(1, T)  # Target tokens
                
                batch_x.append(x)
                batch_y.append(y)
                
                # Update the song position for this song
                self.song_position[song_idx] += T
                # Only move to the next song if we've truly processed the entire sequence
                if self.song_position[song_idx] >= len(song_tokens):
                    self.song_position[song_idx] = 0
                    self.current_song_idx += 1
                    if self.current_song_idx >= len(self.song_tokens):
                        self.current_song_idx = 0
                        random.shuffle(self.song_indices)  # Shuffle the songs for the next epoch
        # Stack all the B samples together
        if len(batch_x) == 0 or len(batch_y) == 0:
            return None, None  # In case no valid batches are produced
        
        batch_x = torch.cat(batch_x, dim=0)  # Shape (B, T)
        batch_y = torch.cat(batch_y, dim=0)  # Shape (B, T)
        
        return batch_x, batch_y