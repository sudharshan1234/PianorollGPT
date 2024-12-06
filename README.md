# PianorollGPT

This repository contains a Google Colab notebook for training and generating music using a GPT-2 model adapted for pianoroll representations.

## Requirements

- Google Colab environment
- Python 3.7 or higher
- PyTorch
- Pretty MIDI
- PyPianoroll
- MIR_EVAL
- Music21
- Triton
- Torch-fidelity
- Pytorch-fid

You can install these dependencies using pip:
`bash !pip install torch-fidelity pytorch-fid pypianoroll mir_eval music21 triton`

## Dataset

The notebook uses the Lakh Piano Dataset. You'll need to download the dataset and place it in your Google Drive.
https://ucsdcloud-my.sharepoint.com/personal/h3dong_ucsd_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fh3dong%5Fucsd%5Fedu%2FDocuments%2Fdata%2Flpd%2Flpd%5F17%2Flpd%5F17%5Fcleansed%2Etar%2Egz&parent=%2Fpersonal%2Fh3dong%5Fucsd%5Fedu%2FDocuments%2Fdata%2Flpd%2Flpd%5F17&ga=1
The notebook assumes the dataset is located at `/content/drive/Shareddrives/Projects/PianorollGPT/Lakh Piano Dataset`.

## Usage

1. **Mount Google Drive:** The notebook first mounts your Google Drive to access the dataset:
   `python from google.colab import drive drive.mount('/content/drive')`

**Data Preparation:**

- Select the desired song IDs from the Lakh Piano Dataset using the `get_random_song_ids` function.
- Create combined pianorolls for the selected songs using the `create_combined_pianorolls` function.
- Generate a text-based dataset from the combined pianorolls using the `generate_dataset` function.
- Clean the dataset by removing extra spaces and newlines using the provided code.

3. **Model Training:**

   - Define the GPT model configuration using the `GPTConfig` class.
   - Create a `PianoRollDataLoader` to load the training data in batches.
   - Instantiate the GPT model and move it to the appropriate device (CPU, GPU, or MPS).
   - Define the optimizer and learning rate scheduler.
   - Train the model using the provided training loop, saving checkpoints periodically.

4. **Music Generation:**
   - Load the trained model checkpoint.
   - Create a `PianoRollTokenizer` to tokenize and detokenize pianoroll sequences.
   - Generate music using the `model.generate` function, providing a starting context and desired length.
   - Detokenize the generated output to obtain the pianoroll representation.
   - Convert the pianoroll to a Pretty MIDI object and play the generated music.

## Load the Model

`model = GPT(GPTConfig()) model.load_state_dict(torch.load("model_checkpoint_step_15000.pth")) model.eval() `

## Generate music

`context = torch.tensor([[tokenizer.token_to_id['']]], dtype=torch.long, device=device) generated_output = model.generate(context, tokenizer, max_new_tokens=2048, temperature=0.9)`

## Detokenize and convert to MIDI

`predicted_tokens = tokenizer.decode(generated_output[0].tolist()) predicted_tensor = detokenize_sequence(predicted_tokens) predicted_multitrack = pypianoroll.Multitrack(name='Original', resolution=4, tracks=tracks) predicted_pm = pypianoroll.to_pretty_midi(predicted_multitrack)`

## Play the generated music

`predicted_midi_audio = predicted_pm.fluidsynth() IPython.display.Audio(predicted_midi_audio, rate=44100)`

## Notes

- The notebook provides basic instructions for using the model. You may need to adjust parameters and code for your specific needs.
- The dataset path and model checkpoint locations are assumed to be in the Google Drive directory specified in the notebook. You may need to modify these paths accordingly.
- The generation process uses a temperature parameter to control the randomness of the output. Experiment with different temperature values to achieve desired results.