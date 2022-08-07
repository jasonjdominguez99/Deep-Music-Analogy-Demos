# preprocess_midi_data.py
#
# main source code for preprocess the Nottingham dataset
# melody and chord midi data files, in accordance with the
# 'DEEP MUSIC ANALOGY VIA LATENT REPRESENTATION
# DISENTANGLEMENT' paper, for use with the ec-squared vae model


# imports
import pypianoroll
import json

# function definitions and implementations


def main():
    config_file_path = "ec_squared_vae/code/ec_squared_vae_model_config.json"
    
    with open(config_file_path) as f:
        args = json.load(f)

    melody_files_dir = f"{args['unprocessed_data_dir']}/melody"
    chord_files_dir = f"{args['unprocessed_data_dir']}/chords"
    
    # testing reading in a midi file
    fname = "ashover1.mid"
    test_melody_track = pypianoroll.read(f"{melody_files_dir}/{fname}")
    test_chord_track = pypianoroll.read(f"{melody_files_dir}/{fname}")    
    
    print()
    print(f"Melody:\n{test_melody_track}")
    print()
    print(f"Chords:\n{test_chord_track}")

if __name__ == "__main__":
    main()
    