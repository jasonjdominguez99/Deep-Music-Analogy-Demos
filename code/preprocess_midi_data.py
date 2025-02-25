# preprocess_midi_data.py
#
# main source code for preprocess the Nottingham dataset
# melody and chord midi data files, in accordance with the
# 'DEEP MUSIC ANALOGY VIA LATENT REPRESENTATION
# DISENTANGLEMENT' paper, for use with the ec-squared vae model


# imports
import glob
import os
import json
import random
import pickle
import numpy as np
import pretty_midi as pm
from tqdm import tqdm
from scipy.sparse import csc_matrix


# function definitions and implementations
def pad_pianorolls(pianoroll, timelen):
    if pianoroll.shape[1] < timelen:
        pianoroll = np.pad(pianoroll, ((0, 0), (0, timelen - pianoroll.shape[1])),
                           mode="constant", constant_values=0)
    return pianoroll


def preprocess_data(root_dir, midi_dir, num_bars, frame_per_bar, pitch_range=48, shift=False,
                    beat_per_bar=4, bpm=120, data_ratio=(0.8, 0.1, 0.1)):
    
    # config_file_path = "code/ec_squared_vae_model_config.json"
    
    # with open(config_file_path) as f:
    #     args = json.load(f)

    # melody_files_dir = f"{args['unprocessed_data_dir']}/melody"
    # chord_files_dir = f"{args['unprocessed_data_dir']}/chords"
    
    # testing reading in a midi file
    
    if shift:
        instance_folder = 'instance_pkl_%dbars_fpb%d_%dp_12keys' % (num_bars, frame_per_bar, pitch_range)
    else:
        instance_folder = 'instance_pkl_%dbars_fpb%d_%dp_ckey' % (num_bars, frame_per_bar, pitch_range)

    dir_name = os.path.join(root_dir, 'pkl_files', instance_folder)
    os.makedirs(dir_name, exist_ok=True)

    instance_len = frame_per_bar * num_bars
    stride = int(instance_len / 2)
    # Default : frame_per_second=8, unit_time=0.125
    frame_per_second = (frame_per_bar / beat_per_bar) * (bpm / 60)
    unit_time = 1 / frame_per_second

    midi_files = sorted(glob.glob(os.path.join(root_dir, midi_dir, '*.mid'))) # changed from */*.mid
    song_list = []
    if not os.path.exists(os.path.join(root_dir, midi_dir, 'songs')):
        # os.makedirs(os.path.join(root_dir, midi_dir, 'songs'))
        for fname in midi_files:
            # print(fname)
            print(fname.split('/')[-1].split('\\')[-1].split('.')[0])
            song_list.append(fname.split('/')[-1].split('\\')[-1].split('.')[0])
    else:
        song_list = sorted(glob.glob(os.path.join(root_dir, midi_dir, 'songs/*'))) # changed from '*'
    

    num_eval = int(len(song_list) * data_ratio[1])
    num_test = int(len(song_list) * data_ratio[2])
    random.seed(0)
    eval_test_cand = set([song.split('/')[-1] for song in song_list])
    eval_set = random.sample(eval_test_cand, num_eval)
    test_set = random.sample(eval_test_cand - set(eval_set), num_test)

    for midi_file in tqdm(midi_files, desc="Processing"):
        song_title = midi_file.split('/')[-1].split('\\')[-1].split('.')[0]
        filename = midi_file.split('/')[-1].split('.')[0]
# # imports
# import pypianoroll
# import pretty_midi as pm
# import json

# # function definitions and implementations

        if song_title in eval_set:
            mode = 'eval'
        elif song_title in test_set:
            mode = 'test'
        else:
            mode = 'train'
        os.makedirs(os.path.join(dir_name, mode, song_title), exist_ok=True)
        key_count = len(sorted(glob.glob(os.path.join(dir_name, mode, song_title, '*_+0_*.pkl')))) # in case of modulation

        if shift:
            pitch_shift = range(-5, 7)
        else:
            pitch_shift = [0]
        for k in pitch_shift:
            midi = pm.PrettyMIDI(midi_file)
            if len(midi.instruments) < 2:
                continue
            on_midi = pm.PrettyMIDI(midi_file)
            off_midi = pm.PrettyMIDI(midi_file)
            note_instrument = midi.instruments[0]
            onset_instrument = on_midi.instruments[0]
            offset_instrument = off_midi.instruments[0]
            for note, onset_note, offset_note in zip(note_instrument.notes, onset_instrument.notes, offset_instrument.notes):
                if k != 0:
                    note.pitch += k
                    onset_note.pitch += k
                    offset_note.pitch += k
                note_length = offset_note.end - offset_note.start
                onset_note.end = onset_note.start + min(note_length, unit_time)
                offset_note.end += unit_time
                offset_note.start = offset_note.end - min(note_length, unit_time)
            pianoroll = note_instrument.get_piano_roll(fs=frame_per_second)
            onset_roll = onset_instrument.get_piano_roll(fs=frame_per_second)
            offset_roll = offset_instrument.get_piano_roll(fs=frame_per_second)

            chord_instrument = midi.instruments[1]
            timelen = min(pianoroll.shape[1], offset_roll.shape[1])
            for chord_note in chord_instrument.notes:
                if k != 0:
                    chord_note.pitch += k
                chord_note.end = chord_note.start + unit_time
            chord_onset = chord_instrument.get_piano_roll(fs=frame_per_second)

            pianoroll = pad_pianorolls(pianoroll, timelen)
            onset_roll = pad_pianorolls(onset_roll, timelen)
            offset_roll = pad_pianorolls(offset_roll, timelen)
            chord_onset = pad_pianorolls(chord_onset, timelen)

            pianoroll[pianoroll > 0] = 1
            onset_roll[onset_roll > 0] = 1
            offset_roll[offset_roll > 0] = 1
            chord_onset[chord_onset > 0] = 1

            for i in range(0, timelen - (instance_len + 1), stride):
                pitch_list = []
                chord_list = []

                pianoroll_inst = pianoroll[:, i:(i+instance_len+1)]
                onset_inst = onset_roll[:, i:(i+instance_len+1)]
                chord_inst = chord_onset[:, i:(i + instance_len + 1)]

                if len(chord_inst.nonzero()[1]) < 4:
                    continue

                rhythm_idx = np.minimum(np.sum(pianoroll_inst.T, axis=1), 1) + np.minimum(np.sum(onset_inst.T, axis=1), 1)
                rhythm_idx = rhythm_idx.astype(int)
                # If more than 75% is not-playing, do not make instance
                if rhythm_idx.nonzero()[0].size < (instance_len // 4):
                    continue

                if pitch_range == 128:
                    base_note = 0
                else:
                    highest_note = max(onset_inst.T.nonzero()[1])
                    lowest_note = min(onset_inst.T.nonzero()[1])
                    base_note = 12 * (lowest_note // 12)
                    if highest_note - base_note >= pitch_range:
                        continue

                prev_chord = np.zeros(12)
                cont_rest = 0
                prev_onset = 0
                for t in range(instance_len+1):
                    if t in onset_inst.T.nonzero()[0]:
                        pitch_list.append(onset_inst[:, t].T.nonzero()[0][0] - base_note)
                        if (t != onset_inst.T.nonzero()[0][0]) and abs(onset_inst[:, t].T.nonzero()[0][0] - base_note - prev_onset) > 12:
                            cont_rest = 30
                            break
                        else:
                            prev_onset = onset_inst[:, t].T.nonzero()[0][0] - base_note
                            cont_rest = 0
                    elif rhythm_idx[t] == 1:
                        pitch_list.append(pitch_range)
                    elif rhythm_idx[t] == 0:
                        pitch_list.append(pitch_range + 1)
                        cont_rest += 1
                        if cont_rest >= 30:
                            break
                    else:
                        print(filename, i, t, rhythm_idx[t], onset_inst.T.nonzero())

                    if len(chord_inst[:, t].nonzero()[0]) != 0:
                        prev_chord = np.zeros(12)
                        for note in sorted(chord_inst[:, t].nonzero()[0][1:] % 12):
                            prev_chord[note] = 1
                    chord_list.append(prev_chord)

                if (cont_rest >= 30) or (len(set(pitch_list)) <= 5):
                    continue

                pitch_list = np.array(pitch_list)
                chord_result = csc_matrix(np.array(chord_list))
                result = {'pitch': pitch_list,
                          'rhythm': rhythm_idx,
                          'chord': chord_result}
                ps = ('%d' % k) if (k < 0) else ('+%d' % k)
                pkl_filename = os.path.join(dir_name, mode, song_title, '%s_%02d_%s_%02d.pkl' % (song_title, key_count, ps, i // stride))
                with open(pkl_filename, 'wb') as f:
                    pickle.dump(result, f)
    
def main():
    config_file_path = "code/ec_squared_vae_model_config.json"
    
    with open(config_file_path) as f:
        args = json.load(f)
    
    preprocess_data(
        args["unprocessed_data_dir"], args["midi_dir"], args["nums_bars"],
        args["frame_per_bar"], args["pitch_range"], args["shift"]
    )
# def main():
#     config_file_path = "ec_squared_vae/code/ec_squared_vae_model_config.json"
    
#     with open(config_file_path) as f:
#         args = json.load(f)

#     melody_files_dir = f"{args['unprocessed_data_dir']}/melody"
#     chord_files_dir = f"{args['unprocessed_data_dir']}/chords"
#     song_files_dir = f"{args['unprocessed_data_dir']}"
    
#     # testing reading in a midi file
#     fname = "ashover2.mid"
#     # test_melody_track = pypianoroll.read(f"{melody_files_dir}/{fname}")
#     # test_chord_track = pypianoroll.read(f"{chord_files_dir}/{fname}")  
#     # test_melody_track = pm.PrettyMIDI(f"{melody_files_dir}/{fname}")
#     # test_chord_track = pm.PrettyMIDI(f"{chord_files_dir}/{fname}") 
#     test_song_track = pm.PrettyMIDI(f"{song_files_dir}/{fname}")
    
#     # print()
#     # print(f"Melody:\n{test_melody_track.instruments}")
#     # print()
#     # print(f"Chords:\n{test_chord_track}")
#     print()
#     print(f"Melody:\n{test_song_track.instruments}")
#     print()

# if __name__ == "__main__":
#     main()
    
    
# Code from CMT
    
import glob
import os
import random
import argparse
import pickle
import numpy as np
import pretty_midi as pm
from tqdm import tqdm
from scipy.sparse import csc_matrix


def pad_pianorolls(pianoroll, timelen):
    if pianoroll.shape[1] < timelen:
        pianoroll = np.pad(pianoroll, ((0, 0), (0, timelen - pianoroll.shape[1])),
                           mode="constant", constant_values=0)
    return pianoroll


def make_instance_pkl_files(root_dir, midi_dir, num_bars, frame_per_bar, pitch_range=48, shift=False,
                            beat_per_bar=4, bpm=120, data_ratio=(0.8, 0.1, 0.1)):
    if shift:
        instance_folder = 'instance_pkl_%dbars_fpb%d_%dp_12keys' % (num_bars, frame_per_bar, pitch_range)
    else:
        instance_folder = 'instance_pkl_%dbars_fpb%d_%dp_ckey' % (num_bars, frame_per_bar, pitch_range)

    dir_name = os.path.join(root_dir, 'pkl_files', instance_folder)
    os.makedirs(dir_name, exist_ok=True)

    instance_len = frame_per_bar * num_bars
    print(instance_len)
    # stride = int(instance_len / 2)
    stride = instance_len
    print(stride)
    # Default : frame_per_second=8, unit_time=0.125
    frame_per_second = (frame_per_bar / beat_per_bar) * (bpm / 60)
    unit_time = 1 / frame_per_second

    song_list = sorted(glob.glob(os.path.join(root_dir, midi_dir, '*')))
    # print(song_list)
    # print(len(song_list))
    print(os.path.join(root_dir, midi_dir, '*.mid'))
    midi_files = sorted(glob.glob(os.path.join(root_dir, midi_dir, '*.mid')))
    print(len(midi_files))

    num_eval = int(len(song_list) * data_ratio[1])
    num_test = int(len(song_list) * data_ratio[2])
    random.seed(0)
    eval_test_cand = set([song.split('/')[-1] for song in song_list])
    eval_set = random.sample(eval_test_cand, num_eval)
    test_set = random.sample(eval_test_cand - set(eval_set), num_test)

    pitches = []
    chords = []

    for midi_file in tqdm(midi_files[:5], desc="Processing"):
        song_title = midi_file.split('/')[-2]
        filename = midi_file.split('/')[-1].split('.')[0]

        if song_title in eval_set:
            mode = 'eval'
        elif song_title in test_set:
            mode = 'test'
        else:
            mode = 'train'
        os.makedirs(os.path.join(dir_name, mode, song_title), exist_ok=True)
        key_count = len(sorted(glob.glob(os.path.join(dir_name, mode, song_title, '*_+0_*.pkl')))) # in case of modulation

        if shift:
            pitch_shift = range(-5, 7)
        else:
            pitch_shift = [0]
        for k in pitch_shift:
            midi = pm.PrettyMIDI(midi_file)
            if len(midi.instruments) < 2:
                continue
            on_midi = pm.PrettyMIDI(midi_file)
            off_midi = pm.PrettyMIDI(midi_file)
            note_instrument = midi.instruments[0]
            onset_instrument = on_midi.instruments[0]
            offset_instrument = off_midi.instruments[0]
            for note, onset_note, offset_note in zip(note_instrument.notes, onset_instrument.notes, offset_instrument.notes):
                if k != 0:
                    note.pitch += k
                    onset_note.pitch += k
                    offset_note.pitch += k
                note_length = offset_note.end - offset_note.start
                onset_note.end = onset_note.start + min(note_length, unit_time)
                offset_note.end += unit_time
                offset_note.start = offset_note.end - min(note_length, unit_time)
            pianoroll = note_instrument.get_piano_roll(fs=frame_per_second)
            onset_roll = onset_instrument.get_piano_roll(fs=frame_per_second)
            offset_roll = offset_instrument.get_piano_roll(fs=frame_per_second)

            chord_instrument = midi.instruments[1]
            timelen = min(pianoroll.shape[1], offset_roll.shape[1])
            for chord_note in chord_instrument.notes:
                if k != 0:
                    chord_note.pitch += k
                chord_note.end = chord_note.start + unit_time
            chord_onset = chord_instrument.get_piano_roll(fs=frame_per_second)

            pianoroll = pad_pianorolls(pianoroll, timelen)
            onset_roll = pad_pianorolls(onset_roll, timelen)
            offset_roll = pad_pianorolls(offset_roll, timelen)
            chord_onset = pad_pianorolls(chord_onset, timelen)

            pianoroll[pianoroll > 0] = 1
            onset_roll[onset_roll > 0] = 1
            offset_roll[offset_roll > 0] = 1
            chord_onset[chord_onset > 0] = 1

            for i in range(0, timelen - (instance_len + 1), stride):
                pitch_list = []
                chord_list = []

                pianoroll_inst = pianoroll[:, i:(i+instance_len+1)]
                onset_inst = onset_roll[:, i:(i+instance_len+1)]
                chord_inst = chord_onset[:, i:(i + instance_len + 1)]

                if len(chord_inst.nonzero()[1]) < 4:
                    continue

                rhythm_idx = np.minimum(np.sum(pianoroll_inst.T, axis=1), 1) + np.minimum(np.sum(onset_inst.T, axis=1), 1)
                rhythm_idx = rhythm_idx.astype(int)
                # If more than 75% is not-playing, do not make instance
                if rhythm_idx.nonzero()[0].size < (instance_len // 4):
                    continue

                if pitch_range == 128:
                    base_note = 0
                else:
                    highest_note = max(onset_inst.T.nonzero()[1])
                    lowest_note = min(onset_inst.T.nonzero()[1])
                    base_note = 12 * (lowest_note // 12)
                    if highest_note - base_note >= pitch_range:
                        continue

                prev_chord = np.zeros(12)
                cont_rest = 0
                prev_onset = 0
                for t in range(instance_len):
                    if t in onset_inst.T.nonzero()[0]:
                        # note is an onset
                        pitch_list.append(onset_inst[:, t].T.nonzero()[0][0] - base_note)
                        if (t != onset_inst.T.nonzero()[0][0]) and abs(onset_inst[:, t].T.nonzero()[0][0] - base_note - prev_onset) > 12:
                            cont_rest = 30
                            break
                        else:
                            prev_onset = onset_inst[:, t].T.nonzero()[0][0] - base_note
                            cont_rest = 0
                    elif rhythm_idx[t] == 1:
                        # note is a held note
                        pitch_list.append(pitch_range)
                        # pitch_list.append(pitch_list[-1])
                    elif rhythm_idx[t] == 0:
                        # note is a rest
                        pitch_list.append(pitch_range + 1)
                        cont_rest += 1
                        if cont_rest >= 30:
                            break
                    else:
                        print(filename, i, t, rhythm_idx[t], onset_inst.T.nonzero())

                    if len(chord_inst[:, t].nonzero()[0]) != 0:
                        prev_chord = np.zeros(12)
                        for note in sorted(chord_inst[:, t].nonzero()[0][1:] % 12):
                            prev_chord[note] = 1
                    chord_list.append(prev_chord)

                if (cont_rest >= 30) or (len(set(pitch_list)) <= 5):
                    continue

                # convert pitch list to one-hot vectors with additional held-note and rest info
                # size N x 130, 128 pitches, 1 held-note and 1 rest        
                pitch_info = []
                for pitch in pitch_list:
                    one_hot_pitch = np.zeros(pitch_range + 2)
                    one_hot_pitch[pitch] = 1.
                    pitch_info.append(one_hot_pitch)
                    
                    
                pitch_info = np.array(pitch_info)
                chord_result = np.array(chord_list)
                
                pitches.append(pitch_info)
                chords.append(chord_result)
                
                # print()
                # print(len(pitches))
                # print(len(chords))
                # print(chords)
                # print()
                    
                # pitch_info = csc_matrix(np.array(pitch_info))
                # chord_result = csc_matrix(np.array(chord_list))
                
                # result = {'pitch': pitch_info,
                #           'chord': chord_result}
                    
                # print()
                # print(result)
                # print()
                    
                # ps = ('%d' % k) if (k < 0) else ('+%d' % k)
                # pkl_filename = os.path.join(dir_name, mode, song_title, '%s_%02d_%s_%02d.pkl' % (song_title, key_count, ps, i // stride))
                # with open(pkl_filename, 'wb') as f:
                #     pickle.dump(result, f)

    pitches = np.array(pitches)
    chord_result = np.array(chords)
    
    print()
    print(pitches.shape)
    print()
    print(chord_result.shape)
    print()
    
    data = {
        'pitch': pitches,
        'chord': chords
    }
    
    # save data here
    save_file_path = "ec_squared_vae/processed_data"
    np.save(save_file_path, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='./ec_squared_vae/nottingham_dataset/MIDI')
    parser.add_argument('--midi_dir', type=str, default='melody_and_chords')
    parser.add_argument('--num_bars', type=int, default=2)
    parser.add_argument('--frame_per_bar', type=int, default=16)
    parser.add_argument('--pitch_range', type=int, default=128)
    parser.add_argument('--shift', dest='shift', action='store_true')

    args = parser.parse_args()
    root_dir = args.root_dir
    midi_dir = args.midi_dir
    num_bars = args.num_bars
    frame_per_bar = args.frame_per_bar
    pitch_range = args.pitch_range
    shift = args.shift

    make_instance_pkl_files(root_dir, midi_dir, num_bars, frame_per_bar, pitch_range, shift)
    