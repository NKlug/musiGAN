import mido
from mido import MidiFile, MidiTrack, Message
import time
import glob
import numpy as np

from params import magnitude_notes, num_notes, min_note, max_len, temp_normalize


class Bar(object):
    def __init__(self):
        pass


def to_midi_track(notes):
    track = MidiTrack()
    notes = np.asarray(notes, dtype=np.int)
    for note in notes:
        track.append(Message('note_on', note=note, time=0))
        track.append(Message('note_off', note=note, time=200))
    return track


def save_as_midi(notes, output_path, ticks_per_beat=480):
    mid = MidiFile(ticks_per_beat=ticks_per_beat)

    track = to_midi_track(notes)
    mid.tracks.append(track)

    mid.save(output_path)


def save_as_temporal_midi(notes, output_path, ticks_per_beat=480):
    mid = MidiFile(ticks_per_beat=ticks_per_beat)

    track = to_temporal_midi_track(notes)
    mid.tracks.append(track)

    mid.save(output_path)


def to_temporal_midi_track(notes):
    track = MidiTrack()
    notes = np.asarray(notes, dtype=np.int)
    for note in notes:
        track.append(Message('note_on', note=note[0], time=0))
        track.append(Message('note_off', note=note[0], time=note[1]))
    return track


def to_midi_pitch(notes):
    return notes + min_note


def from_midi_pitch(notes):
    return notes - min_note


def to_one_hot(notes):
    notes = np.asarray(notes)

    b = np.zeros((len(notes), magnitude_notes))
    b[np.arange(len(notes)), notes] = 1
    return np.asarray(b)


def dataset_from_midi(file_path):
    all_notes = midi_to_notes(file_path)
    all_notes = from_midi_pitch(all_notes)
    batched = []
    for i in range(len(all_notes) // num_notes):
        batch = to_one_hot(all_notes[i * num_notes:(i + 1) * num_notes])
        batched.append(batch)

    batched = np.asarray(batched)
    return batched


def temporal_dataset_from_midi(file_path):
    all_notes = midi_to_temporal_notes(file_path)
    notes = all_notes[:, 0]
    times = all_notes[:, 1]
    notes = from_midi_pitch(notes)
    times = times / temp_normalize

    batched = []
    for i in range(len(notes) // num_notes):
        batch = to_one_hot(notes[i * num_notes:(i + 1) * num_notes])
        batch = np.concatenate([batch, times[i * num_notes:(i + 1) * num_notes, np.newaxis]], axis=-1)
        batched.append(batch)

    batched = np.asarray(batched)
    return batched


def midi_to_notes(file_path):
    midi = mido.MidiFile(file_path)
    tracks = []
    for track in midi.tracks:
        notes = []
        for i in range(len(track)):
            msg = track[i]
            if msg.type == 'note_on':
                if msg.velocity > 0:
                    notes.append(msg.note)
            i += 1
        tracks.append(notes)
    all_notes = []
    for track in tracks:
        rounded = round_to(len(track), num_notes)
        track = track[:rounded]
        if len(track) > 0:
            all_notes.extend(track)
    all_notes = np.asarray(all_notes)
    return all_notes


def round_to(x, base):
    return base * (x // base)


def midi_to_temporal_notes(file_path):
    midi = mido.MidiFile(file_path)
    tracks = []
    for track in midi.tracks:
        notes = []
        for i in range(len(track)):
            msg = track[i]
            if msg.type == 'note_on':
                if msg.velocity > 0:
                    off_msg = track[i + 1]
                    notes.append((msg.note, off_msg.time))
            i += 1
        tracks.append(notes)

    all_notes = []
    for track in tracks:
        rounded = round_to(len(track), num_notes)
        track = track[:rounded]
        if len(track) > 0:
            all_notes.extend(track)

    rounded = round_time(all_notes)
    return np.asarray(rounded)


def round_time(notes, smallest_unit=2):
    rounded = []
    for note in notes:
        t = smallest_unit * round(note[1] / smallest_unit)
        if 0 < t < max_len:
            rounded.append((note[0], t))
    return rounded


if __name__ == '__main__':
    a = [64, 66, 68, 69, 71]
    a = np.asarray(a)
    # a -= -32
    # save_as_midi(a, 'new_song.mid')
    # print(to_one_hot(from_midi_pitch(a)))

    # notes = midi_to_notes('/home/nklug/projects/mgan/zauberflute.mid')
    # save_as_midi(notes, 'tmp.mid')

    # track = midi_to_temporal_notes('/home/nklug/projects/mgan/zauberflute.mid')
    # print(np.max(track[:, 1]))
    # save_as_temporal_midi(track[80:], 'tmp.mid', 64)

    # notes = dataset_from_midi('/home/nklug/projects/mgan/zauberflute.mid')
    # print(notes.shape)

    # all_notes = []
    # total_len = 0
    # for file in glob.glob('/home/nklug/projects/mgan/data/MTC-ANN-2.0.1/mid/*.mid'):
    #     notes = midi_to_temporal_notes(file)
    #     all_notes.append(notes)
    #     total_len += len(notes)
    #
    # concat = []
    # for track in all_notes:
    #     concat.extend(track)
    # save_as_temporal_midi(concat, 'mtclc.mid', 128)

    dataset = temporal_dataset_from_midi('mtclc.mid')
    print()
