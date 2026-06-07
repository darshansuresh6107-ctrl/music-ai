import os
import numpy as np
from music21 import converter, instrument, note, chord

def get_notes(midi_folder):
    notes = []
    files = [f for f in os.listdir(midi_folder) if f.endswith('.mid') or f.endswith('.midi')]
    print(f"Found {len(files)} MIDI files\n")

    for file in files:
        filepath = os.path.join(midi_folder, file)
        print(f"Processing: {file}")
        try:
            midi = converter.parse(filepath)
            parts = instrument.partitionByInstrument(midi)
            notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        except Exception as e:
            print(f"  Skipped {file}: {e}")

    print(f"\nTotal notes extracted: {len(notes)}")
    return notes

def prepare_sequences(notes, seq_length=100):
    unique_notes = sorted(set(notes))
    print(f"Unique notes: {len(unique_notes)}")

    note_to_int = {n: i for i, n in enumerate(unique_notes)}

    X, y = [], []
    for i in range(len(notes) - seq_length):
        X.append([note_to_int[n] for n in notes[i:i+seq_length]])
        y.append(note_to_int[notes[i+seq_length]])

    X = np.reshape(X, (len(X), seq_length, 1)) / float(len(unique_notes))

    y_encoded = np.zeros((len(y), len(unique_notes)))
    for i, val in enumerate(y):
        y_encoded[i][val] = 1

    print(f"Training patterns: {len(X)}")
    return X, y_encoded, unique_notes

if __name__ == "__main__":
    print("=== Preprocessing Started ===\n")
    notes = get_notes("midi_files")

    if len(notes) < 100:
        print("Not enough notes! Add more MIDI files.")
    else:
        X, y, unique_notes = prepare_sequences(notes)
        np.save("X.npy", X)
        np.save("y.npy", y)
        np.save("unique_notes.npy", unique_notes)
        print("\n=== Done! Files saved: X.npy, y.npy, unique_notes.npy ===")