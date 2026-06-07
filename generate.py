import numpy as np
from keras.models import load_model
from music21 import instrument, note, chord, stream
import random

# ===== LOAD MODEL AND DATA =====
print("Loading model and data...")
model = load_model("best_model.keras")
unique_notes = np.load("unique_notes.npy", allow_pickle=True)
X = np.load("X.npy")

n_vocab = len(unique_notes)
int_to_note = {i: n for i, n in enumerate(unique_notes)}

# ===== PICK A RANDOM STARTING SEQUENCE =====
start = random.randint(0, len(X) - 1)
pattern = list(X[start].flatten() * n_vocab)

print(f"Generating music with {n_vocab} unique notes...")
print("Please wait...\n")

# ===== GENERATE 200 NOTES =====
generated_notes = []

for i in range(500):
    input_seq = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
    prediction = model.predict(input_seq, verbose=0)
    index = np.argmax(prediction)
    result = int_to_note[index]
    generated_notes.append(result)
    pattern.append(index)
    pattern = pattern[1:]

    if (i + 1) % 20 == 0:
        print(f"Generated {i+1}/200 notes...")

print("\nConverting to MIDI...")

# ===== CONVERT NOTES TO MIDI =====
output_notes = []
offset = 0

for pattern in generated_notes:
    if ('.' in pattern):
        # It's a chord — numbers joined by dots
        notes_in_chord = pattern.split('.')
        chord_notes = []
        for n in notes_in_chord:
            try:
                new_note = note.Note(int(n))
                new_note.storedInstrument = instrument.Piano()
                chord_notes.append(new_note)
            except:
                pass
        if chord_notes:
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
    else:
        # It's a single note like C4, D#5 etc
        try:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        except:
            pass

    offset += 0.5

# ===== SAVE AS MIDI FILE =====
if output_notes:
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output.mid')
    print("\n=== Done! Music saved as output.mid ===")
    print("Open output.mid with any media player to listen!")
else:
    print("No notes generated. Try running again!")