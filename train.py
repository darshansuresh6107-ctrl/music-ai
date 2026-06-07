import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint

# ===== LOAD PREPROCESSED DATA =====
print("Loading data...")
X = np.load("X.npy")
y = np.load("y.npy")
unique_notes = np.load("unique_notes.npy", allow_pickle=True)

n_vocab = len(unique_notes)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Vocabulary size: {n_vocab}")

# ===== BUILD LSTM MODEL =====
print("\nBuilding model...")
model = Sequential()

# First LSTM layer
model.add(LSTM(
    256,
    input_shape=(X.shape[1], X.shape[2]),
    return_sequences=True
))
model.add(Dropout(0.3))

# Second LSTM layer
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.3))

# Third LSTM layer
model.add(LSTM(256))
model.add(Dropout(0.3))

# Output layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(n_vocab, activation='softmax'))

# Compile
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

# ===== SAVE BEST MODEL AUTOMATICALLY =====
checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor='loss',
    save_best_only=True,
    verbose=1
)

# ===== TRAIN =====
print("\nTraining started...")
print("This will take a while — let it run!\n")

model.fit(
    X, y,
    epochs=100,
    batch_size=64,
    callbacks=[checkpoint]
)

print("\n=== Training Complete! Model saved as best_model.keras ===")