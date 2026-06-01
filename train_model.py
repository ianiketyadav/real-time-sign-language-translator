import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore

DATA_PATH = "landmarks"

X = []
y = []

#  --- labels WITHOUT .npy ---
labels = [f.replace(".npy", "") for f in os.listdir(DATA_PATH)]

# -- Sort for consistency --
labels.sort()

# -- Map labels to numbers --
label_map = {label: num for num, label in enumerate(labels)}


for label in labels:
    file_path = os.path.join(DATA_PATH, f"{label}.npy")
    data = np.load(file_path)

    for sample in data:
        if len(sample) == 42:  # safety check
            X.append(sample)
            y.append(label_map[label])

X = np.array(X)
y = to_categorical(y)

print(f" Total samples: {len(X)}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#  == Model ==
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(42,)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(labels), activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#  -- Train --
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# -- Save model --
os.makedirs("model", exist_ok=True)
model.save("model/sign_model.h5")

# -- labels --
np.save("model/labels.npy", labels)

print(" Landmark model trained successfully!")