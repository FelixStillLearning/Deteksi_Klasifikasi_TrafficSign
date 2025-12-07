from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam

from config import IMG_SIZE, NUM_CLASSES


def build_model():
    model = Sequential()

    # ===== BLOCK 1 =====
    model.add(Conv2D(
        32, (5, 5),
        activation='relu',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    ))
    model.add(MaxPool2D(pool_size=(2, 2)))

    # ===== BLOCK 2 =====
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    # ===== FULLY CONNECTED =====
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    # ===== COMPILE =====
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
