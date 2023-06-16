import tensorflow
from preprocess import generate_training_sequences, SEQUENCE_LENGTH

keras = tensorflow.keras

OUTPUT_UNITS = 42
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 90
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.h5"


def build_model(output_units, num_units, loss, learning_rate):
    """Builds and compiles model

    :param output_units (int): Num output units
    :param num_units (list of int): Num of units in hidden layers
    :param loss (str): Type of loss function to use
    :param learning_rate (float): Learning rate to apply

    :return model (tf model): Where the magic happens :D
    """

    # create the model architecture
    input = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)

    # compile model
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])

    model.summary()

    return model


def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):
    """Train and save TF model.

    :param output_units (int): Num output units
    :param num_units (list of int): Num of units in hidden layers
    :param loss (str): Type of loss function to use
    :param learning_rate (float): Learning rate to apply
    """

    loadFromExist = input("Load model from existing? (Y/N)").lower() == "y"
    # generate the training sequences
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

    # build the network
    model = keras.models.load_model('./model.h5') if loadFromExist else build_model(output_units, num_units, loss, learning_rate)

    # train the model# Create a callback that saves the model's weights
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=SAVE_MODEL_PATH, verbose=1)
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[cp_callback])

    # save the model
    model.save(SAVE_MODEL_PATH)


if __name__ == "__main__":
    train()