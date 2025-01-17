import json
import numpy as np
import tensorflow
import music21 as m21
from preprocess import make_custom_onehot_mapping, SEQUENCE_LENGTH, MAPPING_PATH

keras = tensorflow.keras

def convertNoteToInt(notes: list[str]):
    noteOffset = {
        "C": 0,
        "D": 2,
        "E": 4,
        "F": 5,
        "G": 7,
        "A": 9,
        "B": 11,
    }
    return list(map(lambda note: 12 * (int(note[1]) + 1) + noteOffset[note[0]], notes))

class MelodyGenerator:
    """A class that wraps the LSTM model and offers utilities to generate melodies."""

    def __init__(self, model_path="model.h5"):
        """Constructor that initialises TensorFlow model"""

        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH


    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        """Generates a melody using the DL model and returns a midi file.

        :param seed (str): Melody seed with the notation used to encode the dataset
        :param num_steps (int): Number of steps to be generated
        :param max_sequence_len (int): Max number of steps in seed to be considered for generation
        :param temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
            A number closer to 1 makes the generation more unpredictable.

        :return melody (list of str): List with symbols representing a melody
        """

        # create seed with start symbols
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # map seed to int
        seed = [self._mappings[symbol] for symbol in seed]

        last_output_symbol = ""
        for _ in range(num_steps + 100000):

            # one-hot encode the seed
            onehot_seed = np.array(make_custom_onehot_mapping(
                seed,
                0,
                num_classes=len(self._mappings)
            ))
            # (1, max_sequence_length, num of symbols in the vocabulary)
            onehot_seed = onehot_seed[np.newaxis, ...]

            # make a prediction
            probabilities = self.model.predict(onehot_seed, verbose=0)[0]
            if _ < num_steps:
                probabilities[self._mappings["/"]] = 0
            if last_output_symbol != "":
                probabilities[self._mappings[last_output_symbol]] *= 0.8
            # [0.1, 0.2, 0.1, 0.6] -> 1
            output_int = self._sample_with_temperature(probabilities, temperature)

            # update seed
            seed.append(output_int)

            # map int to our encoding
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]
            print(f"symbol: {output_symbol}")
            # check whether we're at the end of a melody
            if output_symbol == "/":
                break
            elif output_symbol != "_":
                last_output_symbol = output_symbol

            # update melody
            melody.append(output_symbol)

        return melody


    def _sample_with_temperature(self, probabilites, temperature):
        """Samples an index from a probability array reapplying softmax using temperature

        :param predictions (nd.array): Array containing probabilities for each of the possible outputs.
        :param temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
            A number closer to 1 makes the generation more unpredictable.

        :return index (int): Selected output symbol
        """
        predictions = np.power(probabilites, 1 / temperature)
        probabilites = predictions / np.sum(predictions)

        choices = range(len(probabilites)) # [0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilites)

        return index


    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="mel.mid"):
        chordList = list(map(lambda x: convertNoteToInt(x), [
            ["C2", "E2", "G2"],
            ["G2", "B2", "D3"],
            ["D2", "F2", "A2"],
            ["E2", "G2", "B2"],
            ["F2", "A2", "C3"],
            ["A1", "C2", "E2"],
            ["B1", "D2", "F2"],
        ]))

        noteTypesByBar = [ [0] ] * ((len(melody) + 15) // 16)
        for i in range(len(noteTypesByBar)):
            noteTypesByBar[i] = [0] * 12

        # create a music21 stream
        stream = m21.stream.Stream()

        start_symbol = None
        start_time = 0
        step_counter = 1

        for i, symbol in enumerate(melody):
            # handle case in which we have a note/rest
            if symbol != "_" or i + 1 == len(melody):
                # ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter

                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                    # handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)
                        noteTypesByBar[start_time // 16][int(start_symbol) % 12] += 1
                        if ((start_time + step_counter - 1) // 16 != start_time // 16):
                            noteTypesByBar[(start_time + step_counter - 1) // 16][int(start_symbol) % 12] += 1

                    stream.append(m21_event)
                    step_counter = 1
                start_symbol = symbol
                start_time = i
            # handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1
        prev_chord = []
        for i, noteTypes in enumerate(noteTypesByBar):
            max_score = 0
            choose_chord = [0, 0, 0]
            for chord in chordList:
                score = noteTypes[(chord[0] % 12)] * 1.2 + noteTypes[(chord[1] % 12)] * 0.9 + noteTypes[(chord[2] % 12)]
                # If is I chord
                if chord[0] % 12 == 0:
                    if i % 4 == 0 or i == len(noteTypesByBar) - 1:
                        score += 1
                    elif i % 4 == 23:
                        score -= 0.1
                    else:
                        score -= 0.9
                elif prev_chord == chord:
                    score -= 1

                if score > max_score:
                    max_score = score
                    choose_chord = chord
            if max_score > 0:
                new_chord = m21.chord.Chord(choose_chord, quarterLength=4)
                new_chord.volume = m21.volume.Volume(velocity=60)
                stream.insert(i * 4.0, new_chord)
            prev_chord = choose_chord
        # write the m21 stream to a midi file
        stream.write(format, file_name)


if __name__ == "__main__":
    mg = MelodyGenerator()
    seed = "67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _ 62"
    seed2 = "67 _ _ _ _ _ 65 _ 64 _ 62 _ 60 _ _ _"
    melody = mg.generate_melody(seed, 128, SEQUENCE_LENGTH, 1.0)
    print(melody)
    mg.save_melody(melody)