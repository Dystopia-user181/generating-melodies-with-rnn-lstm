import json
import sys
import numpy as np
import tensorflow
import music21 as m21
from preprocess import make_custom_onehot_mapping, SEQUENCE_LENGTH, MAPPING_PATH

keras = tensorflow.keras

def convert_notes_to_int(notes: list[str]):
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

chord_list = list(map(lambda x: convert_notes_to_int(x), [
    ["C2", "E2", "G2"],
    ["D2", "F2", "A2"],
    ["E2", "G2", "B2"],
    ["F2", "A2", "C3"],
    ["G2", "B2", "D3"],
    ["A1", "C2", "E2"],
    ["B1", "D2", "F2"],
]))

def is_partof_chord(chord: list[int], note: int):
    return chord[0] % 12 == note % 12 or chord[1] % 12 == note % 12 or chord[2] % 12 == note % 12

def in_chord_range(chord: list[int], note: int):
    start = chord[0] % 12
    end = chord[2] % 12
    if end < start:
        return note % 12 >= start or note % 12 <= end
    else:
        return note % 12 <= end and note % 12 >= start

class MelodyGenerator:
    """A class that wraps the LSTM model and offers utilities to generate melodies."""

    def __init__(self, model_path="model.h5"):
        """Constructor that initialises TensorFlow model"""

        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)
            self._inverse_mappings = dict()
            for key, val in self._mappings.items():
                self._inverse_mappings[val] = key

        self._start_symbols = ["/"] * SEQUENCE_LENGTH
    
    def set_chord_prog(self, prog: list[int]):
        self.chord_prog = list(map(lambda x: chord_list[x - 1], prog))


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
        print("Started")
        last_output_note = ""
        repeat_penalty = 0.8
        for symbol in seed:
            if self._inverse_mappings[symbol].isnumeric():
                last_output_note = self._inverse_mappings[symbol]
        for _ in range(num_steps * 10):
            chord = self.chord_prog[(len(melody) // 16) % len(self.chord_prog)]
            onehot_seed = np.array(make_custom_onehot_mapping(
                seed,
                0,
                num_classes=len(self._mappings)
            ))
            onehot_seed = onehot_seed[np.newaxis, ...]

            # make a prediction
            probabilities = self.model.predict(onehot_seed, verbose=0)[0]
            # prevent early termination if less than desired range or not I chord
            if _ < num_steps or chord[0] % 12 != 0 or (len(melody) % 16) < 2:
                probabilities[self._mappings["/"]] = 0

            # change probabilities based on chord progression
            for key, val in self._inverse_mappings.items():
                if not val.isnumeric():
                    continue
                if is_partof_chord(chord, int(val)):
                    probabilities[key] *= 2
                else:
                    probabilities[key] = 0
                #elif not in_chord_range(chord, int(val)):
                #    probabilities[key] *= 0.7

            if last_output_note != "":
                # fewer note repeats
                probabilities[self._mappings[last_output_note]] *= repeat_penalty
                # limit note jumps to major sixths at most
                for key, val in self._inverse_mappings.items():
                    if not val.isnumeric():
                        continue
                    if abs(int(val) - int(last_output_note)) > 7:
                        probabilities[key] = 0
            
            # probabilities[self._mappings["_"]] *= 2
            # encourage prolongation if on half-beat, if after bar discourage
            if len(seed) % 2 == 1:
                probabilities[self._mappings["_"]] *= 1.5
            #elif len(seed) % 16 == 1:
            #    probabilities[self._mappings["_"]] *= 0.9
            output_int = self._sample_with_temperature(probabilities, temperature)

            # update seed
            seed.append(output_int)

            # map int to our encoding
            output_symbol = self._inverse_mappings[output_int]
            sys.stdout.write(f"\rsymbol: {output_symbol}  ")
            sys.stdout.flush()
            # check whether we're at the end of a melody
            if output_symbol == "/":
                print("\n", chord, chord[0] % 12 != 0, (len(melody) // 16), len(self.chord_prog))
                break
            elif output_symbol.isnumeric():
                if output_symbol == last_output_note:
                    repeat_penalty *= 0.5
                else:
                    repeat_penalty = 0.8
                last_output_note = output_symbol

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
        stream = m21.stream.Stream()

        start_symbol = None
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

                    stream.append(m21_event)
                    step_counter = 1
                start_symbol = symbol
            # handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1
        for i in range((len(melody) + 15) // 16):
            new_chord = m21.chord.Chord(self.chord_prog[i % len(self.chord_prog)], quarterLength=4)
            new_chord.volume = m21.volume.Volume(velocity=60)
            stream.insert(i * 4.0, new_chord)
        # write the m21 stream to a midi file
        stream.write(format, file_name)


if __name__ == "__main__":
    mg = MelodyGenerator()
    mg.set_chord_prog([1, 5, 6, 4])
    seed = "67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _ 62"
    seed2 = "67 _ _ _ _ _ 65 _ 64 _ 62 _ 60 _ _ _"
    melody = mg.generate_melody(seed2, 128, SEQUENCE_LENGTH, 1.3)
    print(melody)
    mg.save_melody(melody)