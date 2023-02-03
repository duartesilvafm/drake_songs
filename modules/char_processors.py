import tensorflow as tf
from dataclasses import dataclass


@dataclass
class CharProcessor():


    text: list


    def __post_init__(self):

        # get ids from characters
        self.vocab = sorted(set(self.text))
        print(f'{len(self.vocab)} unique characters')

        # create keras StringLookup layer to get ids from chars
        self.ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(self.vocab), \
            mask_token=None)

        # get characters from ids
        self.chars_from_ids = tf.keras.layers.StringLookup(vocabulary=self.ids_from_chars.get_vocabulary(), \
            invert=True, mask_token=None)


    def ragged_tensor(self, text :str):
        return tf.strings.unicode_split(text, input_encoding='UTF-8')


    def get_chars_from_ids(self, ids: list):
        return self.chars_from_ids(ids)

    
    def get_ids_from_text(self, text: list):
        chars_unicoded = self.ragged_tensor(text=text)
        return self.ids_from_chars(chars_unicoded)


    def text_from_ids(self, ids: list):
        return tf.strings.reduce_join(self.get_chars_from_ids(ids=ids), \
            axis=-1)

    
    def create_dataset(self, text: str, pathsave:str, save:bool=True, \
        sequence:str=50, batch_size:int=100, buffer_size:int=10000):

        # process text to feed to dataset
        model_input = self.get_ids_from_text(text)
        model_dataset = tf.data.Dataset.from_tensor_slices(model_input)

        # create sequences
        sequences = model_dataset.batch(sequence + 1, drop_remainder=True)

        # split input and output
        def split_input_target(sequence):

            # input - everything up to last character
            input_text = sequence[:-1]
            
            # last character
            target_text = sequence[1:]
            
            return input_text, target_text

        dataset = sequences.map(split_input_target)

        # shuffle to randomizer, process in batches and prefetch the dataset
        dataset = (
            dataset
            .shuffle(buffer_size)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE))

        # path to save
        if save:
            dataset.save(pathsave)

        return dataset


    @property
    def vocab(self):
        return self._vocab
    
    @vocab.setter
    def vocab(self, value):
        self._vocab=value

    @property
    def chars(self):
        return self._chars
    
    @chars.setter
    def chars(self, value):
        self._chars=value


class OneStep(tf.keras.Model):


    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)


    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids, states=states,
                                            return_state=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        # Apply the prediction mask: prevent "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states