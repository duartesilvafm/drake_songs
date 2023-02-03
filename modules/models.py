import tensorflow as tf


class CharModel(tf.keras.Model):
  
  
    def __init__(self, vocab_size, embedding_dim, rnn_units, dense_units=1024):

        # initialize tf.keras.Model
        super().__init__(self)

        # architecture
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        
        # stack 2 lstm layers with 0.2 dropout rate
        self.lstm = tf.keras.layers.LSTM(rnn_units,
                                    return_sequences=True,
                                    stateful=True,
                                    go_backwards=True,
                                    dropout=0.1)

        self.lstm2 = tf.keras.layers.LSTM(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            go_backwards=True,
                            dropout=0.1)

    
        # one more dense layer
        self.dense = tf.keras.layers.Dense(dense_units, activation='relu')

        # output layer with number units equivalent to the        
        self.dense_output = tf.keras.layers.Dense(vocab_size)


    def call(self, inputs, states=None, return_state=False, training=False):
        
        x = inputs
        x = self.embedding(x, training=training)

        if states is None:
            states = self.lstm.get_initial_state(x)

        x = self.lstm(x, initial_state=states, training=training)
        x = self.lstm2(x, training=training)
        states=self.lstm2.states

        x = self.dense(x, training=training)
        x = self.dense_output(x, training=training)

        if return_state:
            return x, states
        else:
            return x


# Define a Callback class that stops training once loss of 0.10 is reached
class MyCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):

    if(logs.get('loss')<0.10):
      print("\nReached 0.10 loss so cancelling training!")
      self.model.stop_training = True