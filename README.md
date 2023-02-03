# Drake_songs

In this repo, a Drake song generator model has been trained using drakes' songs from the 2010s. The data was obtained from kaggle from the following link: https://www.kaggle.com/datasets/juicobowley/drake-lyrics 

The architecture used is a stacked LSTM with an Embeddings layer preceding the LSTM layers and one Dense and one final output Dense layer after. The architecture is not complex, however to generate text, a subclass of tf.keras.Models was instantiated with its own call method, therefore a call method had to be implemented, carefully managing the states as input is passed to each subsequent layer. The architecture is available in the models.py module in the models folder.

To build this model, this tutorial from tensorflow served as a main inspiration: https://www.tensorflow.org/text/tutorials/text_generation, however a lot of the code was refactored into dataclasses for further use
