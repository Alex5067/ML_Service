import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

class Model:
    def __init__(self):
        try:
            self.model = tf.keras.models.load_model('model_data/emotions.h5', compile=False)
            self.emotions_dict = {'id_tag': {0: 'sadness', 1: 'anger', 2: 'love', 3: 'surprise', 4: 'fear', 5: 'joy'},
                                  'tag_id': {'sadness': 0, 'anger': 1, 'love': 2, 'surprise': 3, 'fear': 4, 'joy': 5}}
            self.tokenizer = pickle.load(open("model_data/tokenizer.pkl", 'rb'))
        except Exception as e:
            logging.info(e)

    def predict(self, txt):
        x = pad_sequences(self.tokenizer.texts_to_sequences([txt]), maxlen=30)
        x = self.model(x)
        x = np.argmax(x)
        return self.emotions_dict["id_tag"][x]


if __name__ == "__main__":
    model = Model()
    res = model.predict("I feel neglected and unimportant in our relationship. "
                        "It hurts me when you prioritize other things over our time together.")
