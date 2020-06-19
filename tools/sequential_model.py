import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM

class SequentialPostProcess:
    def build(num_seq, num_features, output_shape):
        input_shape = (num_seq, num_features)

        inputs = Input(shape=input_shape)

        # Layer 1
        x = LSTM(128, activation='tanh', return_sequences=True)(inputs)
        x = Dropout(0.2)(x)

        # Layer 2
        x = LSTM(128, activation='tanh')(x)
        x = Dropout(0.2)(x)

        # Final layer
        outputs = Dense(output_shape, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs, name='sequential_post_yolo')

        return model