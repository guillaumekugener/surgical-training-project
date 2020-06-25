import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM
from tensorflow.keras.losses import binary_crossentropy, mean_squared_error

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


# Our custom loss function for our model
#
# This model outputs a batch_size x (n * 5) length 1D array, where n is the number of objects predicted
# The first element is the score and the next 4 are the bounding boxes
# We would like to use cross entropy loss on the score (since it should be 0 or 1)
# And we will use mean squared error for the box coordinates
def sequential_model_loss(classes=5):
    # We can reshape the array 
    def custom_loss(y_true, y_pred):
        y_pred = tf.reshape(y_pred, (-1, classes, 5))
        pred_obj, pred_xywh = tf.split(y_pred, [1, 4], axis=-1)

        y_true = tf.reshape(y_true, (-1, classes, 5))
        true_obj, true_xywh = tf.split(y_true, [1, 4], axis=-1)

        # Cross entropy loss first
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = tf.reduce_sum(obj_loss, axis=1)
        
        # MSE loss for the BB
        bb_loss = mean_squared_error(true_xywh, pred_xywh)
        bb_loss = tf.reduce_sum(bb_loss, axis=1)

        return obj_loss + bb_loss

    return custom_loss


