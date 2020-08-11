import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Flatten, Lambda
from tensorflow.keras.losses import binary_crossentropy, mean_squared_error, sparse_categorical_crossentropy

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


    def build_multi_input(image_input_shape, num_seq, num_bboxes, num_classes):
        image_input = Input(shape=image_input_shape)
        bbox_input = Input(shape=(num_seq, num_bboxes, 6))

        # first branch, the resnet on the image followed by a flatten
        resnet50  = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            pooling='max'
        )

        x_i = resnet50.output
        x_i = Flatten()(image_input)
        x_i = Model(inputs=image_input, outputs=x_i)

        # second branch, the bboxes from the object detection model
        x_bbox = Lambda(lambda x: tf.reshape(x, (-1, num_seq, num_bboxes * 6)))(bbox_input)
        x_bbox = LSTM(128, activation='tanh')(x_bbox)
        x_bbox = Dropout(0.2)(x_bbox)
        x_bbox = Model(inputs=bbox_input, outputs=x_bbox)

        combine_i_bbox = tf.concat([x_i.output, x_bbox.output], axis=1)
        x = Dense(num_bboxes*(5 + num_classes), activation='sigmoid')(combine_i_bbox)

        # reshape the output
        x = Lambda(lambda x: tf.reshape(x, (-1, num_bboxes, 5 + num_classes)))(x)

        model = Model(inputs=[x_i.input, x_bbox.input], outputs=x)

        return model



# Our custom loss function for our model
#
# This model outputs a batch_size x (n * 5) length 1D array, where n is the number of objects predicted
# The first element is the score and the next 4 are the bounding boxes
# We would like to use cross entropy loss on the score (since it should be 0 or 1)
# And we will use mean squared error for the box coordinates
def sequential_model_loss(n_objects=5, n_classes=4):
    # We can reshape the array 
    def custom_loss(y_true, y_pred):
        y_pred = tf.reshape(y_pred, (-1, n_objects, 5 + n_classes))
        pred_obj, pred_xywh, pred_class = tf.split(y_pred, [1, 4, n_classes], axis=-1)

        y_true = tf.reshape(y_true, (-1, n_objects, 6))
        true_obj, true_xywh, true_class = tf.split(y_true, [1, 4, 1], axis=-1)

        # Cross entropy loss for score
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = tf.reduce_sum(obj_loss, axis=1)
        
        # MSE loss for the BB
        bb_loss = mean_squared_error(true_xywh, pred_xywh)
        bb_loss = tf.reduce_sum(bb_loss, axis=1)

        # We have to convert the class into a vector
        class_loss = sparse_categorical_crossentropy(true_class, pred_class)
        class_loss = tf.reduce_sum(class_loss, axis=1)

        return obj_loss + bb_loss + class_loss

    return custom_loss


