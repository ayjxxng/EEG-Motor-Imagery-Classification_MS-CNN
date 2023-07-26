import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, MaxPooling2D, Concatenate, Flatten, Softmax
from tensorflow.keras.models import Model
from tensorflow.python.util.tf_export import keras_export
from keras import optimizers, losses, activations, models
from keras.layers import Dense, Input, Dropout, Flatten, concatenate, Bidirectional
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import to_categorical

def MSCBBlock(input_tensor):

    block1 = Conv2D(filters=14, kernel_size=(1, 5), strides=1, padding='same', activation='relu')(input_tensor)
    block1 = MaxPooling2D(pool_size=(1, 5), strides=(1, 5), padding='same')(block1)

    block2 = Conv2D(filters=14, kernel_size=(1, 3), strides=1, padding='same', activation='relu')(input_tensor)
    block2 = MaxPooling2D(pool_size=(1, 3), strides=(1, 5), padding='same')(block2)

    block3 = Conv2D(filters=14, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(input_tensor)
    block3 = MaxPooling2D(pool_size=(1, 1), strides=(1, 5), padding='same')(block3)

    block4 = MaxPooling2D(pool_size=(1, 3), strides=(1, 5), padding='same')(input_tensor)
    block4 = Conv2D(filters=24, kernel_size=(1, 3), strides=1, padding='same', activation='relu')(block4)

    all_blocks = [block1, block2, block3, block4]
    MSCB_out = Concatenate(axis=-1)(all_blocks)

    return MSCB_out


data = output

labels = yy

# cross-validation 
n_folds = 10

# batch_size 
batch_size = 32

accuracies = []

kf = KFold(n_splits=n_folds, shuffle=True)

labels = np.array(labels).flatten()
labels -= 1
labels = to_categorical(labels, num_classes=2)

# learning_rate 설정한 것
initial_lr = 0.01
decay_rate = 0.8

def lr_scheduler(epoch, lr):
    return lr * decay_rate

input_tensors = []

# Apply MSCBBlock to each band
band_outputs = []
for i in range(4):

    input_tensor = Input(shape=(3, 1000, 1))
    input_tensors.append(input_tensor)

    band_output = MSCBBlock(input_tensor)

    band_output = Conv2D(filters=112, kernel_size=(1,3), strides=1, padding='same', activation='relu')(band_output)
    band_output = MaxPooling2D(pool_size=(1,2), strides=(1,5), padding='same')(band_output)

    band_output = Flatten()(band_output)
    band_outputs.append(band_output)

# Concatenate the outputs from MSCBBlock for each band
concatenated_output = Concatenate()(band_outputs)

# Add a fully connected layer for classification
output_layer = Dense(128, activation='relu')(concatenated_output)

# Final classification layer
output_layer = Dense(2, activation='softmax')(output_layer)

# Create the final model
model = Model(inputs=input_tensors, outputs=output_layer)

optimizer = SGD(learning_rate=initial_lr)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

for fold, (train_index, test_index) in enumerate(kf.split(data)):
    print(f"Fold: {fold+1}")

    x_train, x_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    model.fit([x_train[:, i] for i in range(4)], y_train, batch_size=batch_size, epochs=500, verbose=1, callbacks=[LearningRateScheduler(lr_scheduler)])

    _, accuracy = model.evaluate([x_test[:, i] for i in range(4)], y_test, verbose=0)
    print(f"Accuracy for Fold {fold+1}: {accuracy}")

    accuracies.append(accuracy)

avg_accuracy = np.mean(accuracies)
print(f"Average accuracy: {avg_accuracy}")