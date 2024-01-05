import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Flatten, Conv1D, Conv2D, MaxPooling2D, Concatenate
from keras.optimizers import SGD


def MSCBBlock(block_input):

    block1 = Conv2D(filters=14, kernel_size=(1, 5), strides=1, padding='same', activation='relu')(block_input)
    block1 = MaxPooling2D(pool_size=(1, 5), strides=(1, 5), padding='same')(block1)

    block2 = Conv2D(filters=14, kernel_size=(1, 3), strides=1, padding='same', activation='relu')(block_input)
    block2 = MaxPooling2D(pool_size=(1, 3), strides=(1, 5), padding='same')(block2)

    block3 = Conv2D(filters=14, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(block_input)
    block3 = MaxPooling2D(pool_size=(1, 1), strides=(1, 5), padding='same')(block3)

    block4 = MaxPooling2D(pool_size=(1, 3), strides=(1, 5), padding='same')(block_input)
    block4 = Conv2D(filters=14, kernel_size=(1,3), strides=1, padding='same', activation='relu')(block4)

    all_blocks = [block1, block2, block3, block4]
    MSCB_out = Concatenate(axis=-1)(all_blocks)

    return MSCB_out


def build_model(model = 'MSCNN'):
  """
  model='MSCNN' or 'MSCNN+DE' or 'MSCNN+NPS' or 'MSCNN+all'

  """

    initial_lr = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_lr,
        decay_steps=100,
        decay_rate=0.96,
        staircase=True
        )

    # Apply MSCBBlock to each band
    concatenated_outputs = []

    input_tensor = Input(shape=(3, 1000, 4))
    if model in ['MSCNN+DE', 'MSCNN+all']:
      DE_input_tensor = Input(shape=(3, 8, 4))
    if model in ['MSCNN+NPS', 'MSCNN+all']:
      NPS_input_tensor = Input(shape=(3, 6, 4))

    for i in range(4):
        input = input_tensor[:, :, :, i]
        block_input = tf.expand_dims(input, axis=-1)

        band_output = MSCBBlock(block_input)
        band_output = Conv2D(filters=112, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(band_output)
        band_output = MaxPooling2D(pool_size=(1, 5), strides=(1, 5), padding='same')(band_output)

        for j in range(3):
          if model in ['MSCNN+DE', 'MSCNN+all']:
              DE_input = DE_input_tensor[:, j, :, i]
              DE_input_expanded = tf.expand_dims(DE_input, axis=-1)
              Conv_DE = Conv1D(filters=112, kernel_size=1, activation='relu')(DE_input_expanded)
          if model in ['MSCNN+NPS', 'MSCNN+all']:
              NPS_input = NPS_input_tensor[:, j, :, i]
              NPS_input_expanded = tf.expand_dims(NPS_input, axis=-1)
              Con_NPS = Conv1D(filters=112, kernel_size=1, activation='relu')(NPS_input_expanded)

            band_output_j = band_output[:, j, :, :]
            band_flattend = Flatten()(band_output_j)
            if model == 'MSCNN+DE':
              DE_flattened = Flatten()(Conv_DE)
              concat = Concatenate()([band_flattend , DE_flattened])
            if model == 'MSCNN+NPS':
              NPS_flattened = Flatten()(Con_NPS)
              concat = Concatenate()([band_flattend, NPS_flattened])
            if model == 'MSCNN+all':
              DE_flattened = Flatten()(Conv_DE)
              NPS_flattened = Flatten()(Con_NPS)
              concat = Concatenate()([band_flattend , DE_flattened, NPS_flattened])

            concatenated_outputs.append(concat)

    concatenated_output = Concatenate()(concatenated_outputs)

    # Add a fully connected layer for classification
    output_layer = Dense(400, activation='relu', kernel_regularizer=l2(0.02))(concatenated_output)
    output_layer = Dropout(0.5)(output_layer)

    # Final classification layer
    output_layer = Dense(300, activation='relu')(output_layer)
    output_layer = Dense(2, activation='softmax')(output_layer)

    # Create the final model
    if model == 'MSCNN+DE':
      model = Model(inputs=[input_tensor], outputs=output_layer)
    if model == 'MSCNN+NPS':
      model = Model(inputs=[input_tensor, DE_input_tensor], outputs=output_layer)
    if model == 'MSCNN+all':
      model = Model(inputs=[input_tensor, DE_input_tensor, NPS_input_tensor], outputs=output_layer)

    optimizer = SGD(learning_rate=lr_schedule)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
