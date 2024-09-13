from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import keras.layers as layers
from keras.models import Model
from keras.applications.vgg16 import VGG16
import numpy as np


def FSLmodel(encoder='VGG', input_size=(256, 256, 1), k_shot=1, learning_rate=1e-4, no_weight=False):
    VGG_WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                               'releases/download/v0.1/'
                               'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("Se encontraron GPUs. Usando GPU para el entrenamiento.")
        # Configurar una estrategia de distribución para usar una única GPU
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        print("No se encontraron GPUs. Usando CPU para el entrenamiento.")
        # Configurar una estrategia de distribución para usar la CPU
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    with strategy.scope():
        def vgg_encoder(input_size=(256, 256, 3), no_weight=False):
            img_input = layers.Input(input_size)
            modelvgg = VGG16(weights='imagenet', include_top=False)
            block1_conv1 = modelvgg.get_layer('block1_conv1').get_weights()
            weights, biases = block1_conv1
            weights = np.transpose(weights, (3, 2, 0, 1))
            kernel_out_channels, kernel_in_channels, kernel_rows, kernel_columns = weights.shape
            grayscale_weights = np.zeros(
                (kernel_out_channels, 1, kernel_rows, kernel_columns))
            gray_w = (weights[:, 0, :, :] * 0.2989) + (weights[:,
                                                               1, :, :] * 0.5870) + (weights[:, 2, :, :] * 0.1140)
            gray_w = np.expand_dims(gray_w, axis=1)
            grayscale_weights = gray_w
            grayscale_weights = np.transpose(grayscale_weights, (2, 3, 1, 0))

            # Block 1
            if input_size[2] == 1:
                xblock1 = layers.Conv2D(64, (3, 3),
                                        activation='relu',
                                        padding='same',
                                        name='block1_conv1_gray')
                x = xblock1(img_input)
            else:
                x = layers.Conv2D(64, (3, 3),
                                  activation='relu',
                                  padding='same',
                                  name='block1_conv1')(img_input)

            x = layers.Conv2D(64, (3, 3),
                              activation='relu',
                              padding='same',
                              name='block1_conv2')(x)
            x = layers.MaxPooling2D(
                (2, 2), strides=(2, 2), name='block1_pool')(x)

            # Block 2
            x = layers.Conv2D(128, (3, 3),
                              activation='relu',
                              padding='same',
                              name='block2_conv1')(x)
            x = layers.Conv2D(128, (3, 3),
                              activation='relu',
                              padding='same',
                              name='block2_conv2')(x)
            x = layers.MaxPooling2D(
                (2, 2), strides=(2, 2), name='block2_pool')(x)

            # Block 3
            x = layers.Conv2D(256, (3, 3),
                              activation='relu',
                              padding='same',
                              name='block3_conv1')(x)
            x = layers.Conv2D(256, (3, 3),
                              activation='relu',
                              padding='same',
                              name='block3_conv2')(x)
            x = layers.Conv2D(256, (3, 3),
                              activation='relu',
                              padding='same', dilation_rate=(2, 2),
                              name='block3_conv3')(x)

            # Block 4
            x = layers.Conv2D(512, (3, 3),
                              activation='relu',
                              padding='same',
                              name='block4_conv1')(x)
            x = layers.Conv2D(512, (3, 3),
                              activation='relu',
                              padding='same',
                              name='block4_conv2')(x)
            x = layers.Conv2D(512, (3, 3),
                              activation='relu',
                              padding='same',
                              name='block4_conv3')(x)

            # Create model.
            model = Model(img_input, x, name='vgg16_model_with_block1-4')

            # Load weights.
            weights_path = tf.keras.utils.get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                VGG_WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='6d6bbae143d832006294945121d1f1fc')
            if not (no_weight):
                model.load_weights(weights_path, by_name=True)
            if input_size[2] == 1:
                xblock1.set_weights([grayscale_weights, biases])
            return model

        def common_representation(x1, x2):
            x = layers.concatenate([x1, x2], axis=3)
            x = layers.Conv2D(128, 3, padding='same',
                              kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization(axis=3)(x)
            x = layers.Activation('relu')(x)
            return x

        if encoder == 'VGG':
            encoder = vgg_encoder(input_size=input_size, no_weight=no_weight)
        else:
            print('Encoder is not defined yet')

        ###################################### K-shot learning #####################################
        # K shot
        S_input2 = layers.Input(
            (k_shot, input_size[0], input_size[1], input_size[2]))
        Q_input2 = layers.Input(input_size)
        S_mask2 = layers.Input(
            (k_shot, int(input_size[0]/4), int(input_size[1]/4), 1))

        kshot_encoder = tf.keras.models.Sequential()
        kshot_encoder.add(layers.TimeDistributed(encoder, input_shape=(
            k_shot, input_size[0], input_size[1], input_size[2])))

        s_encoded = kshot_encoder(S_input2)
        q_encoded = encoder(Q_input2)
        s_encoded = layers.TimeDistributed(layers.Conv2D(
            128, (3, 3), activation='relu', padding='same'))(s_encoded)
        q_encoded = layers.Conv2D(
            128, (3, 3), activation='relu', padding='same')(q_encoded)

        # Global Representation
        # s_encoded  = GlobalAveragePooling2D_r(S_mask2)(s_encoded)

        repc = int(s_encoded.shape[4])
        m = tf.keras.backend.repeat_elements(S_mask2, repc, axis=4)
        x = layers.multiply([s_encoded, m])
        repx = int(x.shape[2])
        repy = int(x.shape[3])
        x = (tf.keras.backend.sum(x, axis=[
             1, 2, 3], keepdims=True) / (tf.keras.backend.sum(m, axis=[1, 2, 3], keepdims=True)))
        x = tf.keras.layers.Reshape(target_shape=(
            np.int32(x.shape[2]), np.int32(x.shape[3]), np.int32(x.shape[4])))(x)
        x = tf.keras.backend.repeat_elements(x, repx, axis=1)
        s_encoded = tf.keras.backend.repeat_elements(x, repy, axis=2)

        # Common Representation of Support and Query sample
        Bi_rep = common_representation(s_encoded, q_encoded)

        # Decode to query segment
        x = layers.Conv2D(128, 3, padding='same',
                          kernel_initializer='he_normal')(Bi_rep)
        x = layers.BatchNormalization(axis=3)(x)
        x = layers.Activation('relu')(x)
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2D(128, 3, padding='same',
                          kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization(axis=3)(x)
        x = layers.Activation('relu')(x)
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2D(128, 3, padding='same',
                          kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization(axis=3)(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, 3,  activation='relu',
                          padding='same', kernel_initializer='he_normal')(x)
        x = layers.Conv2D(2, 3,   activation='relu',
                          padding='same', kernel_initializer='he_normal')(x)
        final = layers.Conv2D(1, 1,   activation='sigmoid')(x)

        seg_model = Model(inputs=[S_input2, S_mask2, Q_input2], outputs=final)

        seg_model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        device = strategy.extended._device
        if 'CPU' in device:
            print("El modelo se encuentra alojado en la CPU")
        else:
            print("El modelo se encuentra alojado en la GPU")
        return seg_model


if __name__ == "__main__":
    model = FSLmodel()
    model.summary()
