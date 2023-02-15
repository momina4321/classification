import numpy as np
import pandas as pd
import os
import tensorflow as tf
import utils
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


class SCNN:
    def __init__(self, batch_size=None, epochs=None, verbose=None):
        self.callbacks = None
        self.embedding_size = 128  # embedding size
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

        self.siamese_network, self.embedding_module, self.relation_module = self.build_SCNN_model()

        self.SCNN_accs = []
        self.SCNN_recalls = []
        self.SCNN_precisions = []
        self.SCNN_F1s = []

    def build_SCNN_model(self, input_shape):
        # we have to build a SCNN model that contains an embedding and relational module
        tf.keras.backend.clear_session()
        # we need left and right inputs to be fed into the siamese network
        left_input = tf.keras.layers.Input(input_shape)
        right_input = tf.keras.layers.Input(input_shape)
        inputs1 = tf.keras.layers.Input(shape=input_shape)

        # there are two embedding modules and each has 3 1D CNN architecture
        # Convolutional Block 1
        conv1 = tf.keras.layers.Conv1D(filters=128, kernel_size=7, activation='relu', kernel_initializer='he_uniform')(
            inputs1)
        bn1 = tf.keras.layers.BatchNormalization()(conv1)
        drop1 = tf.keras.layers.Dropout(0.4)(bn1)
        pool1 = tf.keras.layers.MaxPooling1D(pool_size=3)(drop1)
        # Convolutional Block 2
        conv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu', kernel_initializer='he_uniform')(
            pool1)
        bn2 = tf.keras.layers.BatchNormalization()(conv2)
        drop2 = tf.keras.layers.Dropout(0.4)(bn2)
        pool2 = tf.keras.layers.MaxPooling1D(pool_size=3)(drop2)
        # Convolutional Block 3
        conv3 = tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu', kernel_initializer='he_uniform')(
            pool2)
        bn3 = tf.keras.layers.BatchNormalization()(conv3)
        drop3 = tf.keras.layers.Dropout(0.4)(bn3)
        pool3 = tf.keras.layers.MaxPooling1D(pool_size=2)(drop3)
        flat3 = tf.keras.layers.Flatten()(pool3)

        # Embedding layer is the dense layer
        embedding = tf.keras.layers.Dense(self.embedding_size, activation='relu')(flat3)  # dense layer of size 128

        # Defining the embedding module
        embedding_module = tf.keras.Model(inputs=inputs1, outputs=embedding, name="Embedding_Module")

        input_l = tf.keras.layers.Input(shape=self.embedding_size)
        input_r = tf.keras.layers.Input(shape=self.embedding_size)

        L1_layer = tf.keras.layers.Lambda(lambda tensors: tf.keras.backend.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([input_l, input_r])
        similarity = tf.keras.layers.Dense(1, activation='sigmoid')(L1_distance)

        # Defining the relation module
        relation_module = tf.keras.models.Model(inputs=[input_l, input_r], outputs=similarity,
                                                name='Relation_Module')

        # defining two embedding modules
        embedded_l = embedding_module(left_input)
        embedded_r = embedding_module(right_input)

        similarity_score = relation_module([embedded_l, embedded_r])  # find similarity score between two embeddings

        # Defining the entire Siamese Network
        siamese_network = tf.keras.Model(inputs=[left_input, right_input], outputs=similarity_score,
                                         name="Siamese_Network")

        siamese_network.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(),
                                metrics=['accuracy'])

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                         min_lr=0.0001)

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(monitor='val_loss',
                                                              save_best_only=True, verbose=1)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)

        self.callbacks = [model_checkpoint, early_stopping]

        return siamese_network, embedding_module, relation_module
