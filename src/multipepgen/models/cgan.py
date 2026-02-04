import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Reshape, Bidirectional, Dense, Concatenate

from multipepgen.config import LABELS
from multipepgen.utils.preprocessing import ohe
from multipepgen.utils.postprocessing import one_hot_max_matrix

class ConditionalGAN(keras.Model):
    def __init__(self, sequence_length=35, vocab_size=21, latent_dim=100, num_classes=7):
        super(ConditionalGAN, self).__init__()
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Trackers
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        
        # Create models
        self.generator = self.create_generator_model()
        self.discriminator = self.create_discriminator_model()

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def create_generator_model(self):
        generator_in_channels = self.latent_dim + self.num_classes
        matrix_shape = (self.sequence_length, self.vocab_size)
        
        model = Sequential(name='generator')
        model.add(Input(shape=(generator_in_channels,)))
        model.add(Reshape((1, generator_in_channels)))
        model.add(Bidirectional(layers.GRU(int(np.prod(matrix_shape)), return_sequences=True)))
        model.add(Bidirectional(layers.GRU(int(np.prod(matrix_shape)))))
        model.add(Dense(int(np.prod(matrix_shape)), activation='tanh'))
        model.add(Reshape((matrix_shape[0], matrix_shape[1], 1)))
        return model

    def create_discriminator_model(self):
        discriminator_in_channels = 1 + self.num_classes
        matrix_shape = (self.sequence_length, self.vocab_size)
        
        d_model = Sequential(name='discriminator')
        d_model.add(Input(shape=(matrix_shape[0], matrix_shape[1], discriminator_in_channels)))
        d_model.add(Reshape((1, matrix_shape[0] * matrix_shape[1] * discriminator_in_channels)))
        d_model.add(Bidirectional(layers.GRU(int(np.prod(matrix_shape)))))
        d_model.add(Dense(1, activation='sigmoid'))
        return d_model

    def train_step(self, data):
        # Unpack the data.
        # Check if data comes as tuple/list or just features.
        # In standard fit(x), data is x.
        # Expecting data to be (real_matrices, labels) if passed as a dataset or tuple
        if isinstance(data, tuple):
             real_matrix, one_hot_labels = data
        else:
             # Handle case where only X is passed? Usually GANs define custom data feeding.
             # Assuming standard Keras format: (x, y)
             real_matrix = data[0] # assuming x
             # labels might be implicit or passed differently. 
             # However, the previous notebook code had `real_matrix, one_hot_labels = data`.
             # This implies the dataset yields (matrix, labels).
             # Let's assume standard behavior where `data` unpacks to x,y
             real_matrix, one_hot_labels = data

        # Cast types if necessary
        real_matrix = tf.cast(real_matrix, dtype=tf.float32)
        one_hot_labels = tf.cast(one_hot_labels, dtype=tf.float32)

        batch_size = tf.shape(real_matrix)[0]

        # Add dummy dimensions to the labels so that they can be concatenated with
        # the matrixs. This is for the discriminator.
        # shape: (batch, seq, vocab, num_classes)
        matrix_one_hot_labels = one_hot_labels[:, :, None, None]
        
        # We need to tile the labels to match the image dimensions (seq_len x vocab_size)
        # matrix_one_hot_labels is (batch, num_classes, 1, 1) initially if one_hot_labels is (batch, num_classes)
        # But wait, one_hot_labels is (batch, num_classes).
        # We need to reshape it first properly.
        
        # Original code:
        # matrix_one_hot_labels = one_hot_labels[:, :, None, None] -> (batch, num_classes, 1, 1)
        # matrix_one_hot_labels = tf.repeat(matrix_one_hot_labels, repeats=[MATRIX_SIZE[0] * MATRIX_SIZE[1]])
        # matrix_one_hot_labels = tf.reshape(...)
        
        # Let's do it cleanly with tf.tile
        # one_hot_labels: (batch, num_classes)
        # Target: (batch, seq_len, vocab_size, num_classes)
        
        labels_reshaped = tf.reshape(one_hot_labels, (batch_size, 1, 1, self.num_classes))
        matrix_one_hot_labels = tf.tile(labels_reshaped, [1, self.sequence_length, self.vocab_size, 1])
        

        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Decode the noise (guided by labels) to fake matrixs.
        generated_matrix = self.generator(random_vector_labels)

        # Combine them with real matrixs. Note that we are concatenating the labels
        # with these matrixs here.
        
        # generated_matrix: (batch, seq, vocab, 1)
        # matrix_one_hot_labels: (batch, seq, vocab, num_classes)
        fake_matrix_and_labels = tf.concat([generated_matrix, matrix_one_hot_labels], -1)
  
        real_matrix_and_labels = tf.concat([real_matrix, matrix_one_hot_labels], -1)
        
        combined_matrixs = tf.concat(
            [fake_matrix_and_labels, real_matrix_and_labels], axis=0
        )

        # Assemble labels discriminating real from fake matrixs.
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_matrixs)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Assemble labels that say "all real matrixs".
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_matrixs = self.generator(random_vector_labels)
            fake_matrix_and_labels = tf.concat([fake_matrixs, matrix_one_hot_labels], -1)
            predictions = self.discriminator(fake_matrix_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }

    def generate_class(self, num_sequences, classes):
        """
        Generate sequences for specific classes.
        classes: List/vector of class names present in LABELS.
        """
        conditional_vector = []
        for label in LABELS:
            conditional_vector.append(1 if (label in classes) else 0)
        
        # Replicate for vectorization
        # Input shape to generator is (batch, latent_dim + num_classes)
        
        noise = np.random.normal(0, 1, (num_sequences, self.latent_dim))
        labels_vec = np.array(conditional_vector).reshape(1, -1)
        labels_mat = np.repeat(labels_vec, num_sequences, axis=0)
        
        latent_input = np.concatenate((noise, labels_mat), axis=1)
        
        gen_imgs = self.generator.predict(latent_input, verbose=0)
        
        pred_gen = []
        for i in range(num_sequences):
            # Apply postprocessing
            matrix = gen_imgs[i, :, :, 0]
            
            # one_hot_max_matrix returns one-hot like structure
            matrix_one_hot = np.array(one_hot_max_matrix(matrix))
            
            # Inverse transform expects 2D array (n_samples, n_features)
            # ohe was fitted on single amino acids.
            # sequences are (35, 21).
            
            output_sequence = ohe.inverse_transform(matrix_one_hot)
            
            output_sequence_str = ""
            for aa in output_sequence:
                if str(aa[0]) == '_':
                    break
                else:
                    output_sequence_str += str(aa[0])
            pred_gen.append(output_sequence_str)
            
        df_gen = pd.DataFrame({"sequence": pred_gen})
        return df_gen

    def generate_class_random(self, num_sequences):
        """
        Generate sequences with random class assignments.
        """
        pred_gen = []
        # Batch generation for efficiency
        
        # Generate random binary vectors for classes
        # Note: Original code used np.random.randint(2, size=len(LABELS)).
        # This means multi-label is possible.
        
        labels_mat = np.random.randint(2, size=(num_sequences, self.num_classes))
        noise = np.random.normal(0, 1, (num_sequences, self.latent_dim))
        
        latent_input = np.concatenate((noise, labels_mat), axis=1)
        
        gen_imgs = self.generator.predict(latent_input, verbose=0)
        
        for i in range(num_sequences):
            matrix = gen_imgs[i, :, :, 0]
            matrix_one_hot = np.array(one_hot_max_matrix(matrix))
            output_sequence = ohe.inverse_transform(matrix_one_hot)
            
            output_sequence_str = ""
            for aa in output_sequence:
                if str(aa[0]) == '_':
                    break
                else:
                    output_sequence_str += str(aa[0])
            pred_gen.append(output_sequence_str)
            
        df_gen = pd.DataFrame({"sequence": pred_gen})
        return df_gen
