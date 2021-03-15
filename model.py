# -*- coding: utf-8 -*-
"""XGAN-Roger

Colab file
    https://colab.research.google.com/drive/1i1tZpF2uRxTROlnogScdJ2pYBnZyM9qm
"""

import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from datetime import datetime


class EncoderPart(layers.Layer):
  """
  Description
  -----------
    Part of the encoder which is different between the cartoon set images
    and the real face images
  
  Returns
  -------
  tf.keras.layers.Layer Object
  """
  
  def __init__(self):
    super(EncoderPart, self).__init__()

    self.conv1 = layers.Conv2D(64, (4, 4), input_shape=[64, 64, 3], strides=(2, 2), padding='same', use_bias=False)
    self.bn1 = layers.BatchNormalization()
    self.r1 = layers.ReLU()

    self.conv2 = layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', use_bias=False)
    self.bn2 = layers.BatchNormalization()
    self.r2 = layers.ReLU()

  def call(self, image):
    x = self.conv1(image)
    x = self.bn1(x)
    x = self.r1(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.r2(x)
    return x

class SharedEncoder(layers.Layer):
  """
  Description
  -----------
    Part of the encoder which shared between the two encoders entries.
  
  Returns
  -------
  tf.keras.layers.Layer Object
  """
  def __init__(self):
    super(SharedEncoder, self).__init__()

    self.conv3 = layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same', use_bias=False)
    self.bn3 = layers.BatchNormalization()
    self.r3 = layers.ReLU()

    self.conv4 = layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', use_bias=False)
    self.bn4 = layers.BatchNormalization()
    self.r4 = layers.ReLU()

    self.flatten = layers.Flatten()
    self.fc1 = layers.Dense(1024, activation='relu') # Add activation?
    self.fc2 = layers.Dense(1024, activation='relu')

  def call(self, image):
    x = self.conv3(image)
    x = self.bn3(x)
    x = self.r3(x)

    x = self.conv4(x)
    x = self.bn4(x)
    x = self.r4(x)

    x = self.flatten(x)
    
    x = self.fc1(x)
    x = self.fc2(x)

    return x

class Encoder(layers.Layer):
  """
  Description
  -----------
    Encoder of the XGAN.

    See that in the call method there is a boolean which indicates if the image
    we are introducing is from the cartoon dataset or not.
  Returns
  -------
  tf.keras.layers.Layer Object
  """
  def __init__(self):
    super(Encoder, self).__init__()

    self.encoder_cartoon = EncoderPart()
    self.encoder_realface = EncoderPart()
    self.encoder_shared = SharedEncoder()

  def call(self, image, is_cartoon):
    if is_cartoon:
      return self.encoder_shared(self.encoder_cartoon(image))
    return self.encoder_shared(self.encoder_realface(image))


class SharedDecoder(layers.Layer):
  """
  Description
  -----------
    Shared part of the decoder. The input is the embedding we obtained from the 
    encoder.

  Returns
  -------
  tf.keras.layers.Layer Object
  """

  def __init__(self):
    super(SharedDecoder, self).__init__()

    self.fc = layers.Dense(4*4*1024, input_shape=(1024,))
    self.reshape = layers.Reshape((4, 4, 1024))

    self.deconv1 = layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', use_bias=False)
    self.bn1 = layers.BatchNormalization()
    self.r1 = layers.ReLU()

    self.deconv2 = layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False)
    self.bn2 = layers.BatchNormalization()
    self.r2 = layers.ReLU()

  def call(self, embedding):
    x = self.fc(embedding)
    x = self.reshape(x)

    x = self.deconv1(x)
    x = self.bn1(x)
    x = self.r1(x)

    x = self.deconv2(x)
    x = self.bn2(x)
    x = self.r2(x)
    
    return x

class DecoderPart(layers.Layer):
  """
  Description
  -----------
    Decoder part different. The input comes from the shared decoder.
    encoder.

  Returns
  -------
  tf.keras.layers.Layer Object
  """

  def __init__(self):
    super(DecoderPart, self).__init__()

    self.deconv3 = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False)
    self.bn3 = layers.BatchNormalization()
    self.r3 = layers.ReLU()

    self.deconv4 = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False)
    self.bn4 = layers.BatchNormalization()
    self.r4 = layers.ReLU()

    self.deconv5 = layers.Conv2DTranspose(3, (1, 1), strides=(1, 1), padding='same', use_bias=False, activation='tanh')

  def call(self, x):
    x = self.deconv3(x)
    x = self.bn3(x)
    x = self.r3(x)
    x = self.deconv4(x)
    x = self.bn4(x)
    x = self.r4(x)
    x = self.deconv5(x)
    
    return x

class Decoder(layers.Layer):
  """
  Description
  -----------
    Decoder of the XGAN.

    See that in the call method there is a boolean which indicates if the image
    we are introducing is from the cartoon dataset or not.
  Returns
  -------
  tf.keras.layers.Layer Object
  """
  def __init__(self):
    super(Decoder, self).__init__()

    self.decoder_cartoon = DecoderPart()
    self.decoder_realface = DecoderPart()
    self.decoder_shared = SharedDecoder()

  def call(self, image):
    img_cartoon = self.decoder_cartoon(self.decoder_shared(image))
    img_real = self.decoder_realface(self.decoder_shared(image))
    return img_cartoon, img_real

class Generator(layers.Layer):
  """
  Description
  -----------
    GENERATOR of the XGAN. It's composed by the encoder-decoder (X)

    See that in the call method there is a boolean which indicates if the image
    we are introducing is from the cartoon dataset or not.
  Returns
  -------
  tf.keras.layers.Layer Object
  """
  def __init__(self):
    super(Generator, self).__init__()

    self.encoder = Encoder()
    self.decoder = Decoder()

  def call(self, image, is_cartoon):
    embedding = self.encoder(image, is_cartoon)
    image_a , image_b = self.decoder(embedding)
    return {'image_a': image_a, 'image_b': image_b, 'embedding':embedding}

class Discriminator(layers.Layer):
  """
  Description
  -----------
    DISCRIMINATOR of the XGAN. It's composed by the encoder-decoder (X)

  Returns
  -------
  tf.keras.layers.Layer Object
  """
  def __init__(self):
    super(Discriminator, self).__init__()
        
    self.conv1 = layers.Conv2D(64, (4, 4), input_shape=[64, 64, 3], strides=(2, 2), padding='same', use_bias=False)
    self.lrelu1 = layers.LeakyReLU()
    self.drop1 = layers.Dropout(0.2)

    self.conv2 = layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', use_bias=False)
    self.bn2 = layers.BatchNormalization()
    self.lrelu2 = layers.LeakyReLU()
    self.drop2 = layers.Dropout(0.2)

    self.conv3 = layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same', use_bias=False)
    self.bn3 = layers.BatchNormalization()
    self.lrelu3 = layers.LeakyReLU()
    self.drop3 = layers.Dropout(0.2)

    self.conv4 = layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', use_bias=False)
    self.bn4 = layers.BatchNormalization()
    self.lrelu4 = layers.LeakyReLU()
    self.drop4 = layers.Dropout(0.2)

    self.flatten = layers.Flatten()
    self.sigmoid = layers.Dense(1, activation='sigmoid')
  
  def call(self, image):
    x = self.drop1(self.lrelu1(self.conv1(image)))
    x = self.drop2(self.lrelu2(self.bn2(self.conv2(x))))
    x = self.drop3(self.lrelu3(self.bn3(self.conv3(x))))
    x = self.drop4(self.lrelu4(self.bn4(self.conv4(x))))

    return self.sigmoid(self.flatten(x))


class Cdann(layers.Layer):
  """
  Cdann
  -----------
    Embedding classifier of the XGAN.

  Returns
  -------
  tf.keras.layers.Layer Object
  """
  def __init__(self):
    super(Cdann, self).__init__()
    self.dense1 = layers.Dense(256, activation='relu', input_shape=(16, 1025))
    self.dense2 = layers.Dense(64, activation='relu', input_shape=(16, 256))
    self.dense3 = layers.Dense(1, activation='softmax', input_shape=(16, 64))

  def call(self, embedding):
    return self.dense3(self.dense2(self.dense1(embedding)))

class XGAN(tf.keras.Model):
  """
  Description
  -----------
    XGAN model. Its composed by:
     * Generator (encoder + decoder)
     * Discriminator
     * Explain or add all the losses here #TODO

  Returns
  -------
  tf.keras.Model Object
  """

  def __init__(self, batch_size=16):
    super(XGAN, self).__init__()
    self.discriminator = Discriminator()
    self.generator = Generator()
    self.cdann = Cdann()
    self.batch_size = batch_size

    self.gen_loss_metric = tf.keras.metrics.Mean('gen_loss_metric', dtype=tf.float32)
    self.disc_loss_metric = tf.keras.metrics.Mean('disc_loss_metric', dtype=tf.float32)
    self.autoencoder_loss_metric = tf.keras.metrics.Mean('autoencoder_loss_metric', dtype=tf.float32)
    self.semantic_loss_metric = tf.keras.metrics.Mean('semantic_loss_metric', dtype=tf.float32)
    self.domain_adversarial_loss_metric = tf.keras.metrics.Mean('domain_adversarial_loss_metric', dtype=tf.float32)
    self.epoch_step = 0
    train_log_dir = 'logs/' + datetime.now().strftime("%Y%m%d-%H%M%S") + '/train'
    self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)

  def abs_criterion(self, in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


  def compile(self, d_optimizer, g_optimizer, loss):
      super(XGAN, self).compile()
      self.d_optimizer = d_optimizer
      self.g_optimizer = g_optimizer
      self.loss_fn = loss

  def autoencoder_loss(self, cartoon_dataset, real_dataset, cartoon_from_cartoon_dataset, real_from_real_dataset):
    """
      This is the first loss of the generator and the main use is to train the autoencoder to work properly. 
      From the cartoon dataset, we pass it through the autoencoder. The caroon images created are compared with the images from the cartoon dataset.
      The same procedure is done with the real face image dataset.
    """
    loss_autoncoder_cartoon = tf.losses.mean_squared_error(cartoon_dataset, cartoon_from_cartoon_dataset)
    loss_autoncoder_real = tf.losses.mean_squared_error(real_dataset, real_from_real_dataset)
    return loss_autoncoder_cartoon + loss_autoncoder_real

  def semantic_consistency_feedback_loss(self, generator_result_from_cartoon, generator_result_from_real):
    """
      This loss is used to learn the semantic consistency between domains. The procedue of the loss is:
        1 Given a cartoon image we generate the embedding
        2 From the embedding we generate the real face
        3 With the real face we generate the embedding again
        * The embeddings of point 1 and point 3 should be the same (calculate the loss)
    """
    # From cartoon dataset
    # result = self.generator(generator_result_from_cartoon['image_a'], is_cartoon = False)
    # from_cartoon_semantic_loss =  tf.reduce_mean(tf.abs(result['embedding'] - generator_result_from_cartoon['embedding']))

    # From real faces
    result = self.generator(generator_result_from_real['image_a'], is_cartoon = True)
    from_real_semantic_loss =  tf.reduce_mean(tf.abs(result['embedding'] - generator_result_from_real['embedding']))
    return from_real_semantic_loss

  def domain_adversarial_loss(self, cartoon_from_cartoon_dataset, real_from_real_dataset):
    cdann_cartoon = self.cdann(cartoon_from_cartoon_dataset['embedding'])
    cdann_real = self.cdann(real_from_real_dataset['embedding'])
    domain_cartoon = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=cdann_cartoon, labels=tf.zeros_like(cdann_cartoon)))
    domain_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=cdann_real, labels=tf.ones_like(cdann_real)))

    return domain_cartoon + domain_real


  def generator_loss(self, img_cartoon_dataset, img_reals_dataset, generator_result_from_real, generator_result_from_cartoon):
    autoencoder_loss = self.autoencoder_loss(img_cartoon_dataset, img_reals_dataset, generator_result_from_cartoon['image_a'], generator_result_from_real['image_b'])
    semantic_loss = self.semantic_consistency_feedback_loss(generator_result_from_cartoon, generator_result_from_real)
    domain_adversarial_loss = self.domain_adversarial_loss(generator_result_from_cartoon, generator_result_from_real)

    self.autoencoder_loss_metric(autoencoder_loss)
    self.semantic_loss_metric(semantic_loss)
    self.domain_adversarial_loss_metric(domain_adversarial_loss)

    return autoencoder_loss + semantic_loss + domain_adversarial_loss

  def discriminator_loss(self, generator_result_from_real, img_cartoon_dataset):
    real_output = self.discriminator(img_cartoon_dataset)
    fake_output = self.discriminator(generator_result_from_real['image_a'])
    real_loss = self.loss_fn(tf.ones_like(real_output), real_output) # correctly classified from the dataset
    fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output) # correctly classified from the generator
    return real_loss + fake_loss

  def on_epoch_begin(self, epoch, logs=None):
    self.epoch_step = 0

  def on_batch_end(self, batch, logs=None):
    self.epoch_step += 1

  def train_step(self, dataset):
    img_cartoon_dataset = dataset[:,self.batch_size:,:,:,:]
    img_reals_dataset = dataset[:,0:self.batch_size,:,:,:]
    img_cartoon_dataset = img_cartoon_dataset[0]
    img_reals_dataset = img_reals_dataset[0]

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generator_result_from_real = self.generator(img_reals_dataset, is_cartoon = False) # image_a (carton from real), image_b(real from real)
      generator_result_from_cartoon = self.generator(img_cartoon_dataset, is_cartoon = True) # image_a (carton from cartoon), image_b(real from carton)
      
      # Discriminator
      disc_loss = self.discriminator_loss(generator_result_from_real, img_cartoon_dataset)

      # Generator
      gen_loss = self.generator_loss(img_cartoon_dataset, img_reals_dataset, generator_result_from_real, generator_result_from_cartoon)

      self.disc_loss_metric(disc_loss)
      self.gen_loss_metric(gen_loss)

      gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
      gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
      self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
      self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

      with self.train_summary_writer.as_default():
        tf.summary.scalar('disc_loss', self.disc_loss_metric.result(), step=self.epoch_step)
        tf.summary.scalar('autoencoder_loss', self.autoencoder_loss_metric.result(), step=self.epoch_step)
        tf.summary.scalar('semantic_loss', self.semantic_loss_metric.result(), step=self.epoch_step)
        tf.summary.scalar('domain_adversarial_loss', self.domain_adversarial_loss_metric.result(), step=self.epoch_step)
        tf.summary.scalar('gen_loss', self.gen_loss_metric.result(), step=self.epoch_step)
      return {"Discriminator loss": disc_loss, "Generator loss": gen_loss}

