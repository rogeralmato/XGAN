from dataset import Dataset
from model import XGAN

import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from random import randrange
import os

# ------------------------------------------------
# -------------------CONFIG-----------------------
# ------------------------------------------------
num_epochs = 5000

betas = (0.5, 0.999)
lr = 0.0002

batch_size = 16
num_val_samples = 16
num_input_channels = 1
big_dataset = False

CARTOON_DATASET_PATH = "dataset_cartoon"
REAL_DATASET_PATH = "dataset_real"


now = datetime.now()
results_folder_name = f"RESULTS/{now.day}-{now.month}-{now.year}_{now.hour}:{now.minute}:{now.second}"
os.mkdir(str(os.getcwd()) + '/' + results_folder_name)


# ------------------------------------------------
# -------------------DATASET----------------------
# ------------------------------------------------

dataset_obj = Dataset(CARTOON_DATASET_PATH, REAL_DATASET_PATH, batch_size)
dataset = dataset_obj.dataset_numpy


# ------------------------------------------------
# -----------------CALLBACKS----------------------
# ------------------------------------------------

class DisplayImages(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs=None):
    image_batch_real = dataset[0]
    result = xgan.generator(image_batch_real, is_cartoon = False, training=False)
    cartoon = result['image_a']
    real = result['image_b']

    fig = plt.figure(figsize=(6,12))
    # real images
    for i in range(16):
        fig.add_subplot(8, 4, i+1)
        img = image_batch_real[i-16, :, :, :] * 127.5 + 127.5
        plt.imshow(img.astype("uint8"))
        plt.axis('off')
    # cartoon images predicted
    for i in range(16, 32):
        fig.add_subplot(8, 4, i+1)
        img_prediction = cartoon[i-16, :, :, :] * 127.5 + 127.5
        plt.imshow(img_prediction.numpy().astype("uint8"))
        plt.axis('off')

    plt.savefig(results_folder_name + '/' + 'image_at_epoch_{:04d}.png'.format(epoch))

    if epoch % 50 == 0:
        n = len(dataset)
        random = randrange(n)
        batch = dataset[random]
        result = xgan.generator(batch, is_cartoon = False, training=False)
        cartoon = result['image_a']
        fig = plt.figure(figsize=(6,12))
    # real images
    for i in range(16):
        fig.add_subplot(8, 4, i+1)
        img = batch[i-16, :, :, :] * 127.5 + 127.5
        plt.imshow(img.astype("uint8"))
        plt.axis('off')
    # cartoon images predicted
    for i in range(16, 32):
        fig.add_subplot(8, 4, i+1)
        img_prediction = cartoon[i-16, :, :, :] * 127.5 + 127.5
        plt.imshow(img_prediction.numpy().astype("uint8"))
        plt.axis('off')

    plt.savefig(results_folder_name + '/' + 'CHECK_{:04d}.png'.format(epoch))

# ------------------------------------------------
# --------------------TRAIN-----------------------
# ------------------------------------------------

xgan = XGAN()
xgan.compile(
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=betas[0], beta_2=betas[1]),
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=betas[0], beta_2=betas[1]),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
)

xgan.fit(dataset, epochs=num_epochs, callbacks=[DisplayImages()])