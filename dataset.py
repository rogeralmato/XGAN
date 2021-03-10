import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop
import numpy as np

class Dataset():
    """
    Dataset class object which combines the cartoon dataset and the real faces dataset
    """
    def __init__(self, path_cartoon_dataset, path_real_faces_dataset, batch_size):
        # Image cropper
        self.cropper = CenterCrop(height=64, width=64)

        self.training_cartoon = tf.keras.preprocessing.image_dataset_from_directory(
            path_cartoon_dataset,
            image_size=(83, 83),
            batch_size=batch_size,
            label_mode=None,
            shuffle=True
        )
        self.training_cartoon = self.training_cartoon.map(self.preprocessing_image)

        self.training_face = tf.keras.preprocessing.image_dataset_from_directory(
            path_real_faces_dataset,
            image_size=(83, 83),
            batch_size=batch_size,
            label_mode=None,
            shuffle=True
        )
        self.training_face = self.training_face.map(self.preprocessing_image)

        self.dataset_numpy = self.join_datasets_to_numpy()
    
    def preprocessing_image(self, image):
        """Image preprocessing"""
        normalized_image = tf.cast((image - 127.5) / 127.5, tf.float32)
        return self.cropper(normalized_image)

    def print_length(self):
        print(f"cartoon -> {len(self.training_cartoon)}")
        print(f"real -> {len(self.training_face)}")

    def min_length(self):
        return min(len(self.training_cartoon), len(self.training_face))

    def join_datasets_to_numpy(self):
        data = None
        real_iterator = self.training_face.as_numpy_iterator()
        cartoon_iterator = self.training_cartoon.as_numpy_iterator()
        for i in range(self.min_length()):
            real_batch = real_iterator.next()
            cartoon_batch = cartoon_iterator.next()
            # print(f"{len(real_batch)} - {len(cartoon_batch)}")
            if data is None:
                data = np.array([np.concatenate((real_batch, cartoon_batch))])

            else:
                n_cartoon = len(cartoon_batch)
                n_real = len(real_batch)
                if n_cartoon != n_real:
                    # n = min(n_cartoon, n_real)
                    # print(n)
                    # print(f"{len(real_batch[:n])} - {len(cartoon_batch[:n])}")
                    # data = np.vstack((data, np.array([real_batch[:n], cartoon_batch[:n]])))
                    continue
                else:
                    data = np.vstack((data, [np.array(np.concatenate((real_batch, cartoon_batch)))]))
        return data