"""
===============================
Infrared Thermal Images of Feet
===============================
"""

import os
from glob import glob
import shutil
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit


class InfraredThermalFeet():

    already_unzipped = False

    def __init__(self, percentage = 30, split=[0.2,0.2], seed: int=42,
                      dataset_path: str = "./gcpds/DataSet",
                        *args, **kwargs):

        self.split = listify(split)
        self.seed = seed
        self.percentage = percentage

        # Local dataset path
        self.__folder = dataset_path
        self.__path_images =  os.path.join(self.__folder, 'dataset')

        if not InfraredThermalFeet.already_unzipped:
            self.__set_env()
            InfraredThermalFeet.already_unzipped = True

        self.file_images = glob(os.path.join(self.__path_images, '*[!(mask)].jpg'))
        self.file_images = list(map(lambda x: x[:-4], self.file_images))
        self.file_images.sort()

        self.groups =  list(map(lambda x: os.path.split(x)[-1].split('_')[0],
                            self.file_images))

        self.num_samples = len(self.file_images)

    def __set_env(self):
        destination_path_zip = os.path.join(self.__folder,
                                            'InfraredThermalFeet.zip')
        unzip(destination_path_zip, self.__folder)

    @staticmethod
    def __preprocessing_mask(mask):
        mask = mask[...,0] > 0.5
        mask = mask.astype(np.float32)
        return mask[...,None]

    def load_instance_by_id(self, id_img):
        root_name = os.path.join(self.__path_images, id_img)
        return self.load_instance(root_name)

    @staticmethod
    def load_instance(root_name):
        img = cv2.imread(f'{root_name}.jpg')/255
        img = img[...,0][...,None]
        mask = cv2.imread(f'{root_name}_mask.png')
        mask = InfraredThermalFeet.__preprocessing_mask(mask)
        id_image = os.path.split(root_name)[-1]
        return img, mask, id_image


    @staticmethod
    def __gen_dataset(file_images):
        def generator():
            for root_name in file_images:
                yield InfraredThermalFeet.load_instance(root_name)
        return generator

    def __generate_tf_data(self, files):
        def generator():
            for root_name in files:
                img, mask, id_image = InfraredThermalFeet.load_instance(root_name)
                yield img, mask, id_image

        # Example shape based on a sample file
        sample_img, sample_mask, _ = InfraredThermalFeet.load_instance(files[0])
        img_shape = sample_img.shape
        mask_shape = sample_mask.shape

        output_signature = (
            tf.TensorSpec(shape=img_shape, dtype=tf.float32),
            tf.TensorSpec(shape=mask_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.string)
        )
        
        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        len_files = len(files)
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(len_files))
        return dataset


    def __get_log_tf_data(self,i,files):
        print(f' Number of images for Partition {i}: {len(files)}')
        return self.__generate_tf_data(files)

    @staticmethod
    def __shuffle(X,seed):
        np.random.seed(seed)
        np.random.shuffle(X)

    @staticmethod
    def _train_test_split(X, groups, random_state=42,test_size=0.2):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                            random_state=random_state)
        indxs_train, index_test = next(gss.split(X, groups=groups))

        InfraredThermalFeet.__shuffle(indxs_train, random_state)
        InfraredThermalFeet.__shuffle(index_test, random_state)

        return X[indxs_train], X[index_test], groups[indxs_train], groups[index_test]

    @staticmethod
    def gen_dots(mask):
      # Generate randomly placed dots from a mask
      # mask: base fully segmented mask
      # n: number of dots
      # rad: radius of the circles
      n = 20
      rad = 10
      h,w = mask.shape[:2]

      max_valueh = h - rad
      max_valuew = w - rad

      #np.random.seed(0)
      old_mask = np.squeeze(mask)
      new_mask = np.full((h,w),0, dtype=np.uint8)

      # Erode the object mask to create the inner margin
      kernel = np.ones((rad, rad), np.uint8)
      eroded_mask = cv2.erode(old_mask, kernel, iterations=1)

      # Dilate the object mask to create the outer margin
      dilated_mask = cv2.dilate(old_mask, kernel, iterations=1)

      # Create the combined margin by subtracting the eroded mask from the dilated mask
      margin_mask = cv2.subtract(dilated_mask, eroded_mask)

      # Extract object indices avoiding the margin
      object_indices = np.argwhere((old_mask > 0) & (margin_mask == 0))
      background_indices = np.argwhere((old_mask == 0)  & (margin_mask == 0))

      num_object_points = n // 4
      num_background_points = n - num_object_points

      # Ensure there are enough points
      if len(object_indices) < num_object_points:
          num_object_points = len(object_indices)
      if len(background_indices) < num_background_points:
          num_background_points = len(background_indices)

      # Generate points for the object
      object_points = object_indices[np.random.choice(len(object_indices), num_object_points, replace=False)]
      for point in object_points:
          y, x = point
          cv2.circle(new_mask, (x, y), radius=rad, color=(2), thickness=-1)

      background_points = background_indices[np.random.choice(len(background_indices), num_background_points, replace=False)]
      for point in background_points:
          y, x = point
          cv2.circle(new_mask, (x, y), radius=rad, color=(1), thickness=-1)

      return new_mask[...,None]

    def __call__(self):
                
        file_images = np.array(self.file_images)
        groups = np.array(self.groups)
        seed = 2

        train_imgs, test_imgs, g_train, _ = InfraredThermalFeet._train_test_split(
            file_images, groups, test_size=self.split[0], random_state=seed
        )

        train_imgs, val_imgs, _, _ = InfraredThermalFeet._train_test_split(
            train_imgs, g_train, test_size=self.split[1], random_state=seed
        )

        if self.percentage < 100:
            num_train = int(len(train_imgs) * self.percentage / 100)
            #num_val = int(len(val_imgs) * self.percentage / 100)
            #num_test = int(len(test_imgs) * self.percentage / 100)
            
            changed_train_imgs = train_imgs[num_train:]
            unchanged_train_imgs = train_imgs[:num_train]

            #val_imgs = val_imgs[:num_val]
            #test_imgs = test_imgs[:num_test]

            def transform_train_images(files):
                for file in files:
                    img, mask, id_image = InfraredThermalFeet.load_instance(file)
                    transformed_mask = InfraredThermalFeet.gen_dots(mask)
                    yield img, transformed_mask, id_image

            changed_train_dataset = tf.data.Dataset.from_generator(
                lambda: transform_train_images(changed_train_imgs),
                output_signature=(tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
                                tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
                                tf.TensorSpec(shape=(), dtype=tf.string))
            )

            def load_train_images(files):
                for file in files:
                    img, mask, id_image = InfraredThermalFeet.load_instance(file)
                    yield img, mask, id_image

            unchanged_train_dataset = tf.data.Dataset.from_generator(
                lambda: load_train_images(unchanged_train_imgs),
                output_signature=(tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
                                tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
                                tf.TensorSpec(shape=(), dtype=tf.string))
            )

            train_dataset = unchanged_train_dataset.concatenate(changed_train_dataset)
        else:
        # If percentage is 100, all images are unchanged
            def load_train_images(files):
                for file in files:
                    img, mask, id_image = InfraredThermalFeet.load_instance(file)
                    yield img, mask, id_image

            train_dataset = tf.data.Dataset.from_generator(
                lambda: load_train_images(train_imgs),
                output_signature=(tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
                                tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
                                tf.TensorSpec(shape=(), dtype=tf.string))
            )
        val_dataset = self.__generate_tf_data(val_imgs)
        test_dataset = self.__generate_tf_data(test_imgs)

        return train_dataset, val_dataset, test_dataset



if __name__ == "__main__":
    import convRFFds.data as data
    from utils import unzip
    from utils import listify
    kwargs_data_augmentation = dict(repeat=1,
                                    batch_size=2,
                                    shape=224,
                                    split=[0.4, 0.3]
                                    )
    dataset = InfraredThermalFeet
    train_dataset, val_dataset, test_dataset = data.get_data(
        dataset_class=dataset, data_augmentation=False, return_label_info=True, **kwargs_data_augmentation)
else:
    from .utils import unzip
    from .utils import listify
