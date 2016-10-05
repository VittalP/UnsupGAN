import numpy as np
from tensorflow.examples.tutorials import mnist
import os
import numpy as np
from infogan.misc.utils import get_image

class Dataset(object):
    def __init__(self, data_root=None, list_file=None, batch_size=64, is_crop=True, is_grayscale=False, output_size=64,images=None, labels=None):

        self.batch_size = batch_size
        self.data_root = data_root
        self.is_crop = is_crop
        self.output_size=output_size
        self.is_grayscale = is_grayscale
        self._images = images
        self.list_file = list_file
        if self.is_grayscale:
            self.image_shape = (self.output_size, self.output_size, None)
        else:
            self.image_shape = (self.output_size, self.output_size, 3)

        if images is not None:
            self._images = images.reshape(images.shape[0], -1)
            self._labels = labels
            self._epochs_completed = -1
            self._num_examples = images.shape[0]
            # shuffle on first run
            self._index_in_epoch = self._num_examples

        if self.list_file:
            with open(self.list_file, 'r') as ff:
                    self.image_list = [path.strip().split(' ')[0] for path in ff.readlines() if 'txt' not in path]
            self.batch_idx = len(self.image_list) // self.batch_size
            self.counter = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        self.batch_size = batch_size
        if self.images:
            """Return the next `batch_size` examples from this data set."""
            start = self._index_in_epoch
            self._index_in_epoch += batch_size
            if self._index_in_epoch > self._num_examples:
                # Finished epoch
                self._epochs_completed += 1
                # Shuffle the data
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self._images[perm]
                if self._labels is not None:
                    self._labels = self._labels[perm]
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size
                assert batch_size <= self._num_examples
            end = self._index_in_epoch
            if self._labels is None:
                return self._images[start:end], None
            else:
                return self._images[start:end], self._labels[start:end]

        if self.list_file:
            idx = self.counter
            self.batch_files = self.image_list[idx*self.batch_size:(idx+1)*self.batch_size]
            #if self.labels:
            #    self.batch_labels = self.label[idx*self.batch_size:(idx+1)*self.batch_size]
            batch = [get_image(os.path.join(self.data_root, batch_file), is_crop=self.is_crop, resize_w=self.output_size) for batch_file in self.batch_files]
            self.counter = (self.counter+1) % self.batch_idx

            if (self.is_grayscale):
                batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
            else:
                batch_images = np.array(batch).astype(np.float32)
            return batch_images


class MnistDataset(object):
    def __init__(self):
        self.name = "mnist"
        data_directory = "MNIST"
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        dataset = mnist.input_data.read_data_sets(data_directory)
        self.train = dataset.train
        # make sure that each type of digits have exactly 10 samples
        sup_images = []
        sup_labels = []
        rnd_state = np.random.get_state()
        np.random.seed(0)
        for cat in range(10):
            ids = np.where(self.train.labels == cat)[0]
            np.random.shuffle(ids)
            sup_images.extend(self.train.images[ids[:10]])
            sup_labels.extend(self.train.labels[ids[:10]])
        np.random.set_state(rnd_state)
        self.supervised_train = Dataset(
            images=np.asarray(sup_images),
            labels=np.asarray(sup_labels),
        )
        self.test = dataset.test
        self.validation = dataset.validation
        self.image_dim = 28 * 28
        self.image_shape = (28, 28, 1)

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data

class ImageNetDatset(Dataset):
    def __init__(self, batch_size=64, output_size=64):
        self.name = "imagenet"
        self.data_root = '/mnt/disk1/vittal/data/ILSVRC2015/Data/CLS-LOC/train/'
        self.is_crop = True
        self.output_size=output_size
        self.is_grayscale=False
        Dataset.__init__(self, data_root=self.data_root, list_file='./data/imagenet/train_shuffle.txt', batch_size=batch_size, is_crop=self.is_crop, is_grayscale=self.is_grayscale, output_size=self.output_size)

class celebADataset(Dataset):
    def __init__(self, batch_size=64, output_size=64):
        self.name = "celebA"
        self.data_root = './data/celebA'
        self.is_crop = True
        self.output_size=output_size
        self.is_grayscale=False

        Dataset.__init__(self, data_root=self.data_root, list_file='./data/celebA/train_shuffle.txt', batch_size=batch_size, is_crop=self.is_crop, is_grayscale=self.is_grayscale, output_size=self.output_size)
