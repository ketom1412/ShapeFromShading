import os
import numpy as np
import scipy.misc
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from data_scripts.imagePreProcNew import imagepreproc
from utils.config import get

class DataSet(object):
	def __init__(self,
				 images,
				 masks,
				 normals,
				 fake_data=False,
				 dtype=dtypes.float32,
				 reshape=True):
		"""Construct a DataSet.
        """
		dtype = dtypes.as_dtype(dtype).base_dtype
		if dtype not in (dtypes.uint8, dtypes.float32):
			raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
							dtype)
		else:
			assert images.shape[0] == normals.shape[0], (
				'images.shape: %s normals.shape: %s' % (images.shape, normals.shape))
			self._num_examples = images.shape[0]
		self._images = images
		self._masks = masks
		self._normals = normals
		self._epochs_completed = 0
		self._index_in_epoch = 0

	@property
	def images(self):
		return self._images

	@property
	def masks(self):
		return self._masks

	@property
	def normals(self):
		return self._normals

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed

	def getImages(self,numbers,type):
		im_array = np.zeros(0)
		first=1
		if(type == "images"):
			for i in numbers:
				strFilename = get('DATA.ImageDirectoryTrain')+ str(int(i)) + ".png"
				encodedImage = scipy.misc.imread(strFilename)
				if first == 1:
					first = 0
					im_array = encodedImage[np.newaxis,...]
				else:
					im_array = np.concatenate((im_array,encodedImage[np.newaxis,...]),axis=0)
		elif(type == "masks"):
			for i in numbers:
				strFilename = get('DATA.ImageDirectoryTrainMask') + str(int(i)) + ".png"
				encodedImage = scipy.misc.imread(strFilename)
				if first == 1:
					first = 0
					im_array = encodedImage[np.newaxis,...]
				else:
					im_array = np.concatenate((im_array,encodedImage[np.newaxis,...]),axis=0)
		elif(type == "normals"):
			for i in numbers:
				strFilename = get('DATA.ImageDirectoryTrainOutput') + str(int(i)) + ".png"
				encodedImage = scipy.misc.imread(strFilename)
				if first == 1:
					first = 0
					im_array = encodedImage[np.newaxis,...]
				else:
					im_array = np.concatenate((im_array,encodedImage[np.newaxis,...]),axis=0)
		return im_array

	def getValidation(self,numbers,type):
		im_array = np.zeros(0)
		first=1
		if(type == "images"):
			for num in numbers:
				strFilename = get('DATA.ImageDirectoryTrain')+ str(int(num)) + ".png"
				encodedImage = scipy.misc.imread(strFilename)
				if first == 1:
					first = 0
					im_array = encodedImage[np.newaxis,...]
				else:
					im_array = np.concatenate((im_array,encodedImage[np.newaxis,...]),axis=0)
		elif(type == "masks"):
			for num in numbers:
				strFilename = get('DATA.ImageDirectoryTrainMask') + str(int(num)) + ".png"
				encodedImage = scipy.misc.imread(strFilename)
				if first == 1:
					first = 0
					im_array = encodedImage[np.newaxis,...]
				else:
					im_array = np.concatenate((im_array,encodedImage[np.newaxis,...]),axis=0)
		elif(type == "normals"):
			for num in numbers:
				strFilename = get('DATA.ImageDirectoryTrainOutput') + str(int(num)) + ".png"
				encodedImage = scipy.misc.imread(strFilename)
				if first == 1:
					first = 0
					im_array = encodedImage[np.newaxis,...]
				else:
					im_array = np.concatenate((im_array,encodedImage[np.newaxis,...]),axis=0)
		return im_array

	def next_batch(self, batch_size, fake_data=False, shuffle=True):
		"""Return the next `batch_size` examples from this data set."""
		start = self._index_in_epoch
		if self._epochs_completed == 0 and start == 0 and shuffle:
			perm0 = np.arange(self._num_examples)
			np.random.shuffle(perm0)
			self._images = self.images[perm0]
			self._normals = self.normals[perm0]

		if start + batch_size > self._num_examples:
			self._epochs_completed += 1
			rest_num_examples = self._num_examples - start
			images_rest_part = self._images[start:self._num_examples]
			masks_rest_part = self._masks[start:self._num_examples]
			normals_rest_part = self._normals[start:self._num_examples]

			if shuffle:
				perm = np.arange(self._num_examples)
				np.random.shuffle(perm)
				self._images = self.images[perm]
				self._normals = self.normals[perm]

			start = 0
			self._index_in_epoch = batch_size - rest_num_examples
			end = self._index_in_epoch
			images_new_part = self._images[start:end]
			masks_new_part = self._masks[start:end]
			normals_new_part = self._normals[start:end]
			return (self.getImages(np.concatenate((images_rest_part, images_new_part), axis=0),"images")/255.0-0.5)*2,\
				   self.getImages(np.concatenate((masks_rest_part, masks_new_part), axis=0),"masks"),\
				   (self.getImages(np.concatenate((normals_rest_part, normals_new_part), axis=0),"normals")/255.0-0.5)*2
		else:
			self._index_in_epoch += batch_size
			end = self._index_in_epoch
			return (self.getImages(self._images[start:end],"images")/255.0-0.5)*2,\
				   self.getImages(self._masks[start:end],"masks"), \
				   (self.getImages(self._normals[start:end],"normals")/255.0-0.5)*2

def read_data_set(data, split_name, one_hot, balance_classes):
	images, normals = data.preprocessed_data(split_name,
											 one_hot=one_hot,
											 balance_classes=balance_classes)
	return images, normals

def read_data_sets(one_hot=True,
				   balance_classes=True,
				   dtype=dtypes.float32,
				   reshape=False):
	data = imagepreproc()
	directoryTrainImage = get('DATA.ImageDirectoryTrain')
	directoryTrainMask = get('DATA.ImageDirectoryTrainMask')
	directoryTrainNormal = get('DATA.ImageDirectoryTrainOutput')
	tr_images_train = (np.linspace(0,16999,17000))
	tr_images_valid = (np.linspace(17000,18499,1500))
	tr_images_test = (np.linspace(18500,19999,1500))
	shuffle_idx_train = np.random.permutation(len(tr_images_train))
	shuffle_idx_valid = np.random.permutation(len(tr_images_valid))
	shuffle_idx_test = np.random.permutation(len(tr_images_test))
	train_images = tr_images_train[shuffle_idx_train[0:]]
	train_masks = tr_images_train[shuffle_idx_train[0:]]
	train_normals = tr_images_train[shuffle_idx_train[0:]]
	val_images= tr_images_valid[shuffle_idx_valid[0:]]
	val_masks= tr_images_valid[shuffle_idx_valid[0:]]
	val_normals= tr_images_valid[shuffle_idx_valid[0:]]
	test_images = shuffle_idx_test[shuffle_idx_test[0:]]
	test_masks = shuffle_idx_test[shuffle_idx_test[0:]]
	test_normals = shuffle_idx_test[shuffle_idx_test[0:]]

	train = DataSet(train_images,
					train_masks,
					train_normals,
					dtype=dtype,
					reshape=reshape)
	validation = DataSet(val_images,
						 val_masks,
						 val_normals,
						 dtype=dtype,
						 reshape=reshape)
	test = DataSet(test_images,
				   test_masks,
				   test_normals,
				   dtype=dtype,
				   reshape=reshape)
	return base.Datasets(train=train, validation=validation, test=test)



	