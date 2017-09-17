import numpy as np
import os
from scipy.misc import imresize
from scipy.misc import imread
from sklearn.utils import resample
import pandas
from utils.config import get, print_if_verbose


class imagepreproc:
	directoryTrainImage = ''
	directoryTrainMask = ''
	directoryTrainNormal = ''
	data_stored = False
	data_sort = False
	images = np.zeros(0)
	masks = np.zeros(0)
	normals = np.zeros(0)
	train_images = np.zeros(0)
	train_masks = np.zeros(0)
	train_normals = np.zeros(0)
	val_images = np.zeros(0)
	val_masks = np.zeros(0)
	val_normals = np.zeros(0)
	test_images = np.zeros(0)
	test_masks = np.zeros(0)
	test_normals = np.zeros(0)

	def __init__(self):
		self.directoryTrainImage = get('DATA.ImageDirectoryTrain')
		self.directoryTrainMask = get('DATA.ImageDirectoryTrainMask')
		self.directoryTrainNormal = get('DATA.ImageDirectoryTrainOutput')
		self.data_stored = False
		self.data_sort = False

	def get_images_labels(self, matrix):
		image_row_len = len(np.fromstring(matrix[0, 1], dtype=int, sep=' '))
		image_dim = int(np.sqrt(image_row_len))
		labels = matrix[:, 0]
		images = []
		for i in range(matrix.shape[0]):
			image_row = np.fromstring(matrix[i, 1], dtype=int, sep=' ')
			images.append(np.reshape(image_row, (image_dim, image_dim)))

		images = np.array(images)
		return images, labels

	def read_files(self):
		first = 1
		print_if_verbose("reading images")
		counter = 0;
		for filename in os.listdir(self.directoryTrainImage):
			if filename.endswith(".png"):
				strFilename = self.directoryTrainImage + filename
				encodedImage = imread(strFilename)
				if(counter%200 == 0):
					print(counter)
				counter+=1

				if first == 1:
					first = 0
					self.images = encodedImage[np.newaxis,...]
				else:
					self.images = np.concatenate((self.images,encodedImage[np.newaxis,...]),axis=0)
				if(counter == 200):
					break
		first = 1
		counter = 0
		print_if_verbose("reading masks")
		for filename in os.listdir(self.directoryTrainMask):
			if filename.endswith(".png"):
				strFilename = self.directoryTrainMask + filename
				encodedImage = imread(strFilename)
				counter+=1
				if first == 1:
					first = 0
					self.masks = encodedImage[np.newaxis,...]
				else:
					self.masks = np.concatenate((self.masks,encodedImage[np.newaxis,...]),axis=0)
				if(counter == 200):
					break
		print("shape of masks:{}".format(np.shape(self.masks)))
		first = 1
		counter = 0
		print_if_verbose("reading normals")
		for filename in os.listdir(self.directoryTrainNormal):
			if filename.endswith(".png"):
				strFilename = self.directoryTrainNormal + filename
				encodedImage = imread(strFilename)
				if(counter%1000 == 0):
					print(counter)
				counter+=1
				if first == 1:
					first = 0
					self.normals = encodedImage[np.newaxis,...]
				else:
					self.normals = np.concatenate((self.normals,encodedImage[np.newaxis,...]),axis=0)
				if(counter == 200):
					break
		print("shape of normals:{}".format(np.shape(self.normals)))
		self.data_stored = True

	def sort_data(self):
		shuffle_idx = np.random.permutation(len(self.images))
		self.train_images = self.images[shuffle_idx[0:100]]
		self.train_masks = self.masks[shuffle_idx[0:100]]
		self.train_normals = self.normals[shuffle_idx[0:100]]
		self.val_images= self.images[shuffle_idx[100:125]]
		self.val_masks= self.masks[shuffle_idx[100:125]]
		self.val_normals= self.normals[shuffle_idx[100:125]]
		self.test_images = self.images[shuffle_idx[125:]]
		self.test_masks = self.masks[shuffle_idx[125:]]
		self.test_normals = self.normals[shuffle_idx[125:]]
		self.data_sort = True
		del self.images
		del self.masks
		del self.normals

	def resize(self, images, new_size=32):
		resized = []
		for i in range(images.shape[0]):
			resized_image = imresize(images[i],
									 size=(new_size, new_size),
									 interp='bicubic')
			resized.append(resized_image)
		return np.array(resized)

	def preprocessed_data(self, split, dim=32, one_hot=False, balance_classes=True):
		if not self.data_stored:
			self.read_files()
		if not self.data_sort:
			self.sort_data()
		if split == 'train':
			print_if_verbose('Loading train data...')
			images, masks,normals = self.train_images, self.train_masks, self.train_normals

			numBlank = []
			print(len(images[:,1,1,1]))
			if(one_hot == False):
				for i in range(0, len(images[:,1,1,1])):
					images[i,:,:,0] = np.multiply(masks[i,:,:],images[i,:,:,0])
					images[i,:,:,1] = np.multiply(masks[i,:,:],images[i,:,:,1])
					images[i,:,:,2] = np.multiply(masks[i,:,:],images[i,:,:,2])
					normals[i,:,:,0] = np.multiply(masks[i,:,:],normals[i,:,:,0])
					normals[i,:,:,1] = np.multiply(masks[i,:,:],normals[i,:,:,1])
					normals[i,:,:,2] = np.multiply(masks[i,:,:],normals[i,:,:,2])

		elif split == 'val':
			print_if_verbose('Loading validation data...')
			images, masks,normals =  self.val_images, self.val_masks,self.val_normals
			numBlank = []
			if(one_hot == False):
				for i in range(0, len(images[:,1,1,1])):
					images[i,:,:,0] = np.multiply(masks[i,:,:],images[i,:,:,0])
					images[i,:,:,1] = np.multiply(masks[i,:,:],images[i,:,:,1])
					images[i,:,:,2] = np.multiply(masks[i,:,:],images[i,:,:,2])
					normals[i,:,:,0] = np.multiply(masks[i,:,:],normals[i,:,:,0])
					normals[i,:,:,1] = np.multiply(masks[i,:,:],normals[i,:,:,1])
					normals[i,:,:,2] = np.multiply(masks[i,:,:],normals[i,:,:,2])
		elif split == 'test':
			print_if_verbose('Loading test data...')
			images, masks,normals =  self.test_images, self.test_masks,self.test_normals
			numBlank = []
			if(one_hot == False):
				for i in range(0, len(images[:,1,1,1])):
					images[i,:,:,0] = np.multiply(masks[i,:,:],images[i,:,:,0])
					images[i,:,:,1] = np.multiply(masks[i,:,:],images[i,:,:,1])
					images[i,:,:,2] = np.multiply(masks[i,:,:],images[i,:,:,2])
					normals[i,:,:,0] = np.multiply(masks[i,:,:],normals[i,:,:,0])
					normals[i,:,:,1] = np.multiply(masks[i,:,:],normals[i,:,:,1])
					normals[i,:,:,2] = np.multiply(masks[i,:,:],normals[i,:,:,2])
		else:
			print_if_verbose('Invalid input!')
			return
		images = np.expand_dims(images,axis=5)
		print(np.shape(images))

		print_if_verbose('---Images shape: {}'.format(images.shape))
		return images, masks, normals



if __name__ == '__main__':
	data = imagepreproc()
	images, masks, normals = data.preprocessed_data('train') 




