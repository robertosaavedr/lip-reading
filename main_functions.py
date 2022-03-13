from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import array_to_img
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from imutils import paths
import re

speakers = ['F01','F02','F04','F05','F06','F07','F08','F09','F10','F11','M01',
	'M02','M04','M07','M08']

data_types = ["phrases", "words"]

folders = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

img_path = list()

for speaker in speakers:
	for i in folders:
		for j in folders:	
			img_path = img_path + list(paths.list_images("lip_reading/" + "data/" + speaker + "/words/" + i + "/" +  j + "/"))

r = re.compile(".*color_.*")
img_path_color = list(filter(r.match, img_path))

data = list()
labels = list()
# loop over the image paths
for imagePath in img_path_color:
	# extract the class label from the filename
	label = imagePath.split(os.path.sep)[4]
	# load the input image (224x224) and preprocess it
	image = load_img(imagePath, target_size=(112, 112))
	image = img_to_array(image)
	image = preprocess_input(image)
	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)

array_to_img(image).show()
# convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)
words_labels = ("Begin", "Choose", "Connection", "Navigation", "Next", 
	"Previous", "Start", "Stop", "Hello", "Web")