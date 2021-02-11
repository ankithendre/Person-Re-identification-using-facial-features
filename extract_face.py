
# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
from timeit import default_timer as timer

start = timer()
dataset= "D:\\Ankit\\Codes\\ankit_face\\dataset" #image of person to extract facial features
         
data = pickle.loads(open("D:\\Ankit\\Codes\\ankit_face\\face_features.pickle", "rb").read())

# grab the paths to the input images in our dataset
imagePaths = list(paths.list_images(dataset))
print("[INFO] Finding faces...")

# initialize the list of known encodings and known names
knownFeatures = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# load the input image and convert it from RGB (OpenCV ordering)
	
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(rgb,
		model="cnn")

	# compute the facial features for the face
	face_features = face_recognition.face_encodings(rgb, boxes)

	# loop over the facial features
	for face_feature in face_features:
		# add each encoding + name to our set of known names and
		# encodings
		knownFeatures.append(face_feature)
		knownNames.append(name)

# dump the facial encodings + names to disk
print("[INFO] serializing facial features...")
data = {"encodings": knownFeatures, "names": knownNames}
f = open("data", "wb")
f.write(pickle.dumps(data))
end = timer()
print("Elapsed Time for CNN in Seconds ", end- start)
f.close()