
# import the necessary packages
import face_recognition
import argparse
import pickle
import cv2
import numpy as np

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open("D:\\Ankit\\Codes\\ankit_face\\face_features.pickle", "rb").read())  # give path to face_features.pickle" file

# load the input image and convert it from BGR to RGB
img1 = cv2.imread("D:\\Ankit\\Codes\\ankit_face\\examples\\ankit\\frame829.jpg")
#a = np.double(img1)
#b = a + 37
img2 = np.uint8(b)      # give path to image to be tested/recognised file
rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# detect the (x, y)-coordinates of the bounding boxes corresponding
# to each face in the input image, then compute the facial embeddings
# for each face
print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb,
	model="hog")
face_features = face_recognition.face_encodings(rgb, boxes)

# initialize the list of names for each face detected
names = []

# loop over the facial embeddings
for face_feature in face_features:
	# attempt to match each face in the input image to our known
	# encodings
	matches = face_recognition.compare_faces(data["face_features"],
		encoding, tolerance = 0.5)
	name = "Unknown"
	print(matches)
	# check to see if we have found a match
	if True in matches:
		# find the indexes of all matched faces then initialize a
		# dictionary to count the total number of times each face
		# was matched
		matchedIdxs = [i for (i, b) in enumerate(matches) if b]
		counts = {}

		# loop over the matched indexes and maintain a count for
		# each recognized face face
		for i in matchedIdxs:
			name = data["names"][i]
			counts[name] = counts.get(name, 0) + 1
			print(counts)

		# determine the recognized face with the largest number of
		# votes (note: in the event of an unlikely tie Python will
		# select first entry in the dictionary)
		name = max(counts, key=counts.get)
	
	# update the list of names
	names.append(name)

# loop over the recognized faces
for ((top, right, bottom, left), name) in zip(boxes, names):
	# draw the predicted face name on the image
	cv2.rectangle(img2, (left, top), (right, bottom), (0, 255, 0), 2)
	y = top - 15 if top - 15 > 15 else top + 15
	cv2.putText(img2, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.75, (0, 255, 0), 2)

# show the output image
cv2.imshow("Image", img2)
cv2.imwrite("D:\\Ankit\\Codes\\ankit_face\\output\\rec_int7.jpg" , img2)
cv2.waitKey(0)