import face_recognition 
import pickle
import cv2
import os
import sys
from PIL import Image

def isImage(path):
	try:
		Image.open(path)
	except IOError:
		return False
	return True


def getImages(path, imagePaths=[]):
	for f in os.listdir(path):
		if isImage(os.path.join(path, f)):
			imagePaths.append(os.path.join(path, f))
		elif os.path.isdir(os.path.join(path, f)):
			getImages(os.path.join(path, f), imagePaths)


dataSetPath = sys.argv[1]
encodeingFilePath = sys.argv[2]
detectionAlgo = sys.argv[3]

print("Encoding images...")
print()
imagePaths = []
getImages(dataSetPath, imagePaths)

encodings = []
# names = []

for (i, path) in enumerate(imagePaths):
	print("Processing image {}/{}".format(i+1, len(imagePaths)))
	# name = path.split(os.path.sep)[-2]

	image = cv2.imread(path)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	boxes = face_recognition.face_locations(rgb, model=detectionAlgo)
	encoding = face_recognition.face_encodings(rgb, boxes)

	for end in encoding:
		encodings.append(end)
		#names.append(name)

print()
print("Processing complete\nSerializing and dumping...")
# data = {"encodings":encodings, "names":names}
data = {"encodings":encodings}
fd = open(encodeingFilePath, "wb")
fd.write(pickle.dumps(data))
fd.close
print("Encoding Complete.")