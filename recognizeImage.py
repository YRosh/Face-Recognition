import face_recognition
import pickle
import cv2
import sys

imgPath = sys.argv[1]
encodingsPath = sys.argv[2]

print("Loading encodings...")
fd = open(encodingsPath, "rb")
data = pickle.loads(fd.read())

image = cv2.imread(imgPath)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

boxes = face_recognition.face_locations(rgb, model="cnn")
encodings = face_recognition.face_encodings(rgb, boxes)
names = []

print("Finding Sachin...")

for encoding in encodings:
	matches = face_recognition.compare_faces(data["encodings"], encoding)

	count = matches.count(True)
	print(count/len(matches))
	if (count/len(matches) >= 0.8):
		names.append("Sachin")
	else :
		names.append("notSachin")

	# if True in matches:
	# 	matchedIndex = [i for (i, match) in enumerate(matches) if match]

	# 	count = 0

	# 	for index in matchedIndex:
	# 		if data["names"][index] == "Sachin":
	# 			count = count + 1

	# 	ratio = count/len(matchedIndex)

	# 	if count >= 80 and ratio > 0.75:
	# 		names.append("Sachin")
	# 	else:
	# 		names.append("Not Sachin")

if "Sachin" in names:
	for ((top, right, bottom, left), name) in zip(boxes, names):
		if name == "Sachin":
			cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
			y = top - 15 if top - 15 > 15 else top + 15
			cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

	print("Sachin found.")
	cv2.imshow("Sachin Found", image)
	cv2.waitKey(0)
else:
	print("Sachin not found.")
