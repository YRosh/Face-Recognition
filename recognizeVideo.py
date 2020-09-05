import face_recognition
import cv2
import pickle
import sys
import imutils
import threading
from math import floor

def mergeFrames(frames = []):
	i = 0
	while(i < len(frames) - 1):
		if frames[i+1][0] - frames[i][1] <= 15:
			ind = []
			ind.append(frames[i][0])
			ind.append(frames[i+1][1])
			frames.pop(i+1)
			frames.pop(i)
			frames.insert(i, ind)
		else:
			i = i + 1
	return frames

videoPath = sys.argv[1]
encodingsPath = sys.argv[2]
outputPath = sys.argv[3]

print("Loading encodings...")
fd = open(encodingsPath, "rb")
data = pickle.loads(fd.read())

print("Processing video...")
stream = cv2.VideoCapture(videoPath)
#writer = None
total = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames = {}".format(total))
c = 0

frameInd = []
flag = True

while True:
	(success, frame) = stream.read()
	if not success:
		break
	c = c + 1

	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(frame, width=750)
	r = frame.shape[1] / float(rgb.shape[1])

	boxes = face_recognition.face_locations(rgb, model="cnn")
	encodings = face_recognition.face_encodings(rgb, boxes)
	# names = []
	sachin = False
	for encoding in encodings:
		matches = face_recognition.compare_faces(data["encodings"], encoding)

		count = matches.count(True)

		if (count/len(matches) >= 0.7):
			sachin = True
			break

	if sachin:
		if flag:
			Range = []
			Range.append(int(stream.get(cv2.CAP_PROP_POS_FRAMES)))
			flag = False
	else:
		if not flag:
			Range.append(int(stream.get(cv2.CAP_PROP_POS_FRAMES))-1)
			frameInd.append(Range)
			flag = True

	print("\rProgress : {0} {1}%".format(('#'*(floor((c/(2*total))*100)))+(' '*(50-floor((c/(2*total)*100)))), round((c/total)*100, 5)), end='')

print("Video Processed.\nSaving new video...")

updatedFrames = mergeFrames(frameInd)

writer = None

for i in range(len(updatedFrames)):
	stream.set(cv2.CAP_PROP_POS_FRAMES, updatedFrames[i][0])
	while True:
		(success, frame) = stream.read()
		if not success:
			break

		if writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(outputPath, fourcc, 24, (frame.shape[1], frame.shape[0]), True)


		if writer is not None:
			if int(stream.get(cv2.CAP_PROP_POS_FRAMES)) <= updatedFrames[i][1]:
				writer.write(frame)
			else:
				break

stream.release()
if writer is not None:
	writer.release()

print("\nVideo Processed, and saved.")
