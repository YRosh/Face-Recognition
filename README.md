# Face-Recognition  
Can be used to trim any video with the frames of specific people, using [face_recognition](https://pypi.org/project/face-recognition/) (a state of the art deep learning trained model) and **OpenCV**.

#### Steps  
* Prepare the dataset with root folder contain sub-folders which contain images of each person. Around 10-20 are sufficient.  
* Run _encodeImages.py_ with arguments: dataset root path and path where to save the trained model.  
```
python3 encodeImages.py "dataset path" "model path"
```  
* Then run the _recognizeVideo.py_ with arguments: input video path, encoding file path and path where to save processed video.
```
python3 recognizeVideo.py "input video path" "saved model path" "output path"
```
