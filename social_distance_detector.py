# USAGE
# python social_distance_detector.py --input pedestrians.mp4
# python social_distance_detector.py --input pedestrians.mp4 --output output.avi

# import the necessary packages
from PIL import Image
from mss import mss
from pyimagesearch import social_distancing_config as config
from pyimagesearch.detection import detect_people, draw, saveVideo
import numpy as np
import argparse
import imutils
import cv2
import os
from enum import Enum
import time
from vidgear.gears import CamGear

class Option(Enum):
	file = '0'
	youtube = '1'
	camera = '2'
	screen = '3'

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if config.USE_GPU:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
else:
	# these are the defaults but its nice to see them for documentation
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print(f"Layer Names: {ln}")


print('Hangi Türde İşlem Yapacağınızı Seçin:')
print(Option.file.value + ': Dosya Yükle')
print(Option.youtube.value + ': Youtube URL Yükle')
print(Option.camera.value + ': Kamera Kullan')
print(Option.screen.value + ': Ekranı Kullan')

myinput = input()

writer = None

if myinput == Option.file.value:
	#dosya yolu
	myinput = input('Dosya Yolu Uzantısını Giriniz: ')
	vs = cv2.VideoCapture(myinput)
	while True:
		start = time.time()
		frame = vs.read()

		if frame is None:
			break

		frame = imutils.resize(frame, width=700)
		results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
		draw(frame, results)
		writer = saveVideo(frame, writer)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

		end = time.time()
		print(1 / (end - start))
elif myinput == Option.youtube.value:
    myinput = input("Youtube URL'ini Giriniz: ")
    stream = CamGear(source = myinput, y_tube=True, logging=True).start()
    while True:
        start = time.time()

        frame = stream.read()

        if frame is None:
            break

        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
        draw(frame, results)
        writer = saveVideo(frame, writer)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        end = time.time()
        print(1 / (end - start))
    stream.stop()
elif myinput == Option.camera.value:
	vs = cv2.VideoCapture(0)
	while True:
		start = time.time()

		(grabbed, frame) = vs.read()
		if not grabbed:
			break
		frame = imutils.resize(frame, width=700)
		results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
		draw(frame, results)
		writer = saveVideo(frame, writer)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

		end = time.time()
		print(1 / (end - start))
elif myinput == Option.screen.value:
	monitor = {"top": 160, "left": 160, "width": 700, "height": 700}
	with mss() as sct:
		while True:
			start = time.time()

			image = sct.grab(monitor)
			image_np = np.array(image)
			frame = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
			results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
			draw(frame, results)
			writer = saveVideo(frame, writer)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

			end = time.time()
			print(1 / (end - start))
else:
	print("Yanlış Değer Giriniz")
cv2.destroyAllWindows()


