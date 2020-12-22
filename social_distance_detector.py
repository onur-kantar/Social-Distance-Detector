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

# Görüntü giriş sorusu için cevaplar
class Input(Enum):
	file = '0'
	youtube = '1'
	camera = '2'
	screen = '3'

# Görüntü kayıt sorusu için cevaplar
class Save(Enum):
	no = '0'
	yes = '1'

# YOLO modelimizin eğitim aldığı COCO sınıfı etiketlerini yükleyin
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# YOLO ağırlıklarına ve model konfigürasyonuna giden yolları belirle
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

print(weightsPath)
# COCO veri seti (80 sınıf) üzerine eğitilmiş YOLO nesne dedektörünü yükle
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# GPU kullanma kontrolü
if config.USE_GPU:
	# GPU ile çalıştır
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
else:
	# CPU ile çalıştır
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# YOLO'da ihtiyacımız olan çıktı katman adlarını belirle
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print(f"Layer Names: {ln}")

# Video giriş türü için kullanıcıdan giriş alınır
print('Hangi Türde İşlem Yapacağınızı Seçin:')
print(Input.file.value + ': Dosya Yükle')
print(Input.youtube.value + ': Youtube URL Yükle')
print(Input.camera.value + ': Kamera Kullan')
print(Input.screen.value + ': Ekranı Kullan')
myinput = input()

# Çıkış görüntüsünün kaydedilip kaydedilmemesi için kullanıcıdan cevap alınır
print('Kaydetmek İstiyor Musun?:')
print(Save.no.value + ': Hayır')
print(Save.yes.value + ': Evet')
save = input()

# Kaydedilecek görüntüler için writer oluşturulur
writer = None

# Video dosyası ile görüntü girişi
if myinput == Input.file.value:
	# Dosya yolu belirlenip görüntü alınır
	myinput = input('Dosya Yolu Uzantısını Giriniz: ')
	vs = cv2.VideoCapture(myinput)
	while True:
		start = time.time()

		# Görüntüden frame alınır
		(grabbed, frame) = vs.read()
		if not grabbed:
			break

		# frame'in boyutu değiştirilir
		# detect_people fonksiyonu çalıştırılır
		# ardından draw fonksiyonu çalıştırılır
		frame = imutils.resize(frame, width=700)
		results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
		draw(frame, results)

		# Çıktı kayıt işlemi
		if save == Save.yes.value:
			writer = saveVideo(frame, writer)

		# Q tuşu çalışmayı durdurur
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

		end = time.time()
		# FPS Ölçümü
		print(1 / (end - start))
# Youtube ile görüntü girişi
elif myinput == Input.youtube.value:
	# Youtube URL belirlenip görüntü alınır
    myinput = input("Youtube URL'ini Giriniz: ")
    stream = CamGear(source = myinput, y_tube=True, logging=True).start()
    while True:
        start = time.time()

		# Görüntüden frame alınır
        frame = stream.read()
        if frame is None:
            break

		# frame'in boyutu değiştirilir
		# detect_people fonksiyonu çalıştırılır
		# ardından draw fonksiyonu çalıştırılır
        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
        draw(frame, results)

		# Çıktı kayıt işlemi
        if save == Save.yes.value:
            writer = saveVideo(frame, writer)

		# Q tuşu çalışmayı durdurur
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        end = time.time()
		# FPS Ölçümü
        print(1 / (end - start))
    stream.stop()
# Kamera ile görüntü girişi
elif myinput == Input.camera.value:
	# Varsayılan kameradan görüntü alınır
	vs = cv2.VideoCapture(0)
	while True:
		start = time.time()

		# Görüntüden frame alınır
		(grabbed, frame) = vs.read()
		if not grabbed:
			break

		# frame'in boyutu değiştirilir
		# detect_people fonksiyonu çalıştırılır
		# ardından draw fonksiyonu çalıştırılır
		frame = imutils.resize(frame, width=700)
		results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
		draw(frame, results)

		# Çıktı kayıt işlemi
		if save == Save.yes.value:
			writer = saveVideo(frame, writer)

		# Q tuşu çalışmayı durdurur
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

		end = time.time()
		# FPS Ölçümü
		print(1 / (end - start))
# Ekran görüntüsü ile görüntü girişi
elif myinput == Input.screen.value:
	# Ekrandan alınacak görüntünün konumu ve büyüklüğü
	monitor = {"top": 160, "left": 160, "width": 700, "height": 700}
	with mss() as sct:
		while True:
			start = time.time()

			# Ekrandan görüntü alınır
			# Görüntü diziye dönüştürülür
			# RGBA renk uzayında olan görüntü RGB'ye dönüştürülür
			# detect_people fonksiyonu çalıştırılır
			# ardından draw fonksiyonu çalıştırılır
			image = sct.grab(monitor)
			image_np = np.array(image)
			frame = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
			results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
			draw(frame, results)

			# Çıktı kayıt işlemi
			if save == Save.yes.value:
				writer = saveVideo(frame, writer)

			# Q tuşu çalışmayı durdurur
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

			end = time.time()
			# FPS Ölçümü
			print(1 / (end - start))
else:
	print("Yanlış Değer Girdiniz")
cv2.destroyAllWindows()