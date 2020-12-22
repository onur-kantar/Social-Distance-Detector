from .social_distancing_config import NMS_THRESH
from .social_distancing_config import MIN_CONF
from .social_distancing_config import MIN_DISTANCE
from scipy.spatial import distance as dist
import numpy as np
import cv2


# frame: Video dosyanızdan veya doğrudan web kameranızdan çerçeve
# net: Önceden başlatılmış ve önceden eğitilmiş YOLO nesne algılama modeli
# ln: YOLO CNN çıktı katmanı adları
# personIdx: YOLO modeli birçok nesne türünü algılayabilir;
# Bu indeks, diğer nesneleri dikkate almayacağımız için özellikle insan sınıfı içindir
def detect_people(frame, net, ln, personIdx=0):

    # Çerçevenin boyutlarını alın ve sonuçların listesini başlatın
    (H, W) = frame.shape[:2]

    # results insan tahmin olasılığından, algılama için sınırlayıcı kutu koordinatlarından ve nesnenin ağırlık merkezinden oluşur.
    results = []

    # blobFromImage: İsteğe bağlı olarak görüntüyü merkezden yeniden boyutlandırır ve kırpar,
    # ortalama değerleri çıkarır,
    # değerleri ölçeklendiriciyle ölçeklendirir,
    # Mavi ve Kırmızı kanalları değiştirir.
    # Kısaca giriş çerçevesi için ön işlem
    network_input_image_size = 416
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (network_input_image_size, network_input_image_size),
                                 swapRB=True, crop=False)

    # YOLO modeline görüntü gönderilir
    # ve çıkarım yapılır
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # Tespit edilen sınırlayıcı kutular, ağırlık merkezleri ve doğruluk listesi oluşturulur
    boxes = []
    centroids = []
    confidences = []

    # Katman çıktılarının her biri üzerinde döngü
    for output in layerOutputs:
        # Algılamaların her birinin üzerinde döngü
        for detection in output:
            # Mevcut nesne algılamanın sınıf kimliğini ve güvenirliğini (yani olasılık) ayıklayın.
            # İlk 4 öğe, center_x, center_y, genişlik ve yüksekliği temsil eder.
            # Beşinci öğe, sınırlayıcı kutunun bir nesneyi çevrelediği konusundaki doğruluğu temsil eder.
            # Öğelerin geri kalanı, her bir sınıfla (yani nesne türü) ilişkili doğruluktur.
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Tespit edilen nesnenin bir insan olduğu ve asgari doğruluk eşiğinin karşılandığı kontrol edilir
            if classID == personIdx and confidence > MIN_CONF:

                # Sınırlayıcı kutu koordinatlarını görüntünün boyutuna göre yeniden ölçeklendirin,
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Kutunun sol üst koordinatlarını üretir
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Sınırlayıcı kutu koordinatları, merkezler ve doğruluk listesi güncellenir
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    # Zayıf, örtüşen sınırlayıcı kutuları bastırmak için maksimum olmayan bastırma uygulanır
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

    # En az bir algılamanın mevcut olduğundan emin olunur
    if len(idxs) > 0:
        # Tuttuğumuz dizinler üzerinde döngü
        for i in idxs.flatten():
            # sınırlayıcı kutu koordinatları ayıklanır
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # results listemizi kişi tahmin olasılığı,
            # sınırlayıcı kutu koordinatları ve
            # ağırlık merkezini içerecek şekilde güncelleyin
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    # return the list of results
    return results
def draw(frame, results):
    # İhlal kümesi oluşturulur
    violate = set()
    # En az 2 kişi tesbit edildiğinden emin olunur.
    if len(results) >= 2:
        # İnsanlar arasındaki mesafe Öklid yöntemiyle hesaplanır.
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        # Oluşan mesafe matriksinin ayrıştırılması
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                # Ölçülen mesafenin eşik değerinden küçük olup olmadığı kontrolü
                # D[i,j]: i. insan ile j. insan arasındaki mesafe
                if D[i, j] < MIN_DISTANCE:
                    # Mesafe ihlali yapanların matriksdeki konumunun kümeye eklenmesi
                    violate.add(i)
                    violate.add(j)

    # Sonuçlar üzerinde döngü
    # i: listenin indeksi
    # prob: doğruluk oranı
    # bbox: sınırlayıcı kutunun konumu
    # centroid: ağırlık merkezi
    for (i, (prob, bbox, centroid)) in enumerate(results):
        # Sınırlayıcı kutuyu ve ağırlık merkezi koordinatlarını çıkar.
        # color değişkenine yeşili ata
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        # index ihlal kümesinde mevcutsa rengi kırmızı yap
        if i in violate:
            color = (0, 0, 255)

        # Kişinin etrafına sınırlayıcı kutu ve merkez koordinatlarını çiz
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)

    # Toplam sosyal mesafe ihlalleri yazdırılır
    text = "Sosyal Mesafe Ihlalleri: {}".format(len(violate))
    cv2.putText(frame, text, (10, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

    # Çıktı ekranda gösterilir
    cv2.imshow("Sosyal Mesafe Algilayici", frame)
def saveVideo(frame, writer):
    # Çıktıyı .avi uzantılı olarak kaydetmek için kullanılır.
    if 'output.avi' != "" and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter('output.avi', fourcc, 30, (frame.shape[1], frame.shape[0]), True)
    if writer is not None:
        writer.write(frame)
    return writer
