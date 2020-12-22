# YOLO dizininin yolu
MODEL_PATH = "yolo-coco"

# Minimum Güven:
# Olasılık tahmini 0.6'dan daha düşük olan tüm kutuları çıkardıktan sonra
# Maksimum olmayan bastırma tekniği:
# nesne için yinelenen ve örtüşen öneri kutuları içinde en uygun temsilleri
# seçerek örtüşmesi düşük olan kutuları kaldırmayı amaçlar
MIN_CONF = 0.3
NMS_THRESH = 0.3

# NVIDIA CUDA GPU'nun kullanılması gerekip gerekmediğini belirten değer
USE_GPU = True

# Piksel cinsinden iki kişinin arasındaki minimum güvenli mesafe
MIN_DISTANCE = 50