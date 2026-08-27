import cv2
from app.services.feature_extractor import ImgFeatureExtractor

path = "test_imgs/1.jpg"
img = cv2.imread(path)
extractor = ImgFeatureExtractor()
features = extractor.extract(img)

print("Extracted features:",features)