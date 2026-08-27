import cv2
import numpy as np

class ImgFeatureExtractor:
    # to extract cv features from img
    # these will alter be used by ml model for img quality classification

    def extract(self,img:np.ndarray)->dict:
        # np.ndarray: img loaded using opencv (bgr format)
        # returns dict of features

        if img is None:
            raise ValueError("Invalid or unreadable img")

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        features = {
            "sharpness":self._calc_sharpness(gray),
            "brightness":self._calc_brightness(gray),   
            "contrast":self._calc_contrast(gray),
            "noise":self._calc_noise(gray),
            "entropy":self._calc_entropy(gray),
            "saturation":self._calc_sat(img),
            "darkpixels":self._calc_darkpixels(gray),
            "brightpixels":self._calc_brightpixels(gray),
        }

        return features

    def _calc_sharpness(self,gray:np.ndarray)->float:
        # higher value indicates sharper img
        # lower value indicates blur.

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def _calc_brightness(self,gray:np.ndarray)->float:
        # higher value indicates brighter img
        # lower value indicates darker img

        return float(np.mean(gray))

    def _calc_contrast(self,gray:np.ndarray)->float:
        # higher value indicates higher contrast
        # lower value indicates lower contrast

        return float(np.std(gray))

    def _calc_noise(self,gray:np.ndarray)->float:
        blur = cv2.GaussianBlur(gray,(5,5),0)
        residual = gray.astype(np.float32) - blur.astype(np.float32)
        noise = np.std(residual)

        return float(noise)

    def _calc_entropy(self,gray:np.ndarray)->float:
        # shannon entropy
        # higher entropy means greater information and texture variation.

        hist = cv2.calcHist([gray],[0],None,[256],[0,256]).flatten()

        probs = hist / hist.sum()
        probs = probs[probs > 0]  # filter out zero probabilities

        entropy = -np.sum(probs * np.log2(probs))

        return float(entropy) 

    def _calc_sat(self,img:np.ndarray)->float:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat = hsv[:,:,1]
        return float(np.mean(sat))

    def _calc_darkpixels(self,gray:np.ndarray)->float:
        darks = np.sum(gray < 30) 
        return float(darks / gray.size)

    def _calc_brightpixels(self,gray:np.ndarray)->float:
        brights = np.sum(gray > 225) 
        return float(brights / gray.size)