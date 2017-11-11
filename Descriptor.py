import numpy as np
import cv2

class Descriptor:
    def __init__(self, bins):
        self.bins = bins
    def describe(self, image):
        #convert image from BGR to HSV color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []
        #h,w height,width of the image
        (h, w) = image.shape[:2]
        #find the center, later we devide the image into 5 smaller parts
        (cX, cY) = (int(w * 0.5), int(h * 0.5))

        #four rectangles
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]
        #an ellip locates in the center of the image
        #create a mask
        (axesX, axesY) = (int(w * 0.75) / 2, int(h * 0.75) / 2) #axes of the ellip
        ellipMask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
        # loop over the segments
        for (startX, endX, startY, endY) in segments:
            # construct a mask for each corner of the image, subtracting
            # the elliptical center from it
            cornerMask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)

            # extract a color histogram from the image, then update the
            # feature vector
            hist = self.histogram(image, cornerMask)
            features.extend(hist)

        # extract a color histogram from the elliptical region and
        # update the feature vector
        hist = self.histogram(image, ellipMask)
        features.extend(hist)

        # return the feature vector
        return features

    def histogram(self, image, mask):
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
                            [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, dst = None).flatten()
        # return the histogram
        return hist



