import numpy as np
import cv2
import glob
import os

class Stitcher:
    def __init__(self, max_features=500, ratio=0.75, reproj_thresh=4.0):
        self.sift = cv2.SIFT_create(nfeatures=max_features)
        self.ratio = ratio
        self.reproj_thresh = reproj_thresh

    def detectAndDescribe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kps, features = self.sift.detectAndCompute(gray, None)
        return np.float32([kp.pt for kp in kps]), features

    def matchKeypoints(self, kpsA, featuresA, kpsB, featuresB):
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(featuresA, featuresB, k=2)
        matches = []

        for m in raw_matches:
            if len(m) == 2 and m[0].distance < m[1].distance * self.ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        if len(matches) >= 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            H, status = self.findHomography(ptsA, ptsB)
            return matches, H, status

        return None, None, None

    def findHomography(self, ptsA, ptsB):
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, self.reproj_thresh)
        return H, status

    def warpImages(self, imgA, imgB, H):
        height, width = imgB.shape[:2]
        warpA = cv2.warpPerspective(imgA, H, (width * 2, height))
        warpA[0:imgB.shape[0], 0:imgB.shape[1]] = imgB
        return warpA

    def blendImages(self, imgA, imgB):
        maskA = self.createMask(imgA)
        maskB = self.createMask(imgB)

        overlap = maskA & maskB
        if not np.any(overlap):
            return np.maximum(imgA, imgB)

        blended = imgA * maskA + imgB * maskB * (1 - maskA)
        return blended

    def createMask(self, img):
        mask = (img != 0).astype(np.uint8)
        return cv2.merge([mask, mask, mask])

    def stitch(self, images):
        if len(images) == 0:
            return None
        
        panorama = images[0]
        for i in range(1, len(images)):
            kpsA, featuresA = self.detectAndDescribe(panorama)
            kpsB, featuresB = self.detectAndDescribe(images[i])
            
            matches, H, _ = self.matchKeypoints(kpsA, featuresA, kpsB, featuresB)
            if H is None:
                print("Error: Homography could not be calculated.")
                continue

            panorama = self.warpImages(panorama, images[i], H)
            panorama = self.blendImages(panorama, images[i])

        return panorama

# Usage
if __name__ == "__main__":
    stitcher = Stitcher()
    img_files = sorted(glob.glob("C:\Users\salon\Downloads\CV_A3\ES666-Assignment3\results\I1*.JPG"))
    images = [cv2.imread(img) for img in img_files]

    result = stitcher.stitch(images)
    if result is not None:
        cv2.imwrite("C:\Users\salon\Downloads\CV_A3\ES666-Assignment3\results\I2\new.jpg", result)
    else:
        print("No panorama created.")
