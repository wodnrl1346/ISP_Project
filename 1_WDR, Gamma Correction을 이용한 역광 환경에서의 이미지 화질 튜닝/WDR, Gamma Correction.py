from __future__ import print_function
from __future__ import division
import cv2
import numpy as np
from matplotlib import pyplot as plt

'''WDR'''
def loadExposureSeq():
  times = np.array([1/3800.0, 1/900.0, 1/150], dtype=np.float32)
  
  filenames = ["1_1.jpg", "1_2.jpg", "1_3.jpg"]

  images = []
  for filename in filenames:
    im = cv2.imread(filename)
    images.append(im)
  
  return images, times

'''Gamma Correction'''
def gamma_correction(img, gamma=1.0):
  normalized_img = img/255
  H, W, C  = img.shape

  corrected_img = np.zeros((H, W, C))

  for i in range(H):
      for j in range(W):
        for k in range(C):
          corrected_img[i,j,k] = normalized_img[i,j,k] ** gamma

  corrected_img = (corrected_img*255).astype(np.uint8)

  return corrected_img


if __name__ == "__main__":
  '''1. WDR'''
  # 1-1. Capture multiple images with different exposures
  images, times = loadExposureSeq()
  
  # 1-2. Align Images
  print("Aligning images ... ")
  alignMTB = cv2.createAlignMTB()
  alignMTB.process(images, images)

  # 1-3. Recover the Camera Response Function
  # CFR이 scene brightness에 nonlinear 하기 때문에 이미지의 픽셀이 exposure time에 비례하지 않아, 너무 밝거나 어두운 bad pixel 有
  # pixel brightness = (pixel value / exposure time)
  # bad pixel이 아닌 것
  
  print("Calculating Camera Response Function (CRF) ... ")
  calibrateDebevec = cv2.createCalibrateDebevec()
  responseDebevec = calibrateDebevec.process(images, times)
  
  # 1-4. Merge images into an HDR linear image
  print("Merging images into one HDR image ... ")
  mergeDebevec = cv2.createMergeDebevec()
  hdrDebevec = mergeDebevec.process(images, times, responseDebevec)
  # Save HDR image.
  cv2.imwrite("hdrDebevec.hdr", hdrDebevec)
  print("saved hdrDebevec.hdr ")
  
  # 1-5. Mantiuk Tonemap
  print("Tonemaping using Mantiuk's method ... ")
  tonemapMantiuk = cv2.createTonemapMantiuk(2.2,0.85, 1.2)
  ldrMantiuk = tonemapMantiuk.process(hdrDebevec)
  ldrMantiuk = 3 * ldrMantiuk
  cv2.imwrite("ldr-Mantiuk.jpg", ldrMantiuk * 255)
  print("saved ldr-Mantiuk.jpg")
  
  
  '''2. Gamma Correction'''
  img_WDR = cv2.imread("ldr-Mantiuk.jpg", 1)
  img_Gamma_Correction = gamma_correction(img_WDR, gamma=0.7)
  
  plt.hist(img_WDR.ravel(), 256, [0,256], color='r', label='WDR image')
  plt.hist(img_Gamma_Correction.ravel(), 256, [0,256], color='g', label='Gamma Corrected image')
  plt.legend(loc='best')
  plt.show()
  
  Original_image = cv2.resize(img_WDR, dsize=(480, 480), interpolation=cv2.INTER_AREA)
  Corrected_image = cv2.resize(img_Gamma_Correction, dsize=(480, 480), interpolation=cv2.INTER_AREA)
  
  cv2.imshow('WDR_image', img_WDR)
  cv2.imshow('Corrected_image', img_Gamma_Correction)

  cv2.waitKey()
  cv2.destroyAllWindows()