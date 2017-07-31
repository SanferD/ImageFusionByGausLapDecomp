from __future__ import division

import cv2, os
import numpy as np
from matplotlib import pyplot as plt
import img_plotter

def main():
	one_img = readImage('camp_thm.bmp')
	two_img = readImage('camp_viz.bmp')

	one = contrast_stretch(one_img)
	two = contrast_stretch(two_img)

	fused = fuse(one, two, METHOD='b')

def fuse(one, two, METHOD = 'a'):
	k1, k2, k3 = (31, 3, 31) if METHOD == 'a' else (131, 5, 51)

	G_one = cv2.GaussianBlur(one, (k1 ,k1), 0)
	G_two = cv2.GaussianBlur(two, (k1, k1), 0)

	L_one = one - G_one
	L_two = two - G_two

	# wm = np.zeros(one.shape, dtype=np.float64)
	wm = 255*np.uint8(np.abs(L_one) > np.abs(L_two))
	wm_blur = cv2.GaussianBlur(wm, (k2, k2), 0)

	L_fus = None
	if METHOD == 'a':
		L_fus_one = np.multiply(one, wm_blur, dtype=np.float64)
		L_fus_two = np.multiply(two, 255 - wm_blur, dtype=np.float64)
		L_fus = np.true_divide(L_fus_one + L_fus_two, 255)
	else:
		L_fus = L_two.copy()
		indexes = np.where(wm > 127)
		L_fus[indexes] = L_one[indexes]

	G1 = np.rint(255*G_one).astype(np.uint8)
	G2 = np.rint(255*G_two).astype(np.uint8)
	
	h_one = cv2.calcHist([G1], [0], None, [256], [0, 256])
	h_two = cv2.calcHist([G2], [0], None, [256], [0, 256])

	hmap_one = np.vectorize(lambda x: h_one[x])
	hmap_two = np.vectorize(lambda x: h_two[x])

	indexes = np.where(  hmap_two(G2) > hmap_one(G1)  )
	mh = G_two.copy()
	mh[indexes] = G_one[indexes]
	G_fus = cv2.GaussianBlur(mh, (k3, k3), 0)	

	fused = contrast_stretch(G_fus + L_fus)
	cv2.imshow('fused', fused)
	# cv2.waitKey(0)

	imgs = [ one, G_one, L_one,
			 two, G_two, L_two,
			 wm, wm_blur, L_fus,
			 mh, G_fus, fused ]
	titles = [ 'viz', 'G_viz', 'L_viz',
			   'thm', 'G_thm', 'L_thm',
			   'Lw', 'Lw-bar', 'L_fus',
			   'Mh', 'G_fus', 'fused' ]
	suptitle = 'Paper Fusion: Optimal' if METHOD == 'a' else 'Paper Fusion: Large'
	img_plotter.plot_images(imgs=imgs, titles=titles, suptitle=suptitle)	

	return fused

dataset = './'
def readImage(fn):
	file_path = os.path.join(dataset, fn)
	return cv2.imread(file_path, 0)

def normalize(img):
	return np.true_divide(img, 255).astype(np.float64)

def contrast_stretch(img):
	dif = img - img.min()
	return np.true_divide(dif, img.max() - img.min())

if __name__ == '__main__':
	main()