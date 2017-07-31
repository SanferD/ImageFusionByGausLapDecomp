from matplotlib import pyplot as plt
import numpy as np
import cv2

def plot_images(imgs, titles, suptitle=None, cols=-1):
	n = min(len(imgs), len(titles))
	if cols <= 0:
		cols = 3
	c = cols if n > cols else n
	r = np.ceil(float(n)/float(c)).astype(np.int)

	imgplot = __make_imgplot(rows=r, cols=c, suptitle=suptitle)
	for img, title in zip(imgs, titles):
		if len(img.shape) == 3:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		imgplot(img, title)
	imgplot.show()

def save_images(imgs, titles):
	for img, title in zip(imgs, titles):
		title += '.png'
		cv2.imwrite(title, img)

def __make_imgplot(rows, cols, suptitle=None):
	fig = plt.figure()
	if suptitle is not None:
		fig.suptitle(suptitle, fontsize=24)
	cache = {'i':1}
	
	def subplot(img, title):
		ax = fig.add_subplot(rows, cols, cache['i'])
		if len(img.shape) == 3:
			ax.imshow(img)
		else:
			ax.imshow(img, cmap='gray')
			
		ax.set_title(title), ax.set_xticks([]), ax.set_yticks([])
		cache['i'] += 1
	
	def plot():
		plt.tight_layout()
		plt.show()	
	subplot.show = plot
	return subplot

def rescale(img):
	mag = np.abs(img) if iscomplex(img) else img 
	I = normalize(mag)
	return np.rint(255*I).astype(np.uint8)

def normalize(img):
	return np.true_divide(img - img.min(), img.max() - img.min())

def iscomplex(img):
	return np.iscomplexobj(img)