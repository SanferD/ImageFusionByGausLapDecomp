# ImageFusionByGausLapDecomp

Image fusion based off Gausian and Laplacian decomposition.
Let I be the image and G be the gaussian of the image.
It can be shown that the laplacian of the image is L = I - G.
Equivalently, I = G + L.
The idea then is to get the gaussian of the visual and thermal images and fuse them to get a new gaussian.
Also get the laplacian of the visual and thermal images and fuse them to get a new laplacian.
Finally add the fused gaussian and the fused laplacian to create the fused image.

Test images were taken from the TNO Image Fusion dataset.

Algorithm is based off the paper

Seohyung Lee and Daeho Lee, “Fusion of IR and Visual Images Based on Gaussian and Laplacian Decomposition Using Histogram Distributions and Edge Selection,” Mathematical Problems in Engineering, vol. 2016, Article ID 3130681, 9 pages, 2016. doi:10.1155/2016/3130681