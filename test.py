import numpy as np

image = np.zeros((256, 256), dtype=np.float32)
mask = np.zeros((256, 256), dtype=np.int64)
print("image:", image.shape, image.ndim, image.size, image.dtype, image.nbytes)
print("mask:", mask.shape, mask.ndim, mask.size, mask.dtype, mask.nbytes)