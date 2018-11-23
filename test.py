import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
from numpy.fft import fft2, ifft2, fftshift, ifftshift

K = 0.05
im_fish = np.float32(mpimg.imread('1.jpg')) / 255
kernel = np.empty(shape=(im_fish.shape[0], im_fish.shape[1]))
sigma = 60
midx = np.floor(im_fish.shape[0] / 2)
midy = np.floor(im_fish.shape[1] / 2)
for x in range(kernel.shape[0]):
    for y in range(kernel.shape[1]):
        kernel[x, y] = np.exp((-(np.square(x - midx) + np.square(y - midy)) / (2 * sigma * sigma)))
# kernel = np.conj(kernel) / (np.square(np.abs(kernel)) + K)
kernel = (1 / kernel) * np.square(np.abs(kernel)) / (np.square(np.abs(kernel)) + K)
fish_res = np.empty(shape=im_fish.shape)
for i in range(3):
    fi_shift = fftshift(fft2(im_fish[:, :, i]))
    fish_filted = fi_shift * kernel
    # fi_spec = np.log(np.abs(fi_shift))
    # fi_spec = np.uint8(255 * (fi_spec - fi_spec.min()) / (fi_spec.max() - fi_spec.min()))
    fish_back = np.real(ifft2(ifftshift(fish_filted)))
    fish_back = (fish_back - fish_back.min()) / (fish_back.max() - fish_back.min())
    fish_res[:, :, i] = fish_back
plt.imshow(fish_res)
plt.show()
# im_word = np.float32(mpimg.imread('2.jpg')) / 255
# T = 1
# a = -0.008
# b = 0.007
# K = 1
# h = np.empty(shape=im_word.shape)
# for x in range(im_word.shape[0]):
#     for y in range(im_word.shape[1]):
#         temp = (x * a + y * b)
#         if temp == 0:
#             temp = ((x + 1e-10) * (a + 1e-15) + (y + 1e-10) * (b + 1e-15))
#         h[x, y] = (T / (np.pi * temp)) * np.sin(np.pi * temp) * np.exp(-(0 + 1j) * np.pi * temp)
#         # h[x, y] = (T / (np.pi * temp))
# h = np.conj(h) / (np.square(np.abs(h)) + K)
# im_ft = fft2(im_word)
# im_shift = fftshift(im_ft)
# im_filted = im_shift * h
# im_back = np.real(ifft2(ifftshift(im_filted)))
# plt.imshow(im_back, 'gray')
# plt.show()
