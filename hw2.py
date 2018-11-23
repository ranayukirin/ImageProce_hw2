import numpy as np
import matplotlib.image as mpimg    # read image
import matplotlib.pyplot as plt     # plot image
import tkinter as tk    # GUI design
from PIL import Image   # used for save image
from numpy.fft import fft2, ifft2, fftshift, ifftshift  # fourier transform

# Set information of GUI
window = tk.Tk()
window.title('HW2')
window.geometry('300x450')


# Produce ideal high-pass filter
def ideal_high(shape, cutoff):
    h = np.ones(shape=shape)
    midx = np.floor(shape[0] / 2)
    midy = np.floor(shape[1] / 2)
    for x in range(shape[0]):
        for y in range(shape[1]):
            if np.sqrt(np.square(x - midx) + np.square(y - midy)) < cutoff:
                h[x, y] = 0
    return h


# Produce Gaussian high-pass filter
def gau_high(shape, sigma, c):
    h = np.empty(shape=shape)
    midx = np.floor(shape[0] / 2)
    midy = np.floor(shape[1] / 2)
    for x in range(shape[0]):
        for y in range(shape[1]):
            h[x, y] = 1 - np.exp(-c * ((np.square(x - midx) + np.square(y - midy)) / (2 * sigma * sigma)))
    return h


# Produce butterworth high-pass filter
def btw_high(shape, cutoff, order):
    h = np.empty(shape=shape)
    midx = np.floor(shape[0] / 2)
    midy = np.floor(shape[1] / 2)
    for x in range(shape[0]):
        for y in range(shape[1]):
            h[x, y] = 1 / (1 + np.power((cutoff / np.sqrt(np.square(x - midx) + np.square(y - midy))), 2 * order))
    return h


# Deblur fish image with wiener filter
def wiener_fish():
    im_fish = np.float32(mpimg.imread('1.jpg')) / 255
    k = 0.05
    sigma = 60
    midx = np.floor(im_fish.shape[0] / 2)
    midy = np.floor(im_fish.shape[1] / 2)

    # Use Gaussian low-pass filter as degradation filter
    h = np.empty(shape=(im_fish.shape[0], im_fish.shape[1]))
    for x in range(h.shape[0]):
        for y in range(h.shape[1]):
            h[x, y] = np.exp((-(np.square(x - midx) + np.square(y - midy)) / (2 * sigma * sigma)))
    # h = np.conj(h) / (np.square(np.abs(h)) + k)
    w = (1 / h) * np.square(np.abs(h)) / (np.square(np.abs(h)) + k)     # wiener filter
    fish_res = np.empty(shape=im_fish.shape)

    # Filtering in frequency domain for each layer of RGB
    for i in range(3):
        fi_shift = fftshift(fft2(im_fish[:, :, i]))
        fish_filted = fi_shift * w
        # fi_spec = np.log(np.abs(fi_shift))
        # fi_spec = np.uint8(255 * (fi_spec - fi_spec.min()) / (fi_spec.max() - fi_spec.min()))
        fish_back = np.real(ifft2(ifftshift(fish_filted)))
        fish_back = np.uint8(255 * (fish_back - fish_back.min()) / (fish_back.max() - fish_back.min()))
        fish_res[:, :, i] = fish_back

    # Plot and save image
    pltshow(im_fish, np.uint8(fish_res))
    fish_save = Image.fromarray(np.uint8(fish_res), 'RGB')
    fish_save.save('result/1_result(fish).jpg')


# Deblur 2.jpg(word) image with wiener filter
def wiener_word():
    im_word = np.float32(mpimg.imread('2.jpg')) / 255
    # Set some coefficient of motion filter and wiener filter
    t = 0.1
    a = -0.0001
    b = 0.0001
    k = 0.5

    # Use motion filter as degradation filter
    h = np.empty(shape=im_word.shape)
    for x in range(im_word.shape[0]):
        for y in range(im_word.shape[1]):
            temp = (x * a + y * b)
            if temp == 0:
                temp = ((x + 1e-10) * (a + 1e-15) + (y + 1e-10) * (b + 1e-15))
            h[x, y] = (t / (np.pi * temp)) * np.sin(np.pi * temp) * np.exp(-(0 + 1j) * np.pi * temp)
    w = np.conj(h) / (np.square(np.abs(h)) + k)     # wiener filter

    # Filtering in frequency domain
    ft_word = fft2(im_word)
    wo_shift = fftshift(ft_word)
    wo_filted = wo_shift * w
    wo_back = np.real(ifft2(ifftshift(wo_filted)))

    # Plot and save image
    pltshow(im_word, wo_back)
    wo_back = 255 * (wo_back - wo_back.min()) / (wo_back.max() - wo_back.min())
    wo_save = Image.fromarray(np.uint8(wo_back))
    wo_save.save('result/2_result(word).tif')


# Restore flower image with notch filter
def notch():
    im_flower = np.uint8(mpimg.imread('4.png')*255)

    # Doing fft and produce power spectrum
    ft_flower = fft2(im_flower)
    f_shift = fftshift(ft_flower)
    f_spec = np.log(np.abs(f_shift))
    f_spec = np.uint8(255 * (f_spec - f_spec.min()) / (f_spec.max() - f_spec.min()))

    # By observing power spectrum of image and produce notch filter to block the period noise
    idx = np.argwhere(f_spec > 200)
    for i in range(len(idx)):
        if idx[i, 0] < 400 or idx[i, 0] > 600:
            for x in range(10, -10, -1):
                for y in range(10, -10, -1):
                    if f_spec[idx[i, 0] - x, idx[i, 1] - y] > 140:
                        f_shift[idx[i, 0] - x, idx[i, 1] - y] = 0

    # Back to time domain and plot/save image
    f_shift_back = ifftshift(f_shift)
    flower_back = np.abs(ifft2(f_shift_back))
    flower_back = 255 * (flower_back - flower_back.min()) / (flower_back.max() - flower_back.min())
    pltshow(im_flower, flower_back)
    flower_save = Image.fromarray(np.uint8(flower_back))
    flower_save.save('result/4_result(flower).tif')


# Restore flower image with band reject filter
def band_reject():
    im_sea = np.uint8(mpimg.imread('3.png')*255)

    # Doing fft and produce power spectrum
    ft_sea = fft2(im_sea)
    s_shift = fftshift(ft_sea)
    s_spec = np.log(np.abs(s_shift))
    s_spec = np.uint8(255 * (s_spec - s_spec.min()) / (s_spec.max() - s_spec.min()))
    # plt.imshow(s_spec, cmap='gray')

    # By observing power spectrum of image and produce band reject filter to block the pattern noise
    idx = np.argwhere(s_spec > 190)
    for i in range(len(idx)):
        if np.abs(idx[i, 0] - 512) + np.abs(idx[i, 1] - 512) > 100:
            s_shift[idx[i, 0], idx[i, 1]] = 0
    s_shift_back = ifftshift(s_shift)
    sea_back = np.abs(ifft2(s_shift_back))
    sea_back = 255 * (sea_back - sea_back.min()) / (sea_back.max() - sea_back.min())

    # Plot and save image
    pltshow(im_sea, sea_back)
    sea_save = Image.fromarray(np.uint8(sea_back))
    sea_save.save('result/3_result(sea).tif')


# Receive filter name then call the responding function to do homomorphic filtering
def homomor(filt_name, rh, rl):
    im_street = np.float32(mpimg.imread('5.jpg')) / 255
    ft_street = fft2(np.log(im_street + 0.01))
    st_shift = fftshift(ft_street)

    # Choose filter
    if filt_name == 'ideal':
        filt = ideal_high(im_street.shape, 20)
    elif filt_name == 'gaussian':
        filt = gau_high(im_street.shape, 120, 2)
    elif filt_name == 'butterworth':
        filt = btw_high(im_street.shape, 120, 2)
    else:
        raise ValueError('Wrong filter name!')

    # Implement homomorphic filter on image then return it
    filt = (rh - rl) * filt + rl
    st_res = np.exp(np.real(ifft2(ifftshift(st_shift * filt)))) - 0.01

    return im_street, st_res


# According which button that user click to input the filter name to homomor and do some process
def homo_choose(name):
    im_ori, im_result = homomor(name, 1.2, 0.2)
    pltshow(im_ori, im_result)
    im_result = 255 * (im_result - im_result.min()) / (im_result.max() - im_result.min())
    homo_save = Image.fromarray(np.uint8(im_result))
    homo_save.save('result/5_result(' + str(name) + ').tif')


# Plot images with both origin and result image
def pltshow(im_ori, im_result):
    if np.ndim(im_result) == 3:
        plt.subplot(1, 2, 1), plt.imshow(im_ori), plt.title('Origin')
        plt.subplot(1, 2, 2), plt.imshow(im_result), plt.title('Result')
    else:
        plt.subplot(1, 2, 1), plt.imshow(im_ori, 'gray'), plt.title('Origin')
        plt.subplot(1, 2, 2), plt.imshow(im_result, 'gray'), plt.title('Result')
    plt.show()


# Set buttons and labels of GUI
lb1 = tk.Label(window, text='A. Image Restoration ',
               width=19, height=1).place(x=80, y=35)
btn1 = tk.Button(window, text='(a) wiener fish', width=15, height=2,
                 command=wiener_fish).place(x=30, y=70)
btn1_2 = tk.Button(window, text='(a) wiener word', width=15, height=2,
                   command=wiener_word).place(x=150, y=70)
btn2 = tk.Button(window, text='(b) flower', width=15, height=2,
                 command=notch).place(x=90, y=120)
btn3 = tk.Button(window, text='(c) sea', width=15, height=2,
                 command=band_reject).place(x=90, y=170)
lb2 = tk.Label(window, text='B. Homomorphic ',
               width=15, height=1).place(x=90, y=235)
btn4 = tk.Button(window, text='(a) ideal', width=15, height=2,
                 command=lambda: homo_choose('ideal')).place(x=90, y=270)
btn5 = tk.Button(window, text='(b) butterworth', width=15, height=2,
                 command=lambda: homo_choose('butterworth')).place(x=90, y=320)
btn6 = tk.Button(window, text='(c) gaussian', width=15, height=2,
                 command=lambda: homo_choose('gaussian')).place(x=90, y=370)

# Start GUI
window.mainloop()
