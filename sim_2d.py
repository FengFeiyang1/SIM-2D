# %%
from itertools import product

import numpy as np
from skimage.io import imsave, imread
from skimage.util import img_as_ubyte
from scipy.stats import multivariate_normal
from scipy.signal import convolve2d
from numpy.fft import fft2, ifft2, fftshift, ifftshift

# %%
img_real: np.ndarray = imread('../latex/images/IMG_1482.JPG', as_gray=True)
imsave('../latex/images/real.png', img_as_ubyte(img_real))

height, width = img_real.shape

# %%
psf_x, psf_y = np.mgrid[-3:3:31j, -3:3:31j]
psf_pos = np.dstack((psf_x, psf_y))
gaussian_2d = multivariate_normal(np.zeros(2), np.eye(2))
psf = gaussian_2d.pdf(psf_pos)
imsave('../latex/images/psf.png', img_as_ubyte(psf / psf.max()))

# %%
img_real_psf = convolve2d(img_real, psf, 'same')
img_real_psf = img_real_psf / img_real_psf.max()
imsave('../latex/images/real_psf.png', img_as_ubyte(img_real_psf))

# %%
theta0, phi0 = 0, 0
map_theta_k = lambda theta: np.array([np.cos(theta), np.sin(theta)]) * 2 * np.pi / 10
k0 = map_theta_k(theta0)

si_x, si_y = np.mgrid[0:height, 0:width]
si_pos = np.dstack((si_x, si_y))

def compute_si(k, phi):
    si = np.zeros_like(img_real)
    for i0, i1 in product(range(height), range(width)):
        si[i0, i1] = 1 + np.cos(np.inner(k, si_pos[i0, i1]) + phi)
    return si

si0 = compute_si(k0, phi0)

# %%
imsave('../latex/images/si0.png', img_as_ubyte(si0 / si0.max()))

img_real_si0 = convolve2d(img_real * si0, psf, 'same')
img_real_si0 = img_real_si0 / img_real_si0.max()
imsave('../latex/images/real_si0.png', img_as_ubyte(img_real_si0))

# %%
psf_augment = np.zeros_like(img_real)
psf_augment[1051:1082, 785:816] = psf
imsave('../latex/images/psf_aug.png', img_as_ubyte(psf_augment / psf_augment.max()))

otf_augment = fftshift(fft2(psf_augment))
imsave('../latex/images/otf_aug.png', img_as_ubyte((np.abs(otf_augment) / np.abs(otf_augment).max()) ** 1))

# %%
img_recipro = fftshift(fft2(img_real))
imsave('../latex/images/recipro.png', img_as_ubyte((np.abs(img_recipro) / np.abs(img_recipro).max()) ** 0.2))

img_recipro_otf = img_recipro * otf_augment
imsave('../latex/images/recipro_otf.png', img_as_ubyte((np.abs(img_recipro_otf) / np.abs(img_recipro_otf).max()) ** 0.2))

img_restore_psf = fftshift(np.real(ifft2(ifftshift(img_recipro_otf))))

imsave('../latex/images/restore_psf.png', img_as_ubyte(img_restore_psf / img_restore_psf.max()))

# %%
img_recipro_si0 = fftshift(fft2(img_real * si0))
imsave('../latex/images/si0_reci.png', img_as_ubyte((np.abs(img_recipro_si0) / np.abs(img_recipro_si0).max()) ** 0.2))

img_si0_net = np.abs(img_recipro_si0) - np.abs(img_recipro)
imsave('../latex/images/si0_net.png', img_as_ubyte((np.abs(img_si0_net) / np.abs(img_si0_net).max()) ** 0.2))

# %%
phi1, phi2 = np.pi / 3, 2 * np.pi / 3
si1, si2 = compute_si(k0, phi1), compute_si(k0, phi2)

img_real_si0 = convolve2d(img_real * si0, psf, 'same')
img_real_si1 = convolve2d(img_real * si1, psf, 'same')
img_real_si2 = convolve2d(img_real * si2, psf, 'same')

img_reci_si0 = fftshift(fft2(img_real_si0))
img_reci_si1 = fftshift(fft2(img_real_si1))
img_reci_si2 = fftshift(fft2(img_real_si2))

reci_A = np.array([
    [1, np.exp(-1j * phi0), np.exp(1j * phi0)],
    [1, np.exp(-1j * phi1), np.exp(1j * phi1)],
    [1, np.exp(-1j * phi2), np.exp(1j * phi2)]
])

reci_b = np.dstack([img_reci_si0, img_reci_si1, img_reci_si2]).reshape((2133 * 1599, 3)).T

reci_ori, reci_posi, reci_nega = np.linalg.solve(reci_A, reci_b)

# %%
reci_ori = reci_ori.reshape(2133, 1599)
reci_posi = reci_posi.reshape(2133, 1599)
reci_nega = reci_nega.reshape(2133, 1599)

imsave('../latex/images/reci_ori.png', img_as_ubyte((np.abs(reci_ori) / np.abs(reci_ori).max()) ** 0.2))
imsave('../latex/images/reci_posi.png', img_as_ubyte((np.abs(reci_posi) / np.abs(reci_posi).max()) ** 0.2))
imsave('../latex/images/reci_nega.png', img_as_ubyte((np.abs(reci_nega) / np.abs(reci_nega).max()) ** 0.2))

# %%
reci_posi_shift = np.zeros_like(img_real, dtype=complex)
reci_posi_shift[:-213, :] =  reci_posi[213:, :]
reci_nega_shift = np.zeros_like(img_real, dtype=complex)
reci_nega_shift[213:, :] =  reci_posi[:-213, :]

reci_full = reci_ori + reci_posi_shift + reci_nega_shift
imsave('../latex/images/reci_full.png', img_as_ubyte((np.abs(reci_full) / np.abs(reci_full).max()) ** 0.2))

# %%
restore_full = ifft2(ifftshift(reci_full))
restore_full = np.real(restore_full)
imsave('../latex/images/restore_full.png', img_as_ubyte(restore_full / restore_full.max()))

# %%
theta1 = np.pi / 2
k1 = map_theta_k(theta1)

si10 = compute_si(k1, phi0)
si11 = compute_si(k1, phi1)
si12 = compute_si(k1, phi2)

img_real_si10 = convolve2d(img_real * si10, psf, 'same')
img_real_si11 = convolve2d(img_real * si11, psf, 'same')
img_real_si12 = convolve2d(img_real * si12, psf, 'same')

img_reci_si10 = fftshift(fft2(img_real_si10))
img_reci_si11 = fftshift(fft2(img_real_si11))
img_reci_si12 = fftshift(fft2(img_real_si12))

reci_b1 = np.dstack([img_reci_si10, img_reci_si11, img_reci_si12]).reshape((2133 * 1599, 3)).T

reci_ori1, reci_posi1, reci_nega1 = np.linalg.solve(reci_A, reci_b1)

reci_ori1 = reci_ori1.reshape(2133, 1599)
reci_posi1 = reci_posi1.reshape(2133, 1599)
reci_nega1 = reci_nega1.reshape(2133, 1599)

reci_posi1_shift = np.zeros_like(img_real, dtype=complex)
reci_posi1_shift[:, :-160] =  reci_posi1[:, 160:]
reci_nega1_shift = np.zeros_like(img_real, dtype=complex)
reci_nega1_shift[:, 160:] =  reci_posi1[:, :-160]

reci_full1 = reci_ori1 + reci_posi1_shift + reci_nega1_shift

restore_full1 = ifft2(ifftshift(reci_full1))
restore_full1 = np.real(restore_full1)
imsave('../latex/images/restore_full1.png', img_as_ubyte(restore_full1 / restore_full1.max()))

# %%
reci_full_bi = (reci_ori + reci_ori1) / 2 + reci_posi_shift + reci_nega_shift + reci_posi1_shift + reci_nega1_shift
imsave('../latex/images/reci_full_bi.png', img_as_ubyte((np.abs(reci_full_bi) / np.abs(reci_full_bi).max()) ** 0.2))

restore_full_bi = ifft2(ifftshift(reci_full_bi))
restore_full_bi = np.real(restore_full_bi)
imsave('../latex/images/restore_full_bi.png', img_as_ubyte(restore_full_bi / restore_full_bi.max()))


