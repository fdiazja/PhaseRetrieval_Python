import numpy as np
from skimage.restoration import inpaint

#NaN, inf, -inf cleaning

def cleanup(_a):
    _b = np.ma.masked_invalid(_a)
    _a = inpaint.inpaint_biharmonic(_a, _b.mask, multichannel=False)
    return _a

#Data reading

Ns = 100	
sh = (Nz, Ny, Nx) = (201, 1022, 579)
measdata = np.memmap('meas_data_%dx%dx%d.raw'%sh[::-1], dtype=np.uint16, offset=0, shape=sh, mode='r').astype(np.float32)
dark = measdata[Ns]
data = measdata[0:Ns] - dark
flat = measdata[Ns + 1:2 * Ns + 1] - dark
data.astype(np.float32).tofile('data_%dx%dx%d.raw'%data.shape[::-1])
flat.astype(np.float32).tofile('flat_%dx%dx%d.raw'%flat.shape[::-1])
dark.astype(np.float32).tofile('dark_%dx%d.raw'%dark.shape[::-1])

#Phase retrieval using FFT

fflat = np.fft.fft(flat, axis=0)
fdata = np.fft.fft(data, axis=0)
ab = abs(fdata[0]) / abs(fflat[0])
df = (abs(fdata[1]) * abs(fflat[0])) / (abs(fflat[1]) * abs(fdata[0]))
dp = (np.mod(np.angle(fdata[1]) - np.angle(fflat[1]) + np.pi, 2 * np.pi)) - np.pi
ab = -np.log(ab)
df = -np.log(df)

#Data cleaning

ab = cleanup(ab)
df = cleanup(df)

ab.astype(np.float32).tofile('ab_%dx%d.raw'%ab.shape[::-1])
df.astype(np.float32).tofile('df_%dx%d.raw'%df.shape[::-1])
dp.astype(np.float32).tofile('dp_%dx%d.raw'%dp.shape[::-1])