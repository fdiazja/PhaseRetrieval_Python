"""
Comparisson of different methods for data cleaning (see cleaunp.py)
"""
import numpy as np
from skimage.restoration import inpaint
from scipy import interpolate

#NaN, inf, -inf cleaning

def cleanup1(_a):
    _b = np.ma.masked_invalid(_a)
    _a = inpaint.inpaint_biharmonic(_a, _b.mask, multichannel=False)
    return _a

def nkernel(_a, ix, iy):
    ix += 1
    iy += 1
    _a = np.lib.pad(_a, (1, 1), 'constant', constant_values=0)
    k = np.array([[_a[iy - 1, ix - 1], _a[iy - 1, ix], _a[iy - 1, ix + 1]],\
	              [_a[iy, ix - 1], _a[iy, ix], _a[iy, ix + 1]],\
				  [_a[iy + 1, ix - 1], _a[iy + 1, ix], _a[iy + 1, ix + 1]]])
    k = np.ma.masked_invalid(k)
    k[np.where(k.mask)] = np.mean(k)
    _a[iy, ix] = k[1, 1]
    _a = _a[1:-1,1:-1]
    return _a

def cleanup2(_a):
    _a = np.where(~np.isinf(_a), _a, np.nan)
    _b = zip(*np.where(np.isnan(_a)))
    for i in range(len(_b)):
        _iy, _ix = _b[i]
        _a = nkernel(_a, _ix, _iy)
    return _a

def cleanup3(_a):
    _a = np.lib.pad(_a, (1, 1), 'constant', constant_values=0)
    _a = np.ma.masked_invalid(_a)
    x = np.arange(0, _a.shape[1])
    y = np.arange(0, _a.shape[0])
    xx, yy = np.meshgrid(x, y)
    x1 = xx[~_a.mask]
    y1 = yy[~_a.mask]
    newarr = _a[~_a.mask]
    _a = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='linear')
    _a = _a[1:-1,1:-1]
    return _a
	
#Data reading

Ns = 100	
sh = (Nz, Ny, Nx) = (201, 1022, 579)
measdata = np.memmap('meas_data_%dx%dx%d.raw'%sh[::-1], dtype=np.uint16, offset=0, shape=sh, mode='r').astype(np.float32)
dark = measdata[Ns] # Dark image 
data = measdata[0:Ns] - dark
flat = measdata[Ns + 1:2 * Ns + 1] - dark #Flat fields
data.astype(np.float32).tofile('data_%dx%dx%d.raw'%data.shape[::-1])
flat.astype(np.float32).tofile('flat_%dx%dx%d.raw'%flat.shape[::-1])
dark.astype(np.float32).tofile('dark_%dx%d.raw'%dark.shape[::-1])

#Phase retrieval using FFT

fflat = np.fft.fft(flat, axis=0)
fdata = np.fft.fft(data, axis=0)
ab = abs(fdata[0]) / abs(fflat[0]) #Absorption image
df = (abs(fdata[1]) * abs(fflat[0])) / (abs(fflat[1]) * abs(fdata[0])) #Dark field image
dp = (np.mod(np.angle(fdata[1]) - np.angle(fflat[1]) + np.pi, 2 * np.pi)) - np.pi #Diff. Phase image

#NaN analysis

nanx = np.random.random_integers(0, 578, 400)
nany = np.random.random_integers(0, 1021, 400)
ab[nany, nanx] = np.nan
ab_nan = ab.copy()
ab_nan[np.where(np.isnan(ab_nan))] = 0 #Display zeros where there are nans (just for visualization)
ab_nan.astype(np.float32).tofile('ab_nan_%dx%d.raw'%ab_nan.shape[::-1])

#Data cleaning

ab_paint = cleanup1(ab)
ab_ker = cleanup2(ab)
ab_int = cleanup3(ab)

ab_paint.astype(np.float32).tofile('ab_paint_%dx%d.raw'%ab_paint.shape[::-1])
ab_ker.astype(np.float32).tofile('ab_ker_%dx%d.raw'%ab_ker.shape[::-1])
ab_int.astype(np.float32).tofile('ab_int_%dx%d.raw'%ab_int.shape[::-1])
df.astype(np.float32).tofile('df_%dx%d.raw'%df.shape[::-1])
dp.astype(np.float32).tofile('dp_%dx%d.raw'%dp.shape[::-1])