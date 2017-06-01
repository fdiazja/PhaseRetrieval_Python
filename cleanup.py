"""
Different methods to cleanup images getting rid of nans and infinity values
"""

import numpy as np
from scipy import interpolate
from skimage.restoration import inpaint

#Cleanup using masked arrays and inpaint (inpaint fills pixel positions using biharmonic equation).
#This method is the fastest and most effective
def cleanup(_a):
    _b = np.ma.masked_invalid(_a)
    _a = inpaint.inpaint_biharmonic(_a, _b.mask, multichannel=False)
    return _a

"""
#Cleanup using a 3x3 kernel and averaging masked values
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

def cleanup(_a):
    _a = np.where(~np.isinf(_a), _a, np.nan)
    _b = zip(*np.where(np.isnan(_a)))
    for i in range(len(b)):
        _iy, _ix = _b[i]
        _a = nkernel(_a, _ix, _iy)
    return _a
"""

"""
# No consecutive Nans (works)
def cleanup(_a):
    _a = np.lib.pad(_a, (1, 1), 'constant', constant_values=0)
    _a = np.where(~np.isinf(_a), _a, np.nan)
    _b = np.where(np.isnan(_a))
    _a[_b] = (_a[_b[0], _b[1] + 1] + _a[_b[0], _b[1] - 1]) / 2
    _a = _a[1:-1,1:-1]
    return _a
"""	

"""	
#Cleanup using 2D linear interpolation (works, but slowly)
def cleanup(_a):
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
"""
a = np.random.rand(5, 5)

a[0, 0] = np.inf
a[0, 1] = np.nan
a[1, 4] = np.inf 
a[4, 0] = -np.nan 
a[4, 3] = -np.inf 
print a
b = cleanup(a)
print b

