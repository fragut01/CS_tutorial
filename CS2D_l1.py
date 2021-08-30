import numpy as np
import tkinter
import matplotlib
import random
import cvxpy as cvx
import scipy.fftpack as spfft
from pylbfgs import owlqn
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

filename="../data/gob_20200731_synthetic_shot-gather.sgy"

### Read a stream of the segy file using ObsPy
from obspy.io.segy.core import _read_segy
st = _read_segy(filename,
                textual_header_encoding="EBCDIC",
                unpack_trace_headers=True)

l1    = st[0].stats.npts ### Get the number of samples
l2    = len(st) ### Get number of traces
fs    = st[0].stats.sampling_rate ### Get the sampling rate(samples by second)
delta = st[0].stats.delta ### Get the time sampling (fs = 1/delta ?)

### Construct an ndarray of all traces.
data = np.stack([t.data for t in st.traces]).T ### shape = (l1, l2)
fig=plt.figure(figsize=(10,10))
plt.subplot(1,1,1)
plt.imshow(data,cmap='seismic',interpolation="nearest",extent=(1,l2+1,fs*l1,0),vmin=-10000,vmax=10000,aspect="auto")
plt.xlabel("Trace")
plt.ylabel("Time(s)")
plt.title("Segy file")
plt.colorbar()
plt.show()

# Coefficient for the L1 norm of variables (see OWL-QN algorithm)
ORTHANTWISE_C = 5

def dct2(x):
    """Return 2D discrete cosine transform.
    """
    return spfft.dct(
        spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


def idct2(x):
    """Return inverse 2D discrete cosine transform.
    """
    return spfft.idct(
        spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

sample_size = 0.5
# create random sampling index vector
k = round(l2 * sample_size)
ri = np.random.choice(l2, k, replace=False) # random sample of indices
b = data[:,ri]
b = b.flat
b = np.expand_dims(b, axis=1)


#k=int(l2*sample_size)
#decimateIdxs = random.sample(range(0, l2), k)
### Generate a mask for quickly extracting subarrays and its complement
#decimateMask = np.ones(data.shape, dtype=bool)
#decimateMask[:,decimateIdxs] = False


#data_error = np.copy(data); ### copy by value; deep-copy not required.
### Drop the traces (set them to zero)
#data_error[:,decimateIdxs] = 0



def evaluate(x, g, step):
    """An in-memory evaluation callback."""

    # we want to return two things: 
    # (1) the norm squared of the residuals, sum((Ax-b).^2), and
    # (2) the gradient 2*A'(Ax-b)

    # expand x columns-first
    x2 = x.reshape((l2, l1)).T

    # Ax is just the inverse 2D dct of x2
    Ax2 = idct2(x2)

    # stack columns and extract samples
    Ax = Ax2.T.flat[ri].reshape(b.shape)

    # calculate the residual Ax-b and its 2-norm squared
    Axb = Ax - b
    fx = np.sum(np.power(Axb, 2))

    # project residual vector (k x 1) onto blank image (ny x nx)
    Axb2 = np.zeros(x2.shape)
    Axb2.T.flat[ri] = Axb # fill columns-first

    # A'(Ax-b) is just the 2D dct of Axb2
    AtAxb2 = 2 * dct2(Axb2)
    AtAxb = AtAxb2.T.reshape(x.shape) # stack columns

    # copy over the gradient vector
    np.copyto(g, AtAxb)

    return fx

# create images of mask (for visualization)
#Z = [np.zeros(data.shape, dtype='uint8')]
mask = np.zeros(data.shape)
#Xm = np.zeros(data.shape)
mask[:,ri] = data[:,ri]
# take random samples of image, store them in a vector b
b = data[:,ri]
b = b.flat
#b = b.astype(float)

# perform the L1 minimization in memory
Xat2 = owlqn(l2*l1, evaluate, None, ORTHANTWISE_C)

# transform the output back into the spatial domain
Xat = Xat2.reshape(l2, l1).T # stack columns
Xa = idct2(Xat)