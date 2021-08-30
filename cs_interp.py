#!/usr/bin/env python

import numpy as np
import tkinter
import matplotlib
import random
import cvxpy as cvx
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from timeit import default_timer as timer
### IMP: User must run the docker image with the following display settings:
### docker run --net=host -e DISPLAY -v $HOME/.Xauthority:/root/.Xauthority:rw'

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

#fig=plt.figure(figsize=(10,10))
#plt.subplot(1,1,1)
#plt.imshow(data,cmap='seismic',interpolation="nearest",extent=(1,l2+1,fs*l1,0),vmin=-10000,vmax=10000,aspect="auto")
#plt.xlabel("Trace")
#plt.ylabel("Time(s)")
#plt.title("Segy file")
#plt.colorbar()
#plt.show()

### Generate the list of traces to be dropped
#numero=int(l2*(1-0.8))
#decimateIdxs = random.sample(range(0, l2), numero)
decimateIdxs = np.arange(1,l2,2) #Best option to experience aliasing

### Generate a mask for quickly extracting subarrays and its complement
decimateMask = np.zeros(data.shape, dtype=bool)
decimateMask[:,decimateIdxs] = True

data_error = np.copy(data); ### copy by value; deep-copy not required.
### Drop the traces (set them to zero)
data_error[:,decimateIdxs] = 0

#fig=plt.figure(figsize=(10,10))
#plt.subplot(1,1,1)
#plt.imshow(data,cmap='Greys_r',interpolation="nearest",extent=(1,l2+1,fs*l1,0),vmin=-10000,vmax=10000,aspect="auto")
#plt.imshow(data_error,cmap='seismic',interpolation="nearest",extent=(1,l2+1,fs*l1,0),vmin=-10000,vmax=10000,aspect="auto")
#plt.xlabel("Trace")
#plt.ylabel("Time(s)")
#plt.title("Segy file with dropped traces")
#plt.colorbar()
#plt.show()

### plot the know matrix where data==data_error
#known = np.zeros((l1, l2))
#known[data==data_error] = 1
#fig=plt.figure(figsize=(20,10))
#plt.subplot(1,1,1)
#plt.imshow(known,cmap='seismic',interpolation="nearest",extent=(1,l2+1,fs*l1,0),aspect="auto")
#plt.xlabel("Trace")
#plt.ylabel("Time(s)")
#plt.title("Segy file")
#plt.colorbar()
#plt.show()

### COMPRESSED SENSING PROBLEM

y=data_error[:,::2] #captions

n=400
ri=np.random.choice(l2,n,replace=False)
Phi = np.identity(l1)
Theta = np.fft.ifft2(Phi, norm='ortho')
T = Theta[ri] #random Dirac delta functions for max. incoherence

### Convex l_1 min problem

alpha = cvx.Variable(l1)
objetivo = cvx.Minimize(cvx.norm(alpha,1))
rest = [T@alpha == y]
prob = cvx.Problem(objetivo, rest)
result = prob.solve(verbose=True)
Xat2 = np.array(alpha.value).squeeze()