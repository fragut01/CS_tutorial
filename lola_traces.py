import numpy as np
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import matplotlib.pyplot as plt
from pylbfgs import owlqn
import imageio

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

def progress(x, g, fx, xnorm, gnorm, step, k, ls):
    """Just display the current iteration.
    """
    print('Iteration {}'.format(k))
    return 0

_image_dims = None  # track target image dimensions here
_ri_vector = None  # reference the random sampling indices here
_b_vector = None  # reference the sampled vector b here
_A_matrix = None  # reference the dct matrix operator A here

def evaluate(X, g, step):
    """An in-memory evaluation callback.
    """

    # we want to return two things:
    # (1) the norm squared of the residuals, sum((Ax-b).^2), and
    # (2) the gradient 2*A'(Ax-b)

    # Ax is just the inverse dct of the spectral image
    X2 = X.reshape((_image_dims[0], _image_dims[1]))
    Ax2 = idct2(X2)
    #Ax2 = np.fft.ifft2(x2, norm = 'ortho').real
    # stack columns and extract samples
    Ax_c = Ax2[:,_ri_vector]
    Ax = np.ravel(Ax_c.T)

    # calculate the residual Ax-b and its 2-norm squared
    Axb = Ax - _b_vector
    fx = np.sum(np.power(Axb, 2))

    # project residual vector (k x 1) onto blank image (ny x nx)
    Axb2 = np.zeros(X2.shape)
    Ax_reshaped=np.reshape(Ax,(-1,len(_ri_vector)),order='F')
    Axb2[:,_ri_vector] = Ax_reshaped

    # A^t(Ax-b) is just the 2D dct of Axb2
    AtAxb2 = 2 * dct2(Axb2)
    #AtAxb2 = 2 * np.abs(np.fft.fft2(Ax, norm = 'ortho'))
    AtAxb = AtAxb2.reshape(X.shape)  # stack columns

    # copy over the gradient vector
    np.copyto(g, AtAxb)

    return fx

SCALE=0.1
SAMPLE=0.3
Xorig = imageio.imread('LOLA.jpg', as_gray=True, pilmode='L')
X = spimg.zoom(Xorig, SCALE)
ny, nx = X.shape
_image_dims = (ny, nx)
k2 = round(nx * SAMPLE)
r2 = np.random.choice(nx, k2, replace = False)
b2_c = X[:,r2]
b2 = np.ravel(b2_c.T)
_ri_vector = r2
_b_vector = np.expand_dims(b2, axis=1)

# perform the L1 minimization in memory
Xat2 = owlqn(nx*ny, evaluate, progress, 10)

    # transform the output back into the spatial domain
Xat = Xat2.reshape(nx, ny)  # stack columns
Xa = idct2(Xat)
#Xa = np.abs(np.fft.ifft2(Xat, norm = 'ortho')).real
    # create images of mask (for visualization)
mask = np.zeros(X.shape)
mask[:,r2] = 255
Xm = 255 * np.ones(X.shape)    
Xm[:,r2] = X[:,r2]

    # display the result
f, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].imshow(X, cmap='gray', interpolation='none')
ax[0].set_axis_off()
ax[1].imshow(Xm, cmap='gray', interpolation='none')
ax[1].set_axis_off()
ax[2].imshow(Xa, cmap='gray', interpolation='none')
ax[2].set_axis_off()
plt.show()
