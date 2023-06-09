from scipy import fftpack
import imageio
import numpy
import sys

if __name__ == '__main__':
    if(len(sys.argv) < 2):
        raise Exception("error: please give an input image name as a parameter, like this: \n"
                     "python3 fourier.py input.jpg")
    
    # load in an image, convert to grayscale if needed
    image = imageio.imread(sys.argv[1], as_gray=True)

    # take the fourier transform of the image
    fft2 = fftpack.fftshift(fftpack.fft2(image))

    # save FFT to a file. To help with visualization, we take
    # the log of the magnitudes, and then stretch them so they
    # fill the whole range of pixel values from 0 to 255.
    imageio.imsave('fft.png', (numpy.log(abs(fft2))* 255 /numpy.amax(numpy.log(abs(fft2)))).astype(numpy.uint8))

    # At this point, fft2 is just a numpy array and you can
    # modify it in order to modify the image in the frequency
    # space. Here's a little example (that makes a nearly 
    # imperceptible change, but demonstrates what you can do.

    fft2[1,1]=fft2[1,1]+1

    # now take the inverse transform to convert back to an image
    ifft2 = abs(fftpack.ifft2(fftpack.ifftshift(fft2)))

    # and save the image
    imageio.imsave('fft-then-ifft.png', ifft2.astype(numpy.uint8))
