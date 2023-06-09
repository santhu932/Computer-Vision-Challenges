from scipy import fftpack
from PIL import Image
from PIL import ImageFilter
import imageio
import numpy
import sys

if __name__ == '__main__':


    if(len(sys.argv) < 2):
        raise Exception("error: please give an input image name as a parameter, like this: \n"
                     "python3 pichu_devil.py input.jpg")
    
    # Load an image 
    im = Image.open(sys.argv[1])
    print("Image is %s pixels wide." % im.width)
    print("Image is %s pixels high." % im.height)
    print("Image mode is %s." % im.mode)

    image = imageio.imread(sys.argv[1], as_gray=True)
    
    # We should get the image in the frequency domain
    noisy_fft = fftpack.fftshift(fftpack.fft2(image))
    noisy_fft_n =  (numpy.log(abs(noisy_fft))* 255 /numpy.amax(numpy.log(abs(noisy_fft)))).astype(numpy.uint8)
    
    # let's save the image to visualize it
    imageio.imsave('noisy_pichu_freq.png', noisy_fft_n)

    # Construct the low pass filter 
    # with the help of: https://wsthub.medium.com/python-computer-vision-tutorials-image-fourier-transform-part-3-e65d10be4492

    r = 30 # The threshold, the greater, the lower the frequency passed is
    ham = numpy.hamming(im.width)[:,None] # 1D hamming
    ham2d = numpy.sqrt(numpy.dot(ham, ham.T)) ** r # expand to 2D hamming

    # Now we do the multiplication in the frequency domain
    fil = numpy.multiply(ham2d,noisy_fft)
    fil_n = (numpy.log(abs(fil))* 255 /numpy.amax(numpy.log(abs(fil)))).astype(numpy.uint8)
    
    # let's tansfer it back to the image space
    ifft2 = abs(fftpack.ifft2(fftpack.ifftshift(fil)))

    imageio.imsave('pitchu_filtered.png', ifft2.astype(numpy.uint8))

    # let's try to implement the box filter 
    d = 0.1
    box = numpy.zeros((im.width,im.height))
    for x in range(im.width):
        if x>im.width*(0.5-d) and x<im.width*(0.5+d):
             for y in range(im.height):
                         if y>im.height*(0.5-d) and y<im.height*(0.5+d):
                             box[x,y]=1
    
    fil_box = numpy.multiply(box,noisy_fft)
    # fil_box_n = (numpy.log(abs(fil_box))* 255 /numpy.amax(numpy.log(abs(fil_box)))).astype(numpy.uint8)
    # imageio.imsave('box_freq.png', fil_box_n)

    ifft2_b = abs(fftpack.ifft2(fftpack.ifftshift(fil_box)))
    # imageio.imsave('box.png', ifft2_b.astype(numpy.uint8))

