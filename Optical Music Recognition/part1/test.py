import numpy as np
from PIL import Image
from PIL import ImageFilter
import math
import time

def trans():
    image = Image.open("rach.png")
    start = time.time()
    h = image.height
    w = image.width
    img = image.filter(ImageFilter.MinFilter(3))
    img = img.filter(ImageFilter.SMOOTH)
    img = img.convert("L")
    threshold = 150
    img = img.point(lambda i: i < threshold and 255)
    img.save("edges.png")

    d = math.ceil(math.sqrt((h**2) + (w**2)))
    r = np.arange(-d,d+1)

    theta_lim = 90
    theta = np.radians(np.arange(-theta_lim,theta_lim+1))
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x_coord, y_coord = np.nonzero(np.array(img))
    x_coord = x_coord.tolist()
    y_coord = y_coord.tolist()

    accumulator = np.zeros((len(r), len(theta)), dtype=int)

    theta_indexes = np.arange(len(theta), dtype=int)
    for pixel in range(len(x_coord)):
        x = x_coord[pixel]
        y = y_coord[pixel]
        r_pixel = np.around((x*cos_theta) + (y*sin_theta)).astype(dtype=int) + d
        np.add.at(accumulator, (r_pixel, theta_indexes), 1)

    print(np.unravel_index(np.amax(accumulator), accumulator.shape))
    print(np.amax(accumulator))
    print(accumulator.tolist())
    end = time.time()
    print(end-start)

if __name__ == '__main__':
    trans()