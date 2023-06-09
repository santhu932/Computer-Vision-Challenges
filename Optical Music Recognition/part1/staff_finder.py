from PIL import Image
from PIL import ImageFilter
import sys
import numpy as np
import math

#Generates hough space for an input
def gen_hough_space(img):
    h = img.height
    w = img.width

    diagonal_len = math.ceil(math.sqrt((h**2) + (w**2)))
    r = {}
    index = 0
    for i in range(-diagonal_len,diagonal_len+1):
        r[i] = index
        index += 1

    theta = []
    cos_theta = []
    sin_theta = []
    theta_lim = 90
    for i in range(-theta_lim,theta_lim+1):
        rad = math.radians(i)
        theta.append(rad)
        cos_theta.append(math.cos(rad))
        sin_theta.append(math.sin(rad))

    accumulator = [[0]*len(theta) for i in range(len(r))]

    for x in range(w):
        for y in range(h):
            if img.getpixel((x,y)) == 255:
                for i in range(len(theta)):
                    r_calc = round((x*cos_theta[i]) + (y*sin_theta[i]))
                    if r_calc in r:
                        r_index = r[r_calc]
                        accumulator[r_index][i] = 255

    max_votes = 0
    for row in range(len(accumulator)):
        col_max = max(accumulator[row])
        if col_max > max_votes:
            max_votes = col_max

    intensity = 255/max_votes

    for row in range(len(accumulator)):
        for col in range(len(accumulator[0])):
            accumulator[row][col] = math.floor(intensity * accumulator[row][col])

    array = np.uint8(np.asarray(accumulator))
    hough_space = Image.fromarray(array, 'L')
    hough_space.show()

#Creates sample image with one line to understand how hough space is generated
def hough_space_sample():
    #Sample Image with line
    w = 200
    h = 140
    sample = Image.new("L", (w, h))
    x = 30
    y = 50
    for i in range(60):
        sample.putpixel((x+i,y+i),255)
    sample.show()

    #Hough space for sample image with 1 line
    gen_hough_space(sample)

#Detects one staff in the image
def hough_transform(image, num):
    h = image.height
    w = image.width

    #Edge Detection Using Pillow: https://stackoverflow.com/questions/9319767/image-outline-using-python-pil
    img = image.filter(ImageFilter.MinFilter(3))
    img = img.filter(ImageFilter.SMOOTH)
    img = img.convert("L")
    threshold = 150
    img = img.point(lambda i: i < threshold and 255)
    #Edge Detection Complete

    #Calculate diagonal length to get range of r (row) values
    diagonal_len = math.ceil(math.sqrt((h**2) + (w**2)))
    r = {}
    r_vals = []
    index = 0
    for i in range(-diagonal_len,diagonal_len+1):
        r[i] = index
        r_vals.append(i)
        index += 1

    #Calculate sin and cos for values of theta in range -90 to 90 degrees
    theta = []
    cos_theta = []
    sin_theta = [] 
    theta_lim = 90
    for i in range(-theta_lim,theta_lim+1):
        rad = math.radians(i)
        theta.append(rad)
        cos_theta.append(math.cos(rad))
        sin_theta.append(math.sin(rad))

    #Create accumulator with 0 values
    accumulator = [[0]*len(theta) for i in range(len(r))]

    #Vote for each pixel in the accumulator for each value of theta
    for x in range(w):
        for y in range(h):
            if img.getpixel((x,y)) > 0:
                for i in range(len(theta)):
                    r_calc = round((x*cos_theta[i]) + (y*sin_theta[i]))
                    if r_calc in r:
                        r_index = r[r_calc]
                        accumulator[r_index][i] += 1

    #Find the r, theta (best line) with the maximum votes
    max_votes = 0
    best_r_index = 0
    best_theta_index = 0
    for row in range(len(accumulator)):
        for col in range(len(accumulator[0])):
            if accumulator[row][col] > max_votes:
                max_votes = accumulator[row][col]
                best_r_index = row
                best_theta_index = col

    #Get the column associated with theta which had max votes
    #Here since the lines are parallel, they will all be on the same theta,
    #as that of the best line
    best_theta_col = []
    for row in range(len(accumulator)):
        best_theta_col.append(accumulator[row][best_theta_index])

    #Avoid pixels around the best line
    nearest_seen = set()
    nearest_seen.add(best_r_index)
    nearest_seen.add(best_r_index-1)
    nearest_seen.add(best_r_index-2)
    nearest_seen.add(best_r_index+1)
    nearest_seen.add(best_r_index+2)

    #Search for second best line in the same theta column close to the best line (within 20 pixels)
    r_range_lower = best_r_index - 20
    if r_range_lower < 0:
        r_range_lower = 0
    r_range_higher = best_r_index + 21
    if r_range_higher >= len(best_theta_col):
        r_range_higher = len(best_theta_col)

    second_best_r_index = 0
    second_best_max_votes = 0
    for i in range(r_range_lower, r_range_higher):
        if best_theta_col[i] > second_best_max_votes and not i in nearest_seen:
            second_best_max_votes = best_theta_col[i]
            second_best_r_index = i

    #Using the 2 lines find the distance between them (Since lines are equidistant)
    r_range_lower = best_r_index - abs(best_r_index-second_best_r_index)*5
    if r_range_lower < 0:
        r_range_lower = 0
    r_range_higher = best_r_index + abs(best_r_index-second_best_r_index)*5 + 1
    if r_range_higher >= len(best_theta_col):
        r_range_higher = len(best_theta_col)

    #Find the 5 best lines which are parallel (same theta) and equidistant,
    #Based on the distance we calculated before
    seen = set()
    staff = []
    limit = num*5
    for count in range(limit):
        best_r_index_temp = 0
        max_votes = 0
        for i in range(r_range_lower, r_range_higher):
            if best_theta_col[i] > max_votes and not i in seen:
                max_votes = best_theta_col[i]
                best_r_index_temp = i
        seen.add(best_r_index_temp)
        seen.add(best_r_index_temp+1)
        seen.add(best_r_index_temp-1)
        seen.add(best_r_index_temp+2)
        seen.add(best_r_index_temp-2)

        staff.append((best_r_index_temp,best_theta_index))

    #Once we have 5 pairs of r, theta for the 5 lines, highlight them
    color_im = image.convert("RGB")
    for stave in staff:
        for x in range(w):
            for y in range(h):
                if r_vals[stave[0]] == round((x*cos_theta[stave[1]]) + (y*sin_theta[stave[1]])):
                    (R,G,B) = (255,0,0)
                    color_im.putpixel((x,y), (R,G,B))

    color_im.save("detected_staff.png")

#Modified hough transform code for line detection in part 2
def hough_transform_part2(image):
    h = image.height
    w = image.width

    #Edge Detection Using Pillow: https://stackoverflow.com/questions/9319767/image-outline-using-python-pil
    img = image.convert("L")
    img = img.filter(ImageFilter.MinFilter(3))
    img = img.filter(ImageFilter.SMOOTH)
    threshold = 150
    img = img.point(lambda i: 255 if i < threshold else 0)
    img.save("edges.png")
    #Edge Detection Complete

    #Calculate diagonal length to get range of r (row) values
    diagonal_len = math.ceil(math.sqrt((h**2) + (w**2)))
    r = {}
    r_vals = []
    index = 0
    for i in range(-diagonal_len,diagonal_len+1):
        r[i] = index
        r_vals.append(i)
        index += 1

    #Calculate sin and cos for values of theta in range -90 to 90 degrees
    theta = []
    cos_theta = []
    sin_theta = [] 
    theta_lim = 90
    for i in range(-theta_lim,theta_lim+1):
        rad = math.radians(i)
        theta.append(rad)
        cos_theta.append(math.cos(rad))
        sin_theta.append(math.sin(rad))

    #Create accumulator with 0 values
    accumulator = [[0]*len(theta) for i in range(len(r))]

    #Vote for each pixel in the accumulator for each value of theta
    for x in range(w):
        for y in range(h):
            if img.getpixel((x,y)) > 0:
                for i in range(len(theta)):
                    r_calc = round((x*cos_theta[i]) + (y*sin_theta[i]))
                    if r_calc in r:
                        r_index = r[r_calc]
                        accumulator[r_index][i] += 1

    #Find the r, theta (best line) with the maximum votes
    top_max_votes = 0
    best_r_index = 0
    best_theta_index = 0
    for row in range(len(accumulator)):
        for col in range(len(accumulator[0])):
            if accumulator[row][col] > top_max_votes:
                top_max_votes = accumulator[row][col]
                best_r_index = row
                best_theta_index = col

    #Get the column associated with theta which had max votes
    #Here since the lines are parallel, they will all be on the same theta,
    #as that of the best line
    best_theta_col = []
    for row in range(len(accumulator)):
        best_theta_col.append(accumulator[row][best_theta_index])

    #Avoid pixels around the best line
    nearest_seen = set()
    nearest_seen.add(best_r_index)
    nearest_seen.add(best_r_index-1)
    nearest_seen.add(best_r_index-2)
    nearest_seen.add(best_r_index+1)
    nearest_seen.add(best_r_index+2)

    #Search for second best line in the same theta column close to the best line (within 20 pixels)
    r_range_lower = best_r_index - 20
    if r_range_lower < 0:
        r_range_lower = 0
    r_range_higher = best_r_index + 21
    if r_range_higher >= len(best_theta_col):
        r_range_higher = len(best_theta_col)

    second_best_r_index = 0
    second_best_max_votes = 0
    for i in range(r_range_lower, r_range_higher):
        if best_theta_col[i] > second_best_max_votes and not i in nearest_seen:
            second_best_max_votes = best_theta_col[i]
            second_best_r_index = i

    #Using the 2 lines find the distance between them (Since lines are equidistant)
    distance = abs(best_r_index - second_best_r_index)

    #Find all the best lines which are parallel (same theta) and equidistant,
    #Based on the distance we calculated before
    seen = set()
    staff = []
    for i in range(len(best_theta_col)):
        if best_theta_col[i] > top_max_votes*0.7 and not i in seen:
            for j in range(i,i+distance-1):
                seen.add(j)
            staff.append((r_vals[i],theta[best_theta_index]))

    #Once we have all pairs of r, theta for the lines, highlight them
    color_im = image.convert("RGB")
    for stave in staff:
        for x in range(w):
            for y in range(h):
                if stave[0] == round((x*math.cos(stave[1])) + (y*math.sin(stave[1]))):
                    (R,G,B) = (255,0,0)
                    color_im.putpixel((x,y), (R,G,B))

    color_im.save("part2_output_red.png")

    return staff

#Given a line r, theta this function checks whether a point x,y is on it or not
def check_point_line(x, y, r, theta):
    if r == round((x*math.cos(theta)) + (y*math.sin(theta))):
        return True

if __name__ == '__main__':

    if(len(sys.argv) < 2):
        raise Exception("error: please give an input image name as a parameter, like this: \n"
                     "python3 staff_finder.py input.jpg")

    image_name = sys.argv[1]
    image = Image.open(image_name)

    hough_transform(image, 1)