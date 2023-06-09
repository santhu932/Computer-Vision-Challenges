from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import math
from imageio import imread
import sys

def template_matching(music_im, template, k):
    temp_x = template.shape[1]
    temp_y = template.shape[0]
    im_x = music_im.shape[1]
    im_y = music_im.shape[0]
    cross_correlation_values = np.zeros((im_y - temp_y, im_x - temp_x))
    box_locations = []
    im_array = np.array(music_im, dtype = 'int')
    temp_array = np.array(template, dtype = 'int')
    # Reference for template matching cross correlation formula: https://www.youtube.com/watch?v=1_hwFc8PXVE
    for i in range(im_y - temp_y):
        for j in range(im_x - temp_x):
            
            receptive_field = im_array[i:i+temp_y, j:j+temp_x]
            cross_cor = np.sum((receptive_field * temp_array))
            #Calculation of normalization constant
            image_energy  = np.sum(receptive_field * receptive_field)
            template_energy = np.sum(temp_array * temp_array)
            norm_constant = np.sqrt(image_energy) * np.sqrt(template_energy)
            #Normalized cross correlation value
            if cross_cor == 0 or norm_constant == 0:
                pass
            else:
                cross_cor /= norm_constant
                
            
            cross_correlation_values[i][j] = cross_cor
            
    #Add boxes if they are similar enough
    threshold = np.mean(cross_correlation_values) + (k * np.std(cross_correlation_values))
    for i in range(cross_correlation_values.shape[0]):
        for j in range(cross_correlation_values.shape[1]):
            if cross_correlation_values[i][j] >= threshold :
                box_locations = get_locs(box_locations, j, i, temp_y, cross_correlation_values[i][j])

    return box_locations


#Update the locations of all boxes.
#If a new box is found such that a box close to it has been found with a lower correlation value,
#replcae it.
def get_locs(locs, j, i, y, cross):
    box_locs = []
    found = False
    for loc in locs:
        if (abs(loc[0] - j) <= int(0.75*y)) & (abs(loc[1] - i) <= int(0.75*y)):
            found = True
            if cross > loc[2]:
                box_locs.append([j,i, cross])
            else:
                box_locs.append(loc)
        else:
            box_locs.append(loc)
    
    if not found:
        box_locs.append([j,i, cross])
    return box_locs
    
#Draw all of the boxes for the notes and rests
def draw_bounding_boxes(music_sheet, box_loc, template, color):
    temp_x = template.shape[1]
    temp_y = template.shape[0]
    for l in range(len(box_loc)):
        bottom_right = [0] * 2
        top_left = box_loc[l]
        bottom_right[0] = top_left[0] + temp_x - 1
        bottom_right[1] = top_left[1] + temp_y - 1
        draw_box = ImageDraw.Draw(music_sheet)
        draw_box.rectangle([(top_left[0]-2, top_left[1]-2), (bottom_right[0]+2, bottom_right[1]+2)], fill = None, outline= color, width = 1)
    return music_sheet

#With this method, We iterate over ranges relative to the staffs and assign the most likely label to that
#For example, if the top of the notebox starts close to the highest staff, it is likely to be an E
def draw_note_class(music_sheet, box_loc, staff, template):
    
    if len(staff) % 10 != 0:
        print("WARNING: Unable to find accurate staff locations. Music note classes may be inaccurate!")
    
    space = template.shape[0]
    notes = ["G", "F", "E", "D", "C", "B", "A"]
    staffs = []
    for st in staff:
        staffs.append(st[0] * math.sin(st[1]))
    staffs = staffs[::-1]
    
    top_staff = []
    bot_staff = []
    note_data = []
    for i in range(int(len(staff) / 10)):
        top_staff = staffs[(i*10):(i*10)+5]
        bot_staff = staffs[(i*10)+5:((i+1)*10)]
    
        for box in box_loc:
            label = "?"
            loc = box[1]
            #If the note is closer to the top staffs, find note relative to top staff
            if (abs(loc - top_staff[4]) < abs(loc - bot_staff[0])):
                
                #Declare initial range. Note that this is two staffs above the actual top staff
                range1 = top_staff[0] - int(1/4 * space) - (5*space)
                range2 = top_staff[0] + int(1/4 * space) + 1 - (5*space)
                bar = 0 #Which staff we are closest to
                note = 0 #The note from the notes array that we start on
                distance = 8 #The distance from the starting range to the actual top staff
                for i in range(22):
                    #If we are on a staff, set its range
                    if (i % 2 == 0) and (bar <= 4) and (distance <= 0):
                        range1 = top_staff[bar] - int(3/4 * space)
                        range2 = top_staff[bar] - int(1/4 * space) - 1
                        
                    #If we are in between staffs, set range
                    elif (i % 2 != 0) and (bar <= 4) and (distance <= 0): 
                        range1 = top_staff[bar] - int(1/4 * space)
                        range2 = top_staff[bar] + int(1/4 * space) + 1
                        bar += 1
                      
                    #If we are either above or below staffs
                    else:
                        range1 += int(1/2 * space)
                        range2 += int(1/2 * space)
                        distance -= 1
                      
                    #If the note falls within this range, declare its note
                    if range1 <= loc <= range2:
                        label = notes[note]
                        note_data.append([box[0], box[1], label])
                        break
                    
                    note += 1
                    if note >= 7:
                        note = 0
                        
            #If the note is closer to the bot staffs, find note relative to bot staff
            else:
                #Declare initial range. Note that this is two staffs above the actual top staff
                range1 = bot_staff[0] - int(1/4 * space) - (4*space)
                range2 = bot_staff[0] + int(1/4 * space) + 1 - (4*space)
                bar = 0 #Which staff we are closest to
                note = 0 #The note from the notes array that we start on
                distance = 6 #The distance from the starting range to the actual top staff
                for i in range(22):
                    #If we are on a staff, set its range
                    if (i % 2 == 0) and (bar <= 4) and (distance <= 0):
                        range1 = bot_staff[bar] - int(3/4 * space)
                        range2 = bot_staff[bar] - int(1/4 * space) - 1
                        
                    #If we are in between staffs, set range
                    elif (i % 2 != 0) and (bar <= 4) and (distance <= 0): 
                        range1 = bot_staff[bar] - int(1/4 * space)
                        range2 = bot_staff[bar] + int(1/4 * space) + 1
                        bar += 1
                      
                    #If we are either above or below staffs
                    else:
                        range1 += int(1/2 * space)
                        range2 += int(1/2 * space)
                        distance -= 1
                      
                    #If the note falls within this range, declare its note
                    if range1 <= loc <= range2:
                        label = notes[note]
                        note_data.append([box[0], box[1], label])
                        break
                
                    note += 1
                    if note >= 7:
                        note = 0
                        
            #Draw the note
            draw_let = ImageDraw.Draw(music_sheet)
            draw_let.text((box[0]-12, box[1]-3), label, fill = "red", outline = "red")
    return music_sheet, note_data

#Draw staff lines
def draw_staff_lines(music_sheet, staff):
    h = music_sheet.height
    w = music_sheet.width
    color_im = music_sheet.convert("RGB")
    for stave in staff:
        for x in range(w):
            for y in range(h):
                if stave[0] == round((x*math.cos(stave[1])) + (y*math.sin(stave[1]))):
                    (R,G,B) = (255,0,255)
                    color_im.putpixel((x,y), (R,G,B))
    return color_im

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

    color_im.save("output_red.png")

    return staff

#Create a txt file with all note locations
def print_to_txt(file, notes, quarters, eights, note_temp, quarter_temp, eight_temp):
    output = []
    
    for n in notes:
        output.append(str(n[1]) + " " + str(n[0]) + " " + str(note_temp.shape[0]) 
                      + " "  + str(note_temp.shape[1]) + " filled_note " 
                      + " " + str(n[2]) + " " + "1.0")
        
    for q in quarters:
        output.append(str(q[1]) + " " + str(q[0]) + " " + str(quarter_temp.shape[0]) 
                      + " "  + str(quarter_temp.shape[1]) + " quarter_rest " 
                      + " " + "_" + " " + "1.0")
        
    for e in eights:
        output.append(str(e[1]) + " " + str(e[0]) + " " + str(eight_temp.shape[0]) 
                      + " "  + str(eight_temp.shape[1]) + " eighth_rest " 
                      + " " + "_" + " " + "1.0")
        
    with open(file, 'w') as f:
        f.write('\n'.join(output))

if __name__ == '__main__':
    
    if(len(sys.argv) < 2):
        raise Exception("error: please give an input image name as a parameter, like this: \n"+
                     "python3 staff_finder.py input.jpg")
        
    #Set threshold values
    if (sys.argv[1] == "music4.png"):
        k1 = 3
    else:    
        k1 = 4
    k2 = 2
    k3 = 1.75
    
    #Read in image and templates
    music_im = Image.open(sys.argv[1])
    music_sheet = imread(sys.argv[1], as_gray = True)
    note = imread("template4.png", as_gray = True)
    quarter_rest = imread("template2.png", as_gray = True)
    eight_rest = imread("template3.png", as_gray = True)

    #Find all of the staffs using a modified version of the hough transform from part 1
    staff = hough_transform_part2(music_im)
    
    #Scale templates. Note that for music1, the templates will not change scale
    scale = abs(staff[0][0] - staff[1][0]) / note.shape[0]
    note = np.array(Image.fromarray(note).resize((int(note.shape[1] * scale), int(note.shape[0] * scale))).convert("L"))
    quarter_rest = np.array(Image.fromarray(quarter_rest).resize((int(quarter_rest.shape[1] * scale), int(quarter_rest.shape[0] * scale))).convert("L"))
    eight_rest = np.array(Image.fromarray(eight_rest).resize((int(eight_rest.shape[1] * scale), int(eight_rest.shape[0] * scale))).convert("L"))

    #get all box locations for each thing we are looking for
    note_box_loc = template_matching(music_sheet, note, k1)
    quarter_rest_box_loc = template_matching(music_sheet, quarter_rest, k2)
    eight_rest_box_loc = template_matching(music_sheet, eight_rest, k3)

    #Draw all lines, boxes, and note classes
    music_im = draw_staff_lines(music_im, staff)
    music_im = draw_bounding_boxes(music_im, note_box_loc, note, "red")
    music_im = draw_bounding_boxes(music_im, quarter_rest_box_loc, quarter_rest, "green")
    music_im = draw_bounding_boxes(music_im, eight_rest_box_loc, eight_rest, "blue")
    music_im, note_data = draw_note_class(music_im, note_box_loc, staff, note)
    music_im.save("detected.png")
    
    #Create the detected.txt file with all notes
    print_to_txt("detected.txt", note_data, quarter_rest_box_loc, eight_rest_box_loc, note, quarter_rest, eight_rest)
