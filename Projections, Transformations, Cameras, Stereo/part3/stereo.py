import sys
import math
from PIL import Image
import numpy as np
MAX_COST = math.inf
ALPHA = 10 # Parameter for distance function
NUM_ITERATIONS = 15 #Number of iterations for MRF

#Distance function
def V(d, d1):
    if d==d1:
        return 0
    else:
        return 1

def mrf_stereo(img1, img2, disp_costs, MAX_DISPARITY, w_size):
    #Intializing the messages
    upward_msg = np.zeros((img1.shape[0], img1.shape[1], MAX_DISPARITY))
    downward_msg = np.zeros((img1.shape[0], img1.shape[1], MAX_DISPARITY))
    forward_msg = np.zeros((img1.shape[0], img1.shape[1], MAX_DISPARITY))
    backward_msg = np.zeros((img1.shape[0], img1.shape[1], MAX_DISPARITY))

    d_map = np.zeros((img1.shape[0], img1.shape[1]))

    #Calculating the distance function for all possible disparity pairs
    v_values = np.zeros((MAX_DISPARITY, MAX_DISPARITY))
    for d in range(MAX_DISPARITY):
        for d1 in range(MAX_DISPARITY):
            v_values[d, d1] = V(d, d1)
    v = ALPHA * v_values.reshape((1, 1, MAX_DISPARITY, MAX_DISPARITY))
    
    for t in range(NUM_ITERATIONS):

        #Storing the incoming messages at all nodes
        previous_upward_m = np.concatenate((upward_msg[1:, :], upward_msg[:1, :]), axis=0)
        previous_downward_m = np.concatenate((downward_msg[-1: , :], downward_msg[:-1, :]), axis=0)
        previous_forward_m = np.concatenate((forward_msg[:, -1:], forward_msg[:, :-1]), axis=1)
        previous_backward_m = np.concatenate((backward_msg[:, 1:], backward_msg[:, :1]), axis=1)

        #Calculating the messages
        #Reference for broadcasting: https://numpy.org/doc/stable/user/basics.broadcasting.html
        U_m = disp_costs + previous_forward_m + previous_backward_m + previous_upward_m
        D_m = disp_costs + previous_forward_m + previous_backward_m + previous_downward_m
        F_m = disp_costs + previous_forward_m + previous_upward_m + previous_downward_m
        B_m = disp_costs + previous_backward_m + previous_upward_m + previous_downward_m
        
        #Adding extra dimension in order to broadcast all possible values of V(d, d1)
        U_m = np.expand_dims(U_m, axis=-1)
        D_m = np.expand_dims(D_m, axis=-1)
        F_m = np.expand_dims(F_m, axis=-1)
        B_m = np.expand_dims(B_m, axis=-1)

        upward_msg_t = U_m + v
        downward_msg_t = D_m +  v
        forward_msg_t = F_m +  v
        backward_msg_t = B_m +  v

        #Minimum d1 of the messages
        upward_msg = np.min(upward_msg_t, axis=-2)
        downward_msg = np.min(downward_msg_t, axis=-2)
        forward_msg = np.min(forward_msg_t, axis=-2)
        backward_msg = np.min(backward_msg_t, axis=-2)

        #Normalizing the messages
        upward_msg -= np.mean(upward_msg, axis=2, keepdims=True)
        downward_msg -= np.mean(downward_msg, axis=2, keepdims=True)
        forward_msg -= np.mean(forward_msg, axis=2, keepdims=True)
        backward_msg -= np.mean(backward_msg, axis=2, keepdims=True)
    
    #Storing the incoming messages at all nodes for cost calculation
    previous_upward_m = np.concatenate((upward_msg[1:, :], upward_msg[:1, :]), axis=0)
    previous_downward_m = np.concatenate((downward_msg[-1: , :], downward_msg[:-1, :]), axis=0)
    previous_forward_m = np.concatenate((forward_msg[:, -1:], forward_msg[:, :-1]), axis=1)
    previous_backward_m = np.concatenate((backward_msg[:, 1:], backward_msg[:, :1]), axis=1)

    #Computing the Final cost function
    Total_D_cost = disp_costs + previous_upward_m + previous_downward_m + previous_forward_m + previous_backward_m

    #Finding the disparity that produce minimum cost at each pixel
    d_map = np.argmin(Total_D_cost, axis = 2)

    return d_map * (255/MAX_DISPARITY)  #Scaling the disparity map for better visualization


def disparity_costs(img1, img2, norm_type, MAX_DISPARITY, w_size):
    d_costs = np.zeros((img1.shape[0], img1.shape[1], MAX_DISPARITY))
    #Padding the borders of the images
    img1 = np.pad(img1, [(int(w_size/2),int(w_size/2)),(int(w_size/2),int(w_size/2)), (0,0)], mode = 'edge')
    img2 = np.pad(img2, [(int(w_size/2),int(w_size/2)),(int(w_size/2) + MAX_DISPARITY,int(w_size/2)), (0,0)], mode = 'edge')

    for i in range(int(w_size/2), img1.shape[0] - int(w_size/2)):
        for j in range(int(w_size/2), img1.shape[1] - int(w_size/2)):
            for d in range(MAX_DISPARITY):
                I_L = img1[i - int(w_size/2) : i + int(w_size/2), j - int(w_size/2) : j + int(w_size/2), :]
                I_R = img2[i - int(w_size/2) : i + int(w_size/2), j - int(w_size/2) - d + MAX_DISPARITY : j + int(w_size/2) - d + MAX_DISPARITY, :]
                #Normalizing the D function
                if norm_type == 1:
                    #Standard Normalization
                    norm_constant = img1.shape[0] * img1.shape[1]
                    d_costs[i - int(w_size/2),j - int(w_size/2),d] = np.sum(((I_L.astype(int) - I_R.astype(int)) ** 2)/norm_constant)
                else:
                    #Normalization for Aloe Images for  better smoother disparity map
                    norm_constant = np.sqrt(np.sum(np.abs(I_L * I_L), axis = (0,1))) * np.sqrt(np.sum(np.abs(I_R * I_R), axis = (0,1)))
                    if all(norm_constant) > 0:
                        d_costs[i - int(w_size/2),j - int(w_size/2),d] = np.sum(((I_L.astype(int) - I_R.astype(int)) ** 2)/norm_constant)
    return d_costs


# This function finds the minimum cost at each pixel
def naive_stereo(disp_costs):
    d_map = np.argmin(disp_costs, axis = 2)
    
    return d_map * (255/MAX_DISPARITY) #Scaling the disparity map for better visualization
                      
if __name__ == "__main__":

    if len(sys.argv) != 3 and len(sys.argv) != 4:
        raise Exception("usage: " + sys.argv[0] + " image_file1 image_file2 [gt_file]")
    input_filename1, input_filename2 = sys.argv[1], sys.argv[2]

    # read in images and gt
    image1 = np.array(Image.open(input_filename1))
    image2 = np.array(Image.open(input_filename2))
    
    MAX_DISPARITY = 55 #Maximum disparity values
    w_size = 7 #Window Size
    norm_type = 1 #Normalization type for D function
    
    #Larger Window size for Flowerpots Image to get better disparity map
    if (sys.argv[1] == "Flowerpots/view1.png"):
        w_size = 15
        
    #Reducing Disparity Values for Aloe Image, to reduce the total time execution
    if (sys.argv[1] == "Aloe/view1.png"):
        norm_type = 0
        MAX_DISPARITY = 45
        
    gt = None
    if len(sys.argv) == 4:
        gt = np.array(Image.open(sys.argv[3]))[:,:,0]

        # gt maps are scaled by a factor of 3, undo this...
        gt = gt / 3.0

    # compute the disparity costs (function D_2())
    disp_costs = disparity_costs(image1, image2, norm_type, MAX_DISPARITY, w_size)

    # do stereo using naive technique
    disp1 = naive_stereo(disp_costs)
    Image.fromarray(disp1.astype(np.uint8)).save("output-naive.png")
    
        
    # do stereo using mrf
    disp3 = mrf_stereo(image1, image2, disp_costs, MAX_DISPARITY, w_size)
    Image.fromarray(disp3.astype(np.uint8)).save("output-mrf.png")

    #Measure error with respect to ground truth, if we have it...
    if gt is not None:
        #Descaling the maps for error calculation
        err = np.sum(((disp1 * (MAX_DISPARITY/255))- gt)**2)/gt.shape[0]/gt.shape[1]
        print("Naive stereo technique mean error = " + str(err))

        err = np.sum(((disp3 * (MAX_DISPARITY/255))- gt)**2)/gt.shape[0]/gt.shape[1]
        print("MRF stereo technique mean error = " + str(err))
        
