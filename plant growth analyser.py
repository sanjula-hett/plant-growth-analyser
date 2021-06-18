import cv2
import math
import os
import numpy as np
import matplotlib.pyplot as pltimport 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Frame extraction
def frame_extraction(file):
    video_file = file
    video_cap = cv2.VideoCapture(video_file)
    ret,image = video_cap.read()


    seconds = 1 # Sets how many interval between each frame is captured
    fps = int(video_cap.get(cv2.CAP_PROP_FPS)) # Gets the frames per second
    multiplier = fps * seconds 

    # Retrieves each frame according to the parameters
    while ret:
        frameId = int(round(video_cap.get(1))) # Retrieves the frame number
        ret, image = video_cap.read() 

        if frameId % multiplier == 0: # If the frame number corresponds to the pre-defined time interval between frames 
            cv2.imwrite("frames/%d.png" % frameId, image) # Saves the frame in a sub-folder

    video_cap.release()
    print ("Complete")


# Retrieves list of all the frames that have been saved
def get_list():
    im_list = [] 
    current_dir = os.path.dirname(os.path.realpath(__file__))
    image_file = (current_dir+"/frames")
    for file_name in os.listdir(image_file):
        if file_name.split(".")[-1].lower() in {"png"}:
            im_list.append(int(file_name.split(".",1)[0]))
    return (im_list)

# Counts the number of green pixesls in each frame
def colour_id(im_list):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    image_file = (current_dir+"/frames")
    l_b = np.array([25,21,69]) # Sets the lower bound for the colour green
    u_b = np.array([100,255,255]) # Sets the lower bound for the colour green
    green_pixels = []
    count = 0
    for image in im_list:
        frame = cv2.imread(image_file+"/"+str(image)+".png")

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Converts frame colour to HSV

        mask = cv2.inRange(hsv, l_b, u_b) # Applies a mask for only the green colors 

        res = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imwrite("masked_frames/frame%d.jpg" % count, res)
        green_pixels.append(np.count_nonzero(res)) # Counts number of green pixels
        count+=1
    return(green_pixels)


# Plots the graph of number of green pixels against time
def plot_graph(im_list,pix_count):
    print(len(im_list),len(pix_count))
    x = np.array(im_list)
    y = np.array(pix_count)
    plt.plot(x,y)
    plt.show()



    
frame_extraction("video.mp4")
im_list = get_list()
list1 = sorted(im_list)
pix_count = colour_id(list1)
plot_graph(list1,pix_count)
