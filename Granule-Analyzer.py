# import neccessary libraries providing functions for the following script
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import xlsxwriter as xw
from pandas import DataFrame
from seaborn import histplot
from PIL import ImageTk, Image
from scipy.stats import shapiro
from tkinter import filedialog

# file dialog window base as the very first GUI after starting the program
file_root = tk.Tk()
file_root.title("Granule Analyzer")
file_root.geometry("525x530")
file_root.resizable(False, False)
img_front = ImageTk.PhotoImage(Image.open("front_pic.jpeg"))
label_name = tk.Label(file_root, text= '"Granule Analyzer" developed by Markus Pieth')
label_name.grid(row= 0, column= 0, columnspan= 3)
label_front = tk.Label(file_root, image= img_front)
label_front.grid(row= 1, column= 0, columnspan= 3)

# function for second window after having chosen an image file for processing
def open_file():
# set up global image variables
    global image_orig
    global image_process
    global image_show
    global file_path
    global choice
# file opening and reading in an image passing it to set up variables
    file_path = filedialog.askopenfilename(initialdir= "/", title= "Select an image for processing!", filetypes= [("All files", "*.*")])
    image_orig = cv.imread(file_path)
    image_process = cv.imread(file_path, cv.IMREAD_GRAYSCALE) 
    image_show = ImageTk.PhotoImage(Image.open(file_path).resize((717,545)))
    choice = tk.StringVar(file_root)
# rearranging GUI window with image and file-path
    label_front.destroy()
    label_open.destroy()
    label_name.destroy()
    label_path = tk.Label(file_root, text= f"Selected file: {file_path.split('/')[-1]}")
    label_path.grid(row= 0, column= 0, columnspan= 4)  
    global label_img_open  
    label_img_open = tk.Label(file_root, image= image_show, justify= "center")
    label_img_open.grid(row= 1, column= 0,columnspan= 4)
# creating label widgets for description
    label_barlength= tk.Label(file_root, text= "Enter reference in micrometer:", justify= "right")
    label_barlength.grid(row= 2, column= 0, pady= 5)
    label_barpxl = tk.Label(file_root, text="Enter reference in pixel:", justify= "right")
    label_barpxl.grid(row= 2, column= 2, pady= 5)
    label_topcut = tk.Label(file_root, text= "cut top %:")
    label_topcut.grid(row= 3, column= 0,pady= 10)
    label_bottomcut = tk.Label(file_root, text= "cut bottom %:")
    label_bottomcut.grid(row= 3, column= 1, pady= 10)
    label_leftcut = tk.Label(file_root, text= "cut left %:")
    label_leftcut.grid(row= 3, column= 2, pady= 10)
    label_rightcut = tk.Label(file_root, text= "cut right %:")
    label_rightcut.grid(row= 3, column= 3,pady= 10)
    label_choice = tk.Label(file_root, text= "image mode:")
    label_choice.grid(row= 5, column= 0, pady= 10)
# setting input widgets for processing parameter
    entry_barlength = tk.Entry(file_root, width= 5)
    entry_barlength.grid(row= 2, column= 1, pady= 5)
    entry_barpxl = tk.Entry(file_root, width= 5)
    entry_barpxl.grid(row= 2, column= 3, pady= 5)
    entry_topcut = tk.Entry(file_root, width= 5)
    entry_topcut.grid(row= 4, column= 0)
    entry_bottomcut = tk.Entry(file_root, width= 5)
    entry_bottomcut.grid(row= 4, column= 1)
    entry_leftcut = tk.Entry(file_root, width= 5)
    entry_leftcut.grid(row= 4, column= 2)
    entry_rightcut = tk.Entry(file_root, width= 5)
    entry_rightcut.grid(row= 4, column= 3)
# radiobuttons for image mode
    radio_bright = tk.Radiobutton(file_root, text= "brightened", variable= choice, value= "bright")
    radio_bright.grid(row= 5, column= 1, pady= 15)
    radio_invert = tk.Radiobutton(file_root, text= "inverted", variable= choice, value= "invert")
    radio_invert.grid(row= 5, column= 2, pady= 15)
# buttons for calling funtions
    button_dist_check = tk.Button(file_root, text= "Intensity distribution", command= lambda: pixel_distribution())
    button_dist_check.grid(row= 5, column= 3, pady= 15)
    button_open.grid(row= 6, column= 0, pady= 5)
    button_process = tk.Button(file_root, text= "Process image", command= lambda:process_img(conv_factor= int(entry_barlength.get())/int(entry_barpxl.get())))
    button_process.grid(row= 6, column= 1, pady= 5)
    button_cutout = tk.Button(file_root, text= "Cut out area", 
                              command= lambda: [reset_global_images(int(entry_leftcut.get()), int(entry_topcut.get()), int(entry_rightcut.get()), int(entry_bottomcut.get()), file_path)])
    button_cutout.grid(row= 6, column= 2, pady= 5)
    button_help = tk.Button(file_root, text= "Help", command= open_manual_1)
    button_help.grid(row= 6, column= 3, pady= 5)
# fitting window size accordingly to added widgets
    file_root.geometry(str(717+5) + "x" + str(545+215))
    file_root.resizable(False, False)

# function for showing a histogram of the pixel intensities in the image file to be processed
def pixel_distribution():
    plt.hist(image_process.ravel(),256,[0,256])
    plt.title(f'Distribution of pixel intensities in "{file_path.split('/')[-1]}"')
    plt.xlabel("intensity values from black to white")
    plt.ylabel("count")
    plt.show()

# function for image processing and initial statistical evaluation
def process_img(conv_factor):
# apply bilateral blurring to smooth and contain edges, eliminating a bit noise
    blurred_img = cv.bilateralFilter(image_process, 10, 20, 20)
# switching algorithm for bright or dark images
    if str(choice.get()) == 'bright':
# Canny-edge detection
        canny_img = cv.Canny(blurred_img, 50, 255)
    elif str(choice.get()) == 'invert':
# inversion of blurred image
        invert_img = cv.bitwise_not(blurred_img)
# Canny-edge detection
        canny_img = cv.Canny(invert_img, 50, 255)
# defining conturs and refining from binarised image
    contours, hierarchy = cv.findContours(canny_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# looping throug all found contours, creating list of measures from contours
    global blank_included
    blank_included = image_orig.copy()
    areas_list = []
    diameters_list = []
    lengths_list = []
    widths_list = []
    shape_set = set()
    for i, cnt in enumerate(contours):
        area = round(cv.contourArea(cnt),1)
        areas_list.append(float(area))
        diameter = round(2*np.sqrt(area/np.pi)*conv_factor,1)
        diameters_list.append(float(diameter))
        length = round(np.max(cv.minAreaRect(cnt)[1])*conv_factor,1)
        lengths_list.append(float(length))
        width = round(np.min(cv.minAreaRect(cnt)[1])*conv_factor,1)
        widths_list.append(float(width))
        shape_set.add(hierarchy[0][i][3])
# divide list of desired measure into five equal chunks
    start = 0
    end = len(diameters_list)
    step = round(len(diameters_list)/5)
    parts = []
    for i in range(start, end, step):
        parts.append(list(sorted(diameters_list)[i:(i+step)]))
# finding best part-intervall combination regarding normal distribution
    p_value = 1
    best_intervall = []
    for j in range(len(parts)):
        intervall_combo = []
        for k in range(len(parts)-j):
            intervall_combo += parts[k+j]
            p = shapiro(intervall_combo)[1]
            if p < p_value:
                p_value = p
                best_intervall = intervall_combo
# overlaying included contours belonging to the best_intervall on image_orig
    for l in range(len(diameters_list)):
        if diameters_list[l] >= min(best_intervall) and diameters_list[l] <= max(best_intervall):
            cv.drawContours(blank_included, contours[l], -1, (0,255,255), 2)
# transforming array into png-file
    cv.imwrite("blank_included.png", blank_included)
    image_contours = "blank_included.png"
# calling function for refinement parameter setting, opens new window
    refine_window_new(image_contours, best_intervall, lengths_list, widths_list, 20, conv_factor, shape_set)

# threshold finder tool function
def threshold_finder():
    def callback(input):
        pass
# resizing function for the cv-window below
    def rescaleFrame(frame, scale = 0.7): 
        width = int(frame.shape[1] * scale) 
        height = int(frame.shape[0] * scale) 
        dimensions = (width, height) 
        return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
# cv-windows for depicting and setting
    winname = file_path.split('/')[-1]
    cv.namedWindow(winname)
    cv.createTrackbar('Min-Thresh', winname, 0, 255, callback)
    cv.createTrackbar('Max-Thresh', winname, 0, 255, callback)
# feedback loop depicting the important edges depending on threshold setting
    while True:
        if cv.waitKey(1) == ord('q'):
            break
        thresh_min = cv.getTrackbarPos('Min-Thresh', winname)
        thresh_max = cv.getTrackbarPos('Max-Thresh', winname)
        blurred_img = cv.bilateralFilter(image_process, 10, 20, 20)
        if str(choice.get()) == 'invert':
            invert_img = cv.bitwise_not(blurred_img)
            cannyEdge = cv.Canny(invert_img, thresh_min, thresh_max)
        elif str(choice.get()) == 'bright':
            cannyEdge = cv.Canny(blurred_img, thresh_min, thresh_max)
        cannyEdge_scaled = rescaleFrame(cannyEdge, scale= 0.4)
        cv.imshow(winname, cannyEdge_scaled)    
    cv.destroyAllWindows()

# function to open a new window for setting refinement parameters
def refine_window_new(image, diameters, lengths, widths, num_bins, conversion, shapes):
    global img
# calcultating descriptive statistics from list values
    count = len(shapes)
    mini = min(diameters)
    maxi = max(diameters)
    std = round(np.std(diameters), 2)
    mean = round(np.mean(diameters), 2)
    median = round(np.median(diameters))
# set up grid for subplots
    axes = plt.subplots(2,2)[1]
# distribution plot for diameter
    histplot(data= diameters, bins= num_bins, kde= True, ax=axes[0,0])
    axes[0,0].axvline(x= mean, linestyle= "solid", color= "k")
    axes[0,0].axvline(x= median, linestyle= "dotted", color= "k") 
    axes[0,0].set_ylabel("count")
    axes[0,0].set_xlabel("avg. diameter in µm")
# distribution plot for length and width combined
    histplot(data= [lengths,widths], bins= num_bins, kde= True, ax=axes[0,1], legend= False)
    axes[0,1].axvline(x= round(np.mean(lengths),2), linestyle= "solid", color= "blue")
    axes[0,1].axvline(x= round(np.median(lengths)), linestyle= "dotted", color= "blue")
    axes[0,1].axvline(x= round(np.mean(widths),2), linestyle= "solid", color= "orange")
    axes[0,1].axvline(x= round(np.median(widths)), linestyle= "dotted", color= "orange") 
    axes[0,1].set_ylabel("count")
    axes[0,1].set_xlabel("width & length in µm")
# distribution plot for lengths
    histplot(data= lengths, bins= num_bins, kde= True, ax=axes[1,0])
    axes[1,0].axvline(x= round(np.mean(lengths),2), linestyle= "solid", color= "k")
    axes[1,0].axvline(x= round(np.median(lengths)), linestyle= "dotted", color= "k") 
    axes[1,0].set_ylabel("count")
    axes[1,0].set_xlabel("length in µm")
# distribution plot for widths
    histplot(data= widths, bins= num_bins, kde= True, ax=axes[1,1])
    axes[1,1].axvline(x= round(np.mean(widths),2), linestyle= "solid", color= "k")
    axes[1,1].axvline(x= round(np.median(widths)), linestyle= "dotted", color= "k") 
    axes[1,1].set_ylabel("count")
    axes[1,1].set_xlabel("width in µm")
# set up a new window base
    global refine_root
    refine_root = tk.Tk()
    refine_root.title("Refine the choosen contours!")
    refine_root.geometry("415x300")
    refine_root.resizable(False, False)
# input widgets for setting the refinement parameters
    global entry_min
    entry_min = tk.Entry(refine_root, width= 5)
    entry_min.grid(row= 4, column= 1, pady= 10, padx= 10)
    global entry_max
    entry_max = tk.Entry(refine_root, width= 5)
    entry_max.grid(row= 4, column= 3, pady= 10, padx= 10)    
    global entry_min_threshold
    entry_min_threshold = tk.Entry(refine_root, width= 5)
    entry_min_threshold.grid(row= 5, column= 1, pady= 10, padx= 10)
    global entry_max_threshold
    entry_max_threshold = tk.Entry(refine_root, width= 5)
    entry_max_threshold.grid(row= 5, column= 3, pady= 10, padx= 10)
    entry_bins= tk.Entry(refine_root, width= 5)
    entry_bins.grid(row= 6, column= 1, pady= 10, padx= 10)
    entry_wlratio_max = tk.Entry(refine_root, width= 5)
    entry_wlratio_max.grid(row= 6, column= 3, pady= 10, padx= 10)
# label widgets for output and description
    global label_min
    label_min = tk.Label(refine_root, text= f"Minimum: {mini} µm")
    label_min.grid(row= 0, column= 0, columnspan= 2, pady= 5)
    global label_max
    label_max = tk.Label(refine_root, text= f"Maximum: {maxi} µm")
    label_max.grid(row= 0, column= 2, columnspan= 2, pady= 5)
    global label_mean
    label_mean = tk.Label(refine_root, text= f"Mean: {mean} µm")
    label_mean.grid(row= 1, column= 0, columnspan= 2, pady= 5)
    global label_median
    label_median = tk.Label(refine_root, text= f"Median: {median} µm")
    label_median.grid(row= 1, column= 2, columnspan= 2, pady= 5)
    global label_std
    label_std = tk.Label(refine_root, text= f"Std: +/- {std} µm")
    label_std.grid(row= 2, column= 0, columnspan= 2, pady= 5)
    global label_count
    label_count = tk.Label(refine_root, text= f"Count: {count}")
    label_count.grid(row= 2, column= 2, columnspan= 2, pady= 5)
    global label_min_threshold
    label_min_threshold = tk.Label(refine_root, text= "Lower Threshold: 50")
    label_min_threshold.grid(row= 3, column= 0, columnspan= 2, pady= 5)
    global label_max_threshold
    label_max_threshold = tk.Label(refine_root, text= "Upper Treshold: 255")
    label_max_threshold.grid(row= 3, column= 2, columnspan= 2, pady= 5)
    label_min_set = tk.Label(refine_root, text= "Set Minimum size")
    label_min_set.grid(row= 4, column= 0, pady= 10)
    label_max_set = tk.Label(refine_root, text= "Set Maximum size")
    label_max_set.grid(row= 4, column= 2, pady= 10)
    label_min_thresh_set = tk.Label(refine_root, text= "Set lower contrast threshold")
    label_min_thresh_set.grid(row= 5, column= 0, pady= 10)
    label_max_thresh_set = tk.Label(refine_root, text= "Set upper contrast threshold")
    label_max_thresh_set.grid(row= 5, column= 2, pady= 10)
    label_bin_set = tk.Label(refine_root, text= "Set number of bins")
    label_bin_set.grid(row= 6, column= 0, pady= 10)
    label_wlratio_max = tk.Label(refine_root, text= "Set max. L/W-Ratio")
    label_wlratio_max.grid(row= 6, column= 2, pady= 10)
# button widgets for calling different image manipulation functions
    button_thresh_find = tk.Button(refine_root, text= 'Threshold tool', command= lambda: threshold_finder())
    button_thresh_find.grid(row= 7, column= 0, pady= 10)
    global button_refine
    button_refine = tk.Button(refine_root, text= "Refine", 
                              command= lambda:refine_img(image_process, float(entry_min.get()), float(entry_max.get()), 
                                                        int(entry_bins.get()), conversion, float(entry_wlratio_max.get())))
    button_refine.grid(row= 7, column= 1, pady= 20)
    global button_excel
    button_excel = tk.Button(refine_root, text= "Create Excel", 
                               command= lambda:create_excel(diameters, lengths, widths, num_bins, float(entry_min.get()), float(entry_max.get()), file_path, image))
    button_excel.grid(row= 7, column= 2, pady= 20)
    button_help = tk.Button(refine_root, text= "Help", command= open_manual_2)
    button_help.grid(row= 7, column= 3, pady= 20)
# additional window for showing the distribution diagrams and image_orig with contour overlay
    global window_2
    window_2 = tk.Toplevel()
    window_2.resizable(False, False)
    window_2.title(f"From: {mini} µm  to: {maxi} µm")
    img = Image.open(image).resize((932,708))
    img_photo = ImageTk.PhotoImage(img)
    global label_img
    label_img = tk.Label(window_2, image= img_photo)
    label_img.grid(row= 0, column= 0)
    plt.show()
# starting the eventloop of refine_window
    refine_root.mainloop()

# function to update and overwrite refine window 
def refine_window_ow(image, diameters, lengths, widths, num_bins, min_thresh, max_thresh, shapes):
# calcultating descriptive statistics from list values
    count = len(shapes)
    mini = min(diameters)
    maxi = max(diameters)
    std = round(np.std(diameters), 2)
    mean = round(np.mean(diameters), 2)
    median = round(np.median(diameters))
# set up grid for subplots
    plt.close()
    axes = plt.subplots(2,2)[1]
# distribution plot for diameter
    histplot(data= diameters, bins= num_bins, kde= True, ax=axes[0,0])
    axes[0,0].axvline(x= mean, linestyle= "solid", color= "k")
    axes[0,0].axvline(x= median, linestyle= "dotted", color= "k") 
    axes[0,0].set_ylabel("count")
    axes[0,0].set_xlabel("diameter in µm")
# distribution plot for length and width combined
    histplot(data= [lengths,widths], bins= num_bins, kde= True, ax=axes[0,1], legend= False)
    axes[0,1].axvline(x= round(np.mean(lengths),2), linestyle= "solid", color= "blue")
    axes[0,1].axvline(x= round(np.median(lengths)), linestyle= "dotted", color= "blue")
    axes[0,1].axvline(x= round(np.mean(widths),2), linestyle= "solid", color= "orange")
    axes[0,1].axvline(x= round(np.median(widths)), linestyle= "dotted", color= "orange") 
    axes[0,1].set_ylabel("count")
    axes[0,1].set_xlabel("width & length in µm")
# distribution plot for lengths
    histplot(data= lengths, bins= num_bins, kde= True, ax=axes[1,0])
    axes[1,0].axvline(x= round(np.mean(lengths),2), linestyle= "solid", color= "k")
    axes[1,0].axvline(x= round(np.median(lengths)), linestyle= "dotted", color= "k") 
    axes[1,0].set_ylabel("count")
    axes[1,0].set_xlabel("length in µm")
# distribution plot for widths
    histplot(data= widths, bins= num_bins, kde= True, ax=axes[1,1])
    axes[1,1].axvline(x= round(np.mean(widths),2), linestyle= "solid", color= "k")
    axes[1,1].axvline(x= round(np.median(widths)), linestyle= "dotted", color= "k") 
    axes[1,1].set_ylabel("count")
    axes[1,1].set_xlabel("width in µm")
# update the output and input widgets
    label_min.config(text= f"Minimum: {mini} µm")
    label_max.config(text= f"Maximum: {maxi} µm")
    label_mean.config(text= f"Mean: {mean} µm")
    label_median.config(text= f"Median: {median} µm")
    label_std.config(text= f"Std: +/- {std} µm")
    label_count.config(text= f"Count: {count}")
    if min_thresh > 0 and max_thresh > 0:
        label_min_threshold.config(text= f"Lower Threshold: {min_thresh}")
        label_max_threshold.config(text= f"Upper Threshold: {max_thresh}")
# feeding new image to additional window
    window_2.title(f"From: {mini} µm to: {maxi} µm")
    img = (Image.open(image).resize((932,708)))
    img_photo = ImageTk.PhotoImage(img)
    label_img.config(image= img_photo)
    button_excel.config(command= lambda:create_excel(diameters, lengths, widths, num_bins, float(entry_min.get()), float(entry_max.get()), file_path, image))
    plt.show()
# restart mainloop after configurations
    refine_root.mainloop()

# function for reprocessing image with refined parameters  
def refine_img(image, minimum, maximum, num_bins, conversion, max_lw_ratio):
# apply bilateral blurring to smooth and contain edges, eliminating a bit noise
    blurred_img = cv.bilateralFilter(image, 10, 20, 20)
# binarisation of pixel colours by using thresholding, resulting in black-white-image
    min_thresh = int(entry_min_threshold.get())
    max_thresh = int(entry_max_threshold.get())
# switching algorithm for bright or dark images
    if str(choice.get()) == 'bright':
# Canny-edge detection
        canny_img = cv.Canny(blurred_img, min_thresh, max_thresh)
    elif str(choice.get()) == 'invert':
# inversion of blurred image
        invert_img = cv.bitwise_not(blurred_img)
# Canny-edge detection
        canny_img = cv.Canny(invert_img, min_thresh, max_thresh)
# defining conturs and refining from binarised image
    contours, hierarchy = cv.findContours(canny_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# looping throug all found contours, creating images from included and excluded contours
    global blank_included
    blank_included = image_orig.copy()
    areas_list = []
    diameters_list = []
    lengths_list = []
    widths_list = []
    shapes = []
    for i, cnt in enumerate(contours):
        area = round(cv.contourArea(cnt),1)
        length = round(np.max(cv.minAreaRect(cnt)[1])*conversion,1)
        width = round(np.min(cv.minAreaRect(cnt)[1])*conversion,1)
        diameter = round((length+width)/2, 1)
        parent = hierarchy[0][i][3]
# defining filtering conditions depending on the refinement parameters put in
        filter_diameter = (minimum <= width and length <= maximum)
        filter_lw_ratio = (1 <= (length/(width+0.0001)) and (length/(width+0.0001)) <= max_lw_ratio)
        filter_parent = (parent not in shapes)
# condional statement for implementation of defined filters on contour selection
        if (filter_diameter and filter_lw_ratio):
            cv.drawContours(blank_included, cnt, -1, (0,255,255), 2)
            if filter_parent:
                areas_list.append(float(area))
                diameters_list.append(float(diameter))
                lengths_list.append(float(length))
                widths_list.append(float(width))
                shapes.append(parent)
        else:
            cv.drawContours(blank_included, cnt, -1, (153,76,0), 2)
    #p_value = shapiro(diameters_list)[1]
    if len(areas_list) <= 1 or len(diameters_list) <= 1:
        tk.messagebox.showerror(title= 'No contours', message='There were no contours/structures found matching your parameters defined!')
# transforming array into png-file
    cv.imwrite("blank_included.png", blank_included)
    image_contours = "blank_included.png"
# calling function for refinement parameter setting, opens new window
    refine_window_ow(image_contours, diameters_list, lengths_list, widths_list, num_bins, min_thresh, max_thresh, shapes)

# function for opening window with table of reference measurements
def open_manual_1():
    window_manual = tk.Toplevel()
    window_manual.title("Table of reference values to put in before processing")
    label_header = tk.Label(window_manual, text= """Depending on the magnification used at the microscope to generate the image, a reference bar of chosen length can be integrated.
                Here are reference bars' comparison length and its actual length in pixel, valid for 2048x1536 px images created by 'smart SEM software' by ZEISS .
    These values must be entried to allow the conversional calculation during the image processing.\n""")
    label_header.grid(row= 0, column= 0, columnspan= 3)
    label_row11 = tk.Label(window_manual, text= "Magnification")
    label_row11.grid(row= 1, column= 0)
    label_row12 = tk.Label(window_manual, text= "Reference Bar Length in µm")
    label_row12.grid(row= 1, column= 1)
    label_row13 = tk.Label(window_manual, text= "Bar Length in Pixel")
    label_row13.grid(row= 1, column= 2)
    label_row21 = tk.Label(window_manual, text= "200")
    label_row21.grid(row= 2, column= 0)
    label_row22 = tk.Label(window_manual, text="100")
    label_row22.grid(row= 2, column= 1)
    label_row23 = tk.Label(window_manual, text="140")
    label_row23.grid(row= 2, column= 2)
    label_row31 = tk.Label(window_manual, text= "300")
    label_row31.grid(row= 3, column= 0)
    label_row32 = tk.Label(window_manual, text="100")
    label_row32.grid(row= 3, column= 1)
    label_row33 = tk.Label(window_manual, text="208")
    label_row33.grid(row= 3, column= 2)
    label_row41 = tk.Label(window_manual, text= "400")
    label_row41.grid(row= 4, column= 0)
    label_row42 = tk.Label(window_manual, text="100")
    label_row42.grid(row= 4, column= 1)
    label_row43 = tk.Label(window_manual, text="276")
    label_row43.grid(row= 4, column= 2)
    label_row51 = tk.Label(window_manual, text= "500")
    label_row51.grid(row= 5, column= 0)
    label_row52 = tk.Label(window_manual, text="20")
    label_row52.grid(row= 5, column= 1)
    label_row53 = tk.Label(window_manual, text="72")
    label_row53.grid(row= 5, column= 2)
    label_row61 = tk.Label(window_manual, text= "600")
    label_row61.grid(row= 6, column= 0)
    label_row62 = tk.Label(window_manual, text="20")
    label_row62.grid(row= 6, column= 1)
    label_row63 = tk.Label(window_manual, text="86")
    label_row63.grid(row= 6, column= 2)
    label_row71 = tk.Label(window_manual, text= "700")
    label_row71.grid(row= 7, column= 0)
    label_row72 = tk.Label(window_manual, text="20")
    label_row72.grid(row= 7, column= 1)
    label_row73 = tk.Label(window_manual, text="100")
    label_row73.grid(row= 7, column= 2)
    label_row81 = tk.Label(window_manual, text= "800")
    label_row81.grid(row= 8, column= 0)
    label_row82 = tk.Label(window_manual, text="20")
    label_row82.grid(row= 8, column= 1)
    label_row83 = tk.Label(window_manual, text="112")
    label_row83.grid(row= 8, column= 2)
    label_row91 = tk.Label(window_manual, text= "900")
    label_row91.grid(row= 9, column= 0)
    label_row92 = tk.Label(window_manual, text="20")
    label_row92.grid(row= 9, column= 1)
    label_row93 = tk.Label(window_manual, text="126")
    label_row93.grid(row= 9, column= 2)
    label_row101 = tk.Label(window_manual, text= "1000")
    label_row101.grid(row= 10, column= 0)
    label_row102 = tk.Label(window_manual, text="20")
    label_row102.grid(row= 10, column= 1)
    label_row103 = tk.Label(window_manual, text="140")
    label_row103.grid(row= 10, column= 2)
    label_row111 = tk.Label(window_manual, text= "1100")
    label_row111.grid(row= 11, column= 0)
    label_row112 = tk.Label(window_manual, text="20")
    label_row112.grid(row= 11, column= 1)
    label_row113 = tk.Label(window_manual, text="154")
    label_row113.grid(row= 11, column= 2)
    label_row121 = tk.Label(window_manual, text= "1200")
    label_row121.grid(row= 12, column= 0)
    label_row122 = tk.Label(window_manual, text="20")
    label_row122.grid(row= 12, column= 1)
    label_row123 = tk.Label(window_manual, text="168")
    label_row123.grid(row= 12, column= 2)
    label_row131 = tk.Label(window_manual, text= "1300")
    label_row131.grid(row= 13, column= 0)
    label_row132 = tk.Label(window_manual, text="20")
    label_row132.grid(row= 13, column= 1)
    label_row133 = tk.Label(window_manual, text="180")
    label_row133.grid(row= 13, column= 2)
    label_row141 = tk.Label(window_manual, text= "1400")
    label_row141.grid(row= 14, column= 0)
    label_row142 = tk.Label(window_manual, text="10")
    label_row142.grid(row= 14, column= 1)
    label_row143 = tk.Label(window_manual, text="100")
    label_row143.grid(row= 14, column= 2)
    label_row151 = tk.Label(window_manual, text= "1500")
    label_row151.grid(row= 15, column= 0)
    label_row152 = tk.Label(window_manual, text="10")
    label_row152.grid(row= 15, column= 1)
    label_row153 = tk.Label(window_manual, text="106")
    label_row153.grid(row= 15, column= 2)
    label_row161 = tk.Label(window_manual, text= "1600")
    label_row161.grid(row= 16, column= 0)
    label_row162 = tk.Label(window_manual, text="10")
    label_row162.grid(row= 16, column= 1)
    label_row163 = tk.Label(window_manual, text="112")
    label_row163.grid(row= 16, column= 2)
    label_row171 = tk.Label(window_manual, text= "1700")
    label_row171.grid(row= 17, column= 0)
    label_row172 = tk.Label(window_manual, text="10")
    label_row172.grid(row= 17, column= 1)
    label_row173 = tk.Label(window_manual, text="120")
    label_row173.grid(row= 17, column= 2)
    label_row181 = tk.Label(window_manual, text= "1800")
    label_row181.grid(row= 18, column= 0)
    label_row182 = tk.Label(window_manual, text="10")
    label_row182.grid(row= 18, column= 1)
    label_row183 = tk.Label(window_manual, text="126")
    label_row183.grid(row= 18, column= 2)
    label_row191 = tk.Label(window_manual, text= "1900")
    label_row191.grid(row= 19, column= 0)
    label_row192 = tk.Label(window_manual, text="10")
    label_row192.grid(row= 19, column= 1)
    label_row193 = tk.Label(window_manual, text="132")
    label_row193.grid(row= 19, column= 2)
    label_row201 = tk.Label(window_manual, text= "2000")
    label_row201.grid(row= 20, column= 0)
    label_row202 = tk.Label(window_manual, text="10")
    label_row202.grid(row= 20, column= 1)
    label_row203 = tk.Label(window_manual, text="138")
    label_row203.grid(row= 20, column= 2)

# function for opening window with explanations to refinement parameters
def open_manual_2():
    window_manual = tk.Toplevel()
    window_manual.title("Explanations of refinement parameters")
    label_1 = tk.Label(window_manual, text= "The desriptive statistics refer to all collected average diameters of the individual contours detected.\n\n")
    label_1.grid(row= 0, column= 0)
    label_2 = tk.Label(window_manual, text= "The Mean value is marked by a solid line in the plots.\n\n")
    label_2.grid(row= 1, column= 0)
    label_3 = tk.Label(window_manual, text= "The Median value is marked by a dotted line in the plots.\n\n")
    label_3.grid(row= 2, column= 0)
    label_4 = tk.Label(window_manual, text= """The Count value shows how many structures/connected outlines of granules within 
                                            the minimum to maximum range are included in the statistics shown.\n\n""")
    label_4.grid(row= 3, column= 0)
    label_5 = tk.Label(window_manual, text= "Setting a minimum average diameter excludes all contours with a lower average diameter.\n\n")
    label_5.grid(row= 4, column= 0)
    label_6 = tk.Label(window_manual, text= "Setting a maximum average diameter excludes all contours with a higher average diameter.\n\n")
    label_6.grid(row= 5, column= 0)
    label_7 = tk.Label(window_manual, text= "Setting the number of bins defines the number of columns in the distribution plots.\n\n")
    label_7.grid(row=6, column= 0)
    label_8 = tk.Label(window_manual, text= """The lower threshold value sets the pixel intensity where pixels with intensities below
                                            will be set to white during image binarisation within the processing algorithm.\n\n""")
    label_8.grid(row= 7, column= 0)
    label_9 = tk.Label(window_manual, text= """The upper threshold value sets the pixel intensity where pixels with intensities above
                                            will be considered belonging to contour forming edges and set to black during image binarisation.\n\n""")
    label_9.grid(row= 8, column= 0)
    label_10 = tk.Label(window_manual, text= """It should take into consideration that a big difference between lower and upper threshold excludes
                                            more edges and therefore less contours are found. This is also important to exclude small unwanted 
                                            contours resulting from including to many pixels as edge building ones. But choosing the thresholds to
                                            narrow will lead to lots of unwanted small contours due to too many included pixels as edge building.\n\n""")
    label_10.grid(row= 9, column= 0)
    label_11 = tk.Label(window_manual, text= """The L/W-Ratio describes the coefficient of length divided by the width of a granule surrounding outline.
                                            In that way circles and ellipses of different length and width proportions can be in- or excluded. 
                                            1 means perfect circle, is already set as minimum and the higher the max. L/W-Ratio the more elongated
                                            elliptic structures can be selected.\n\n""")
    label_11.grid(row= 10, column= 0)

# function for creating a basic excel file with data tables and plot
def create_excel(diameters, lengths, widths, bins, interval_minimum, interval_maximum, filename, contour_image):
    path = filedialog.asksaveasfile(defaultextension=("Excel Files",'*.xlsx'))
    df1 = DataFrame({"Diameter": diameters, "Length": lengths, "Width": widths}).round(1)
    df2 = DataFrame({"Length and Width":(list(lengths) + list(widths))}).round(1)
# counted occurances of each value found in the initially fed lists
    diameter_counts = df1.value_counts("Diameter").sort_index()
    length_counts = df1.value_counts("Length").sort_index()
    width_counts = df1.value_counts("Width").sort_index()
    l_w_counts = df2.value_counts("Length and Width").sort_index()
# empty lists to be filled with values for the excelfile
    diameter_values = [0]* bins
    categories = [""]* bins
    length_values = [0]* bins
    width_values = [0]* bins
    combi_values = [0]* bins
# filling the empty lists with summed counts per bin range of values contained within
    for bin in range(0, bins):
        int_min = round(float(interval_minimum + ((interval_maximum - interval_minimum) * bin/bins)), 1)
        int_max = round(float(interval_minimum + ((interval_maximum - interval_minimum) * (bin+1)/bins)), 1)
        categories[bin] = f"]{int_min} ; {int_max}]"
        for idx in diameter_counts.index:
            if idx > int_min and idx <= int_max:
                diameter_values[bin] += diameter_counts[idx]
        for idx in length_counts.index:
            if idx > int_min and idx <= int_max:
                length_values[bin] += length_counts[idx]
        for idx in width_counts.index:
            if idx > int_min and idx <= int_max:
                width_values[bin] += width_counts[idx]
        for idx in l_w_counts.index:
            if idx > int_min and idx <= int_max:
                combi_values[bin] += l_w_counts[idx]
# lists for probabilites of sizes, width and length
    diameter_probs = [round(val/sum(diameter_values), 2) for val in diameter_values]
    length_probs = [round(val/sum(length_values), 2) for val in length_values]
    width_probs = [round(val/sum(width_values), 2) for val in width_values]
# transforming the image array into dictionary for counting intensities
    intensities_list = []
    for i in image_process:
        intensities_list.extend(i)
    intensities_count_dict = {}
    for j in intensities_list:
        key = int(j)
        if key in intensities_count_dict.keys():
            intensities_count_dict[key] += 1
        else:
            intensities_count_dict[key] = 1
    category_list = []
    for k in range(256):
        category_list.append(k)
    count_values_list = []
    for l in category_list:
        if l in list(intensities_count_dict.keys()):
            count_values_list.append(intensities_count_dict[l])
        else:
            count_values_list.append(0)
# creating workbook and sheets in excel, setting some formats
    wb = xw.Workbook(path.name)
    ws_diameter = wb.add_worksheet("Avg_Diameter")
    ws_l_w = wb.add_worksheet("Length_and_Width")
    ws_length = wb.add_worksheet("Length")
    ws_width = wb.add_worksheet("Width")
    ws_raw_data = wb.add_worksheet("Raw_Data")
    ws_contour_img = wb.add_worksheet("Contours")
    ws_pxl_dist = wb.add_worksheet("Intensities")
    bold = wb.add_format({"bold": True})
    italic = wb.add_format({"italic":True})
    filename = filename.split('/')[-1]
    owner_phrase = 'Analysis results created with "Granule Analyzer" developed by Markus Pieth'
# writing diameter data created into respective excel sheet
    ws_diameter.merge_range("A1:H1", filename, italic)
    ws_diameter.merge_range("I1:P1", owner_phrase, bold)
    ws_diameter.write("A3", "avg diameter", bold)
    ws_diameter.write("B3", "count", bold)
    ws_diameter.write("C3", "prob.", bold)
    ws_diameter.write_column("A5", categories)
    ws_diameter.write_column("B5", diameter_values)
    ws_diameter.write_column("C5", diameter_probs)
# writing length data created into respective excel sheet
    ws_length.merge_range("A1:H1", filename, italic)
    ws_length.merge_range("I1:P1", owner_phrase, bold)
    ws_length.write("A3", "length", bold)
    ws_length.write("B3", "count", bold)
    ws_length.write("C3", "prob.", bold)
    ws_length.write_column("A5", categories)
    ws_length.write_column("B5", length_values)
    ws_length.write_column("C5", length_probs)
# writing width data created into respective excel sheet
    ws_width.merge_range("A1:H1", filename, italic)
    ws_width.merge_range("I1:P1", owner_phrase, bold)
    ws_width.write("A3", "width", bold)
    ws_width.write("B3", "count", bold)
    ws_width.write("C3", "prob.", bold)
    ws_width.write_column("A5", categories)
    ws_width.write_column("B5", width_values)
    ws_width.write_column("C5", width_probs)
# writing overlaid length and width data created into respective excel sheet
    ws_l_w.merge_range("A1:H1", filename, italic)
    ws_l_w.merge_range("I1:P1", owner_phrase, bold)
    ws_l_w.write("A3", "size", bold)
    ws_l_w.write("B3", "length count", bold)
    ws_l_w.write("C3", "width count", bold)
    ws_l_w.write("D3", "length prob.", bold)
    ws_l_w.write("E3", "width prob.", bold)
    ws_l_w.write_column("A5", categories)
    ws_l_w.write_column("B5", length_values)
    ws_l_w.write_column("C5", width_values)
    ws_l_w.write_column("D5", length_probs)
    ws_l_w.write_column("E5", width_probs)
# writing raw data from image processing in raw data sheet
    desc_stats_labels = ["min", "max", "mean", "median", "std", "p-value", "count"]
    ws_raw_data.merge_range("A1:H1", filename, italic)
    ws_raw_data.merge_range("I1:P1", owner_phrase, bold)
    ws_raw_data.merge_range("C3:D3", "diameters statistics", bold)
    ws_raw_data.merge_range("H3:I3", "lengths statistics", bold)
    ws_raw_data.merge_range("M3:N3", "widths statistics", bold)
    ws_raw_data.write("A3", "diameters [µm]", bold)
    ws_raw_data.write("F3", "lengths [µm]", bold)
    ws_raw_data.write("K3", "widths [µm]", bold)
    ws_raw_data.write_column("A5", diameters)
    ws_raw_data.write_column("C5", desc_stats_labels)
    ws_raw_data.write_column("F5", lengths)
    ws_raw_data.write_column("H5", desc_stats_labels)
    ws_raw_data.write_column("K5", widths)
    ws_raw_data.write_column("M5", desc_stats_labels)
    ws_raw_data.write("D5", min(diameters))
    ws_raw_data.write("D6", max(diameters))
    ws_raw_data.write("D7", round(np.mean(diameters),2))
    ws_raw_data.write("D8", np.median(diameters))
    ws_raw_data.write("D9", round(np.std(diameters),2))
    ws_raw_data.write("D10", shapiro(diameters)[1])
    ws_raw_data.write("D11", len(diameters))
    ws_raw_data.write("I5", min(lengths))
    ws_raw_data.write("I6", max(lengths))
    ws_raw_data.write("I7", round(np.mean(lengths),2))
    ws_raw_data.write("I8", np.median(lengths))
    ws_raw_data.write("I9", round(np.std(lengths),2))
    ws_raw_data.write("I10", shapiro(lengths)[1])
    ws_raw_data.write("I11", len(lengths))
    ws_raw_data.write("N5", min(widths))
    ws_raw_data.write("N6", max(widths))
    ws_raw_data.write("N7", round(np.mean(widths),2))
    ws_raw_data.write("N8", np.median(widths))
    ws_raw_data.write("N9", round(np.std(widths),2))
    ws_raw_data.write("N10", shapiro(widths)[1])
    ws_raw_data.write("N11", len(widths))
# writing contour image data from image processing in contours sheet
    ws_contour_img.merge_range("A1:H1", filename, italic)
    ws_contour_img.merge_range("I1:P1", owner_phrase, bold)
    ws_contour_img.insert_image("E3", contour_image)
# writing pixel intensity data in intensities sheet
    ws_pxl_dist.merge_range("A1:H1", filename, italic)
    ws_pxl_dist.merge_range("I1:P1", owner_phrase, bold)
    ws_pxl_dist.write("A3", "intensity", bold)
    ws_pxl_dist.write("B3", "count", bold)
    ws_pxl_dist.write_column("A5", category_list)
    ws_pxl_dist.write_column("B5", count_values_list)
# creating diameter chart within excel using data put in before
    chart_col_diameter = wb.add_chart({"type": "column"})
    chart_col_diameter.add_series({"values": f"=Avg_Diameter!$B$5:$B${5+(bins-1)}",
                                   'fill': {'color':'#000000'},
                                   "name": "counts per bin",
                                   "categories": f"=Avg_Diameter!$A$5:$A${5+(bins-1)}"})
    chart_line_diameter = wb.add_chart({"type": "line"})
    chart_line_diameter.add_series({"values": f"=Avg_Diameter!$C$5:$C${5+(bins-1)}",
                                    'line': {'color':'#A0A0A0'},
                                   "name": "probability per bin",
                                   "categories": f"=Avg_Diameter!$A$5:$A${5+(bins-1)}",
                                   "y2_axis": True})
    chart_col_diameter.combine(chart_line_diameter)
    chart_col_diameter.set_title({"name": "Avg Diameters of Granules"})
    chart_col_diameter.set_x_axis({"name": "avg diameter [µm]"})
    chart_col_diameter.set_y_axis({"name": "count"})
    chart_col_diameter.set_y2_axis({"name": "probability"})
# creating length chart within excel using data put in before
    chart_col_length = wb.add_chart({"type": "column"})
    chart_col_length.add_series({"values": f"=Length!$B$5:$B${5+(bins-1)}",
                                 'fill': {'color':'#000000'},
                                   "name": "counts per bin",
                                   "categories": f"=Length!$A$5:$A${5+(bins-1)}"})
    chart_line_length = wb.add_chart({"type": "line"})
    chart_line_length.add_series({"values": f"=Length!$C$5:$C${5+(bins-1)}",
                                  'line': {'color':'#A0A0A0'},
                                   "name": "probability per bin",
                                   "categories": f"=Length!$A$5:$A${5+(bins-1)}",
                                   "y2_axis": True})
    chart_col_length.combine(chart_line_length)
    chart_col_length.set_title({"name": "Length of Granules"})
    chart_col_length.set_x_axis({"name": "lenght [µm]"})
    chart_col_length.set_y_axis({"name": "count"})
    chart_col_length.set_y2_axis({"name": "probability"})
# creating width chart within excel using data put in before
    chart_col_width = wb.add_chart({"type": "column"})
    chart_col_width.add_series({"values": f"=Width!$B$5:$B${5+(bins-1)}",
                                   "fill": {"color":"#000000"},
                                   "name": "counts per bin",
                                   "categories": f"=Width!$A$5:$A${5+(bins-1)}"})
    chart_line_width = wb.add_chart({"type": "line"})
    chart_line_width.add_series({"values": f"=Width!$C$5:$C${5+(bins-1)}",
                                   "line": {"color":"#A0A0A0"},
                                   "name": "probability per bin",
                                   "categories": f"=Width!$A$5:$A${5+(bins-1)}",
                                   "y2_axis": True})
    chart_col_width.combine(chart_line_width)
    chart_col_width.set_title({"name": "Width of Granules"})
    chart_col_width.set_x_axis({"name": "width [µm]"})
    chart_col_width.set_y_axis({"name": "count"})
    chart_col_width.set_y2_axis({"name": "probability"})
# creating length and width overlaid charts within excel using data put in before
    chart_col_l_w = wb.add_chart({"type": "column"})
    chart_col_l_w.add_series({"values": f"=Length_and_Width!$B$5:$B${5+(bins-1)}",
                                   "fill": {"color":"#000000"},
                                   "name": "length counts per bin",
                                   "categories": f"=Length_and_Width!$A$5:$A${5+(bins-1)}"})
    chart_col_l_w.add_series({"values": f"=Length_and_Width!$C$5:$C${5+(bins-1)}",
                                   "fill": {"color":"#0000FF"},
                                   "name": "width counts per bin",
                                   "categories": f"=Length_and_Width!$A$5:$A${5+(bins-1)}"})
    chart_line_l_w = wb.add_chart({"type": "line"})
    chart_line_l_w.add_series({"values": f"=Length_and_Width!$D$5:$D${5+(bins-1)}",
                                   "line": {"color":"#A0A0A0"},
                                   "name": "length probability per bin",
                                   "categories": f"=Length_and_Width!$A$5:$A${5+(bins-1)}",
                                   "y2_axis": True})   
    chart_line_l_w.add_series({"values": f"=Length_and_Width!$E$5:$E${5+(bins-1)}",
                                   "line": {"color":"#66B2FF"},
                                   "name": "width probability per bin",
                                   "categories": f"=Length_and_Width!$A$5:$A${5+(bins-1)}",
                                   "y2_axis": True})
    chart_col_l_w.combine(chart_line_l_w)
    chart_col_l_w.set_title({"name": "Lengths and Widths of Granules"})
    chart_col_l_w.set_x_axis({"name": "length and width [µm]"})
    chart_col_l_w.set_y_axis({"name": "count"})
    chart_col_l_w.set_y2_axis({"name": "probability"})
# creating pixel intensities chart within excel using data put in before
    chart_col_intensities = wb.add_chart({"type": "column"})
    chart_col_intensities.add_series({"values":"=Intensities!$B$5:$B$261",
                                    "fill": {"color": "#000000"},
                                    "categories": "=Intensities!$A$5:$A$261"})
    chart_col_intensities.set_title({"name": f'Pixel intensity distribution of "{filename}"'})
    chart_col_intensities.set_x_axis({"name": "pixel intesity (black to white)"})
    chart_col_intensities.set_y_axis({"name": "count"})
# inserting charts/images and saving file
    ws_diameter.insert_chart("E3", chart_col_diameter)
    ws_length.insert_chart("E3", chart_col_length)
    ws_width.insert_chart("E3", chart_col_width)
    ws_l_w.insert_chart("G3", chart_col_l_w)
    ws_pxl_dist.insert_chart("E3", chart_col_intensities)
    wb.close()

# function that overwrites global image variables for processing and presentation with reduced image
def reset_global_images(left, top, right, bottom, file_path):
    img = Image.open(file_path)
    img_width, img_height = img.size
    cut_left = int(img_width*(left/100))
    cut_top = int(img_height*(top/100))
    cut_right = int(img_width - (img_width*(right/100)))
    cut_bottom = int(img_height - (img_height*(bottom/100)))
    img_cut = img.crop([cut_left, cut_top, cut_right, cut_bottom])
    image_show = ImageTk.PhotoImage(img_cut.resize((717, 545)))
    label_img_open.config(image= image_show)
    global image_process
    image_process = cv.imread(file_path, cv.IMREAD_GRAYSCALE)[cut_top:(cut_bottom+1),cut_left:(cut_right+1)]
    global image_orig
    image_orig = cv.imread(file_path)[cut_top:(cut_bottom+1),cut_left:(cut_right+1)]
    file_root.mainloop()

# buttons of the file dialog window for triggering functions defined above
button_open = tk.Button(file_root, text= "Open file", command= open_file)
button_open.grid(row= 2, column= 0, pady= 20)
label_open = tk.Label(file_root, text= "Open an image file to get to the next window where\n the image processing parameters will be set.")
label_open.grid(row= 2, column= 1)

file_root.mainloop()