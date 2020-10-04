# -*- coding: utf-8 -*-
"""
Created on Friday Oct  2 2020

@author: Eduardo Carvalho Nunes

e-mail: eduardocarvnunes@gmail.com
"""
#import libraries
import cv2
import numpy as np
import yaml

# load parameter.yaml
def yaml_load():
    with open("parameter.yaml") as stream:
        param = yaml.safe_load(stream)
    return param

def read_image(image_file):
    """
    Read an image    

    Parameters
    ----------
    image_file : String
        Image file path
        
    Returns
    -------
    image : Array of unit8 or None
        Returns a color image or return None (if there is an error).

    """
    try:
        image = cv2.imread(image_file)
        return image
    except:
        print('[ERROR]: could not read image')
        return None

def bgr_gray(image):  
    """    
    Convert BGR to Gray using the OpenCV
    Parameters
    ----------
    image : Array of uint8
        A color image

    Returns
    -------
    image_gray : Array of uint8 or None
        Return the grayscale image or return None (if there is an error)
    """   
    try:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)        
        return image_gray
    except:
        print('[ERROR]: could not read image ')
        return None
        
def blur_image(image_gray):
    """
    Smoothing Images using Gaussian Blur from the OpenCV library

    Parameters
    ----------
    image_gray : Array of uint8
        Grayscale image

    Returns
    -------
    blurred : Array of uint8 or None
        Returns a grayscale image with a Gaussian Blur Filter 
        or returns None (if there is an error)
    """
    try:    
        #kernel(17,17)
        blurred = cv2.GaussianBlur(image_gray, (17, 17), 0)
        
        if len(blurred.shape) > 2:
            print('[ERROR]: Dimension > 2. Is an image gray?')
            return None
        else:
            return blurred
    except:
        print('[ERROR]: could not convert image')
        return None

def detect_circles(image):
    """
    Detect cicle using the HoughCircles fuction from OpenCV

    Parameters
    ----------
    image : Array of uint8
        A grayscale image with a Gaussian Blur Filter

    Returns
    -------
    circles : Array of float32 or None
        Returns the detector circles with coordinates and radius 
        or returns None (if there is an error)
    """    
    try:
        if len(image.shape) > 2:
            print('[ERROR]: Dimension > 2. Is an image gray?')
            return None
        
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT,1, 
                                   image.shape[0]/8, param1=100,
                                   param2=50,minRadius=0,maxRadius=0)        
        if len(circles) == 0:
            print('[ERROR]: not possible to detect circles')
            return None            
        else:
            return circles
    except:
        print('[ERROR]: could not detect circles')
        return None
    
def segmentation_bin(image_gray, th):
    
    """
    Segments the image using the threshold function of the OpenCV library

    Parameters
    ----------
    image_gray : Array of uint8
        Grayscale image
        
    th : int
        Threshold for segmentation

    Returns
    -------
    image_bin : Array of uint8
        Returns a segmented image with a value of 0 (black pixels) and 
        255 (white pixels) or returns None (if there is an error)

    """
    try:
        if len(image_gray.shape) > 2:
            print('[ERROR]: Dimension > 2. Is an image gray?')
            return None 
          
        ret, image_bin = cv2.threshold(image_gray, th, 255, cv2.THRESH_BINARY_INV)
        
        return image_bin
    except:
        print('[ERROR]: could not segmentation image')
        return None
           


def morphological_transformation(image_gray):
    """
    apply morphological transformation (CLOSING) of the OpenCV library
    reference: https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html

    Parameters
    ----------
    image_gray : Array of uint8
        Grayscale image 

    Returns
    -------
    image_closing : Array of uint8
        Returns an image with morphological transformation (CLOSING) 
        or returns None (if there is an error)

    """
    try:
        if len(image_gray.shape) > 2:
            print('[ERROR]: Dimension > 2. Is an image gray?')
            return None 
    
        #kernel 5x5
        kernel = np.ones((5,5),np.uint8)    
        #closing : dilatation followed by Erosion
        image_closing = cv2.morphologyEx(image_gray, cv2.MORPH_CLOSE, kernel, iterations=3)
        return image_closing
    except:
        print('[ERROR]: could not detect circles')
        return None
    
def simple_blob_detector(image_morpho):
    """
    It uses the SimpleBlobDetector method to detect the coins in an image. 
    This method is implemented in the OpenCV library
    
    params reference: https://www.youtube.com/watch?v=3UjNRJ8jbXE

    Parameters
    ----------
    image_morpho : Array of uint8
        image with morphological transformation.

    Returns
    -------
    detector : SimpleBlobDetector or None
        Returns keypoints of coins or returns None (if there is an error)

    """
    
    try:
        if len(image_morpho.shape) > 2:
            print('[ERROR]: Dimension > 2. Is an image gray?')
            return None
        
        # params 
        params = cv2.SimpleBlobDetector_Params()        
        
        # load parameter.yaml
        param = yaml_load()
        
        # change thresholds
        params.minThreshold = param['low_color']
        params.maxThreshold = param['max_color']
        params.filterByColor = param['threshold']['filterByColor']
        params.blobColor = param['threshold']['blobColor']
        params.minDistBetweenBlobs = param['threshold']['minDistBetweenBlobs']
        params.thresholdStep = param['threshold']['thresholdStep']
        params.minRepeatability = param['threshold']['minRepeatability']
        
        # filter by area
        params.filterByArea = param['area']['filterByArea']
        params.minArea = param['low_area']
        params.maxArea = param['area']['maxArea']
        
        # Filter by Circularity
        params.filterByCircularity = param['circularity']['filterByCircularity']
        params.minCircularity = param['low_circularity'] / 100
        
        # Filter by Convexity
        params.filterByConvexity = param['convexity']['filterByConvexity']
        params.minConvexity = param['low_convexity'] / 100
        
        # Filter by Inertia
        params.filterByInertia = param['inertia']['filterByInertia']
        params.minInertiaRatio = param['low_inertia'] / 100        
        
        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)
        
        # detect coins
        keypoints = detector.detect(image_morpho)
        
        if len(keypoints) == 0:
            print('[ERROR]: not possible detect coins')
            return None
        
        return keypoints      
    except:
        print('[ERROR]: could not detect circles')
        return None 
  
def draw_circles_hough(image, circles):
    """
    draws the border and center of the detected coins

    Parameters
    ----------
    image : Array of uint8
        a color image
    circles : Array of float32
        circle coordinates

    Returns
    -------
    image_final : Array of uint8
        image with the border and center drawn

    """
    try:
        # Convert the circle parameters a, b and r to integers. 
        detected_circles = np.uint16(np.around(circles)) 
  
        for pt in detected_circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2] 
      
            # Draw the circumference of the circle. 
            image = cv2.circle(image, (a, b), r, (0, 255, 0), 2) 
      
            # Draw a small circle (of radius 1) to show the center. 
            image_final = cv2.circle(image, (a, b), 1, (0, 0, 255), 3) 
            
        return image_final
    except:
        print('[ERROR]: could not draw image')
        return None
  
def draw_key_pts(image, keypoints):
    """
    draws the edges of the detected currencies using the drawKeypoints 
    function of the OpenCV library
    
    reference: https://stackoverflow.com/questions/19748020/visualizing-opencv-keypoints

    Parameters
    ----------
    image : Array of uint8
        a color image
    circles : Array of float32
        circle coordinates

    Returns
    -------
    image_final : Array of uint8
        image with the border and center drawn

    """
    
    # Draw blobs on our image as green circles 
    blank = np.zeros((1, 1))  
    image = cv2.drawKeypoints(image, keypoints, blank, (0, 255, 0), 
                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    for curKey in keypoints:
        x=np.int(curKey.pt[0])
        y=np.int(curKey.pt[1])
        #size = np.int(curKey.size)
        image_final = cv2.circle(image,(x,y),2,(255, 0, 0), 3)
        
    return image_final
    
"""    
#test 1    
image = read_image('real_original.jpg')
image_gray = bgr_gray(image) 
image_seg = blur_image(image_gray)
detected_circles = detect_circles(image_seg)
image_final_real = draw_circles_hough(image, detected_circles)
print('[RESULT] Number of coins detected = ' + str(len(detected_circles[0])))
cv2.imshow("1", image) 
cv2.imshow('2', image_gray)
cv2.imshow('3', image_seg)
cv2.imshow('4', image_final_real)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
"""
#test 2
image = read_image('dolar_original.png')
image_gray = bgr_gray(image)
image_seg = segmentation_bin(image_gray, 25)
image_morpho = morphological_transformation(image_seg)
keypoints = simple_blob_detector(image_morpho)
image_final = draw_key_pts(image, keypoints)
print('[RESULT] Number of coins detected = ' + str(len(keypoints)))
cv2.imshow("1", image) 
cv2.imshow('2', image_gray)
cv2.imshow('3', image_seg)
cv2.imshow('4', image_morpho)
cv2.imshow('5', image_final)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
