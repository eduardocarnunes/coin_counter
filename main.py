# -*- coding: utf-8 -*-
"""
Created on Friday Oct  2 2020

@author: Eduardo Carvalho Nunes

e-mail: eduardocarvnunes@gmail.com
"""
import argparse
import cv2
import CoinCounter as cc

param = cc.yaml_load()

def detect_coin_real():
    """
    

    Returns
    -------
    None.

    """
    
    # Test 1 Coin Counter Real (Brazil)
    
    # read image
    image = cc.read_image(param['images']['image1'])
    image_show = image.copy()
    # convert bgr to gray
    image_gray = cc.bgr_gray(image) 
    # image blur with Gaussian filter
    image_blur = cc.blur_image(image_gray)
    # detect circles with HoughCircles
    detected_circles = cc.detect_circles(image_blur)
    # draw border and center 
    image_final_real = cc.draw_circles_hough(image_show, detected_circles)
    
    print('[RESULT] Number of coins detected = ' + str(len(detected_circles[0])))
    
    cv2.imshow('Image Input Real', image) 
    cv2.imwrite('image_result/real/image_input.jpg', image)
    
    cv2.imshow('Image Gray Real', image_gray)
    cv2.imwrite('image_result/real/image_gray.jpg', image_gray)
    
    cv2.imshow('Image Blur Real', image_blur)
    cv2.imwrite('image_result/real/image_blur.jpg', image_blur)
    
    cv2.imshow('Image Result Real', image_final_real)
    cv2.imwrite('image_result/real/real_result.jpg', image_final_real)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_coin_dollar():
    """
    

    Returns
    -------
    None.

    """
    # Test 2 Coin Counter Dollar
    
    # read image
    image = cc.read_image('dolar_original.png')
    image_show = image.copy()
    # bgr to gray
    image_gray = cc.bgr_gray(image)
    # image segmentation with threshold:25
    image_seg = cc.segmentation_bin(image_gray, 25)
    # apply morphological transformation
    image_morpho = cc.morphological_transformation(image_seg)
    keypoints = cc.simple_blob_detector(image_morpho)
    # draw border and center
    image_final_dolar = cc.draw_key_pts(image_show, keypoints)
    
    print('[RESULT] Number of coins detected = ' + str(len(keypoints)))
    
    cv2.imshow("Image Input Dollar", image) 
    cv2.imwrite('image_result/dollar/image_input.jpg', image)
    
    cv2.imshow('Image Gray Dollar', image_gray)
    cv2.imwrite('image_result/dollar/image_gray.jpg', image_gray)
    
    cv2.imshow('Image Segmentation Dollar', image_seg)
    cv2.imwrite('image_result/dollar/image_segmentation.jpg', image_seg)
    
    cv2.imshow('Image Morphological Dollar', image_morpho)
    cv2.imwrite('image_result/dollar/image_morphological.jpg', image_morpho)
    
    cv2.imshow('Image Result Dollar', image_final_dolar)
    cv2.imwrite('image_result/dollar/dolar_result.jpg', image_final_dolar)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    # detect real: python main.py real
    # detect dollar: python main.py dollar
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--opt", required=True, 
                    help=" 'real' for coin counter Real or 'dollar' for coin counter Dollar")
    args = vars(ap.parse_args())
    
    if args['opt'] == "real":
        detect_coin_real()
    elif args['opt'] == "dollar":
        detect_coin_dollar()
    else:
        print("[ERROR]: Option invalid")
        print("Coin Counter Real: python main.py -o real")
        print("Coin Counter Dollar: python main.py -o dollar")

    
    
    