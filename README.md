# COINTADOR: A Coin Counter Brazilian Real and Dollar
This work aims to count coins in images using only the OpenCV library and **Python 3.7**

## :books: Libraries (requirements.txt)
- certifi==2020.6.20
- numpy==1.19.2
- opencv-contrib-python==4.4.0.44
- opencv-python==4.4.0.44
- PyYAML==5.3.1
- wincertstore==0.2

Install the libraries needed for this work:

        { pip install -r requirements.txt }        

## :heavy_check_mark: Run First Steps

#### 1. Clone repository
Clone this repository from Github. 

#### 2. Detect Coins
##### 2.1 Command to detect coins Brazilian Real:
        { python main.py -o real }

##### 2.2 Command to detect coins Dollar:
        { python main.py -o dollar }

#### 3. Result images
##### 3.1 Result for Coins Brazilian Real:
        { image_result/real/ }

##### 3.2 Result for Coins Dollar:
        { image_result/dollar/ }


## :chart_with_upwards_trend: Step-by-Step Image Results
#### Coins Brazilian Real
##### Step 1: Read Image
![InputReal](https://github.com/eduardocarnunes/coin_counter/blob/main/image_result/real/image_input.jpg)

##### Step 2: Convert BGR Image to Grayscale
![GrayReal](https://github.com/eduardocarnunes/coin_counter/blob/main/image_result/real/image_gray.jpg)

##### Step 3: Blur Image
![BlurReal](https://github.com/eduardocarnunes/coin_counter/blob/main/image_result/real/image_blur.jpg)

##### Step 4: Detect circles with HoughCircles and Final Result
![CirclesReal](https://github.com/eduardocarnunes/coin_counter/blob/main/image_result/real/real_result.jpg)

#### Coins Dollar
##### Step 1: Read Image
![InputDollar](https://github.com/eduardocarnunes/coin_counter/blob/main/image_result/dollar/image_input.jpg)

##### Step 2: Convert BGR Image to Grayscale
![GrayDollar](https://github.com/eduardocarnunes/coin_counter/blob/main/image_result/dollar/image_gray.jpg)

##### Step 3: Binary Segmentation
![SegDollar](https://github.com/eduardocarnunes/coin_counter/blob/main/image_result/dollar/image_segmentation.jpg)

##### Step 4: Apply Morphological Transformations
![MorphoDollar](https://github.com/eduardocarnunes/coin_counter/blob/main/image_result/dollar/image_morphological.jpg)

##### Step 5: Detect circler with SimpleBlobDetector and Final Result
![CirclesDollar](https://github.com/eduardocarnunes/coin_counter/blob/main/image_result/dollar/dolar_result.jpg)

