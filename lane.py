import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)


# Canny Edge detection technique to identify boundaries of an image where there is a sharp change in intensity
# Reducing noise in the image is done by a gaussian blur filter, 5x5 kernel is good for almost all cases though it may vary
# Hough T(ransform for detecting the lanes
def canny(image):
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    blur_image = cv2.GaussianBlur(gray,(5,5),0)
    boundary = cv2.Canny(blur_image,50,150)
    return boundary  

def region(image):
    height = image.shape[0]
    triangle = np.array([[(200,height),(1100,height),(550,250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,triangle,255)
    masked = cv2.bitwise_and(image,mask)
    return masked

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image


#Hough transform is used to detect straight lines (lane lines) in the frame

#image_return = canny(lane_image)
#cropped = region(image_return) 
#lines = cv2.HoughLinesP(cropped,2,np.pi/180,100, np.array([]),minLineLength=40,maxLineGap=5)
#line_image = display_lines(lane_image,lines)
#complete_image = cv2.addWeighted(lane_image, 0.8, line_image,1 , 1)
#cv2.imshow("Result",complete_image)
#cv2.waitKey(0)

video = cv2.VideoCapture("test2.mp4")
while(video.isOpened()):
    frame = video.read()
    image_return = canny(frame)
    cropped = region(image_return) 
    lines = cv2.HoughLinesP(cropped,2,np.pi/180,100, np.array([]),minLineLength=40,maxLineGap=5)
    line_image = display_lines(frame,lines)
    complete_image = cv2.addWeighted(frame, 0.8, line_image,1 , 1)
    cv2.imshow("Result",complete_image)
    cv2.waitKey(1)



