import cv2
import numpy as np 
import imutils

"""
Hand tracking Code

Cristian C. Velandia C.

Creation in C++ 10/04/2019
Migration to Python 17/08/2020
"""
#Binarize using color to detect better the difference when passing the hand in fornt of the face (skin color)
def DetectBlobs(morphed, original):
    Orig_Copy = original.copy()
    approx = []
    BoundingRect = []
    center = []
    radius = []

    IMG_Contours = cv2.findContours(morphed.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #Find contours
    contours = imutils.grab_contours(IMG_Contours) # Get contours
    if len(contours) > 0:        
        for c in range(len(contours)):
            peri = cv2.arcLength(contours[c], True)
            approx.append(cv2.approxPolyDP(contours[c], 0.04 * peri, True))
            BoundingRect.append(cv2.boundingRect(approx[c]))
            center, radius = cv2.minEnclosingCircle(approx[c])

            if cv2.contourArea(contours[c]) > 20000:
                cv2.circle(Orig_Copy, (int(center[0]),int(center[1])), int(radius), (0,255,0), 2)
        #cv2.drawContours(Orig_Copy, [approx], -1, (0, 0, 255), 3)

    return Orig_Copy
    

Video = cv2.VideoCapture() #Define camera object for video

Video.open(0)  # Open camera

if (Video.isOpened() == 0): #Validate if camera could be opened
    print("Could not open the camera")

ret, Background = Video.read()  # background image initialization 
binary = Background.copy()  # bin image initialization
diff = Background.copy()  # diff image initialization

binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)  # Convert bin to greyscale 		

# Window definition
cv2.namedWindow("Original", cv2.WINDOW_AUTOSIZE) 
cv2.namedWindow("Substracted", cv2.WINDOW_AUTOSIZE) 
cv2.namedWindow("Binary", cv2.WINDOW_AUTOSIZE) 
cv2.namedWindow("Morphed", cv2.WINDOW_AUTOSIZE) 

# Create kernel for morph op
kernel = np.ones((5,5), np.uint8)

#Data for video capture
#int frame_width = Video.get(CAP_PROP_FRAME_WIDTH) 
#int frame_height = Video.get(CAP_PROP_FRAME_HEIGHT) 
#Mat Vid = Mat::zeros(Background.rows, Background.cols * 2, CV_8U) 
#VideoWriter video("HandTrack.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 3, Size(frame_width * 2, frame_height), true) 

while (Video.grab()):
    retr, _ = Video.read()
    if (retr):
        _, image = Video.retrieve()  #Get Image from camera		
        imageCopy = image.copy() #copy actual frame			
        
        #Calculate the difference to remove the background of the new image
        diff = cv2.absdiff(imageCopy, Background) 

        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # Convert diff to greyscale 
        
        mask = diff>30 # Create a mask to detect the new object

        binary = np.zeros_like(imageCopy, np.uint8)
        binary[mask] = imageCopy[mask]
        binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(binary, 50, 200, cv2.THRESH_BINARY) 
        #binary = cv2.bitwise_not(binary)

        #Morphology operations to remove noise and clear hand (Opening morphology = Erosion + dilatation)
        morphed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # Apply operator using 6 iterations for Openning

        #Detect blobs(groups of pixels)
        image = DetectBlobs(morphed, image)  #Detect groups using countour area (Green formula)
    

    if (cv2.waitKey(30) == 27): #wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        print("ESC pressed, program will stop") 
        break

    #cvtColor(morphed, morphed, COLOR_GRAY2BGR) 
    #hconcat(image, morphed, Vid) 
    #video.write(Vid) 

    #imshow("Original", Vid) 

    cv2.imshow("Original", image) 
    cv2.imshow("Substracted", diff) 
    cv2.imshow("Binary", binary) 
    cv2.imshow("Morphed", morphed) 

