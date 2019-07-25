import time
import cv2
import os
import numpy as np
import pandas as pd
import pytesseract
import signal

#constants based on video input
PT1 = (500,2200)
PT2 = (3000,2600)
SIZE = (1920,1080)

class TimeoutException(Exception):   # Custom exception class
    pass

class GeneralException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException    

# Change the behavior of SIGALRM
#signal.signal(signal.SIGALRM, timeout_handler)

def load_img(img_path):
    # Read image using opencv
    img = cv2.imread(img_path)
    return img
    
def process_img(img):
    # identify region of interest
    roi = img[PT1[1]:PT2[1],PT1[0]:PT2[0]]
    # Convert to grayscay
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Equalize
    roi = cv2.equalizeHist(roi)
    # Apply dilation and erosion to remove some noise
    kernel = np.ones((2, 2), np.uint8)
    roi = cv2.erode(roi, kernel, iterations=1)
    roi = cv2.dilate(roi, kernel, iterations=1)
    # Apply threshold to remove noise 
    roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 155, 15)
    roi = cv2.erode(roi, kernel, iterations=1)
    roi = cv2.dilate(roi, kernel, iterations=1)
    return roi

def img_to_str(img):
    roi = process_img(img)
    result = pytesseract.image_to_string(roi, lang="eng")
    return result.replace(" ", "").upper()

def show_img_with_box(img, match=False):
    thickness=15
    if not match:
        line_color = (0,0,255)
    else:
        line_color = (0,255,0)
    cv2.rectangle(img, PT1, PT2, line_color, thickness)
    return img

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)    
    
def process_stream(input_stream=None):
    # finite stream vs live stream
    if not input_stream:
        cap = cv2.VideoCapture(0)
    else: 
        cap = cv2.VideoCapture(input_stream)
    
    if not cap.isOpened():
        print('Error opening video stream or file')

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, SIZE)
    
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            match = False
            signal.setitimer(signal.ITIMER_REAL, 1)
            try:
                # Our operations on the frame come here
                str_out = img_to_str(frame)
                if str_out == '':
                    continue
		#TODO: actually check against names; below is for testing
                if 'MATTHEWVANANTWERP' in str_out:
                    match = True
            except TimeoutException:
                pass
            except GeneralException:
                pass
            else:
                # Reset the alarm
                signal.setitimer(signal.ITIMER_REAL, 0)
            # Display frame out
            frame_out = show_img_with_box(frame, match) # this will need to be sent back out
            yield cv2.imencode('.jpg', frame_out)[1].tobytes()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
