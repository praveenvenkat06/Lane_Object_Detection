import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_line(image, line_parameters):
    slope = line_parameters[0]
    intercept = line_parameters[1]
    y1 = image.shape[0]  #lane detection line starts from bottom of image
    y2 = int(y1*(3/5))  #lane detection line ends at 3/5 of height of image
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

#convert to grayscale, blur and return the canny image
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    blur = cv2.GaussianBlur(gray, (5, 5), 0) #(image, (kernel size), standard deviation)
    canny_img = cv2.Canny(blur, 50, 150) #(image, low_threshold, high threshold)
    return canny_img

def region_of_interest(image):
    # plt.imshow(image)
    # plt.show()
    mask = np.zeros_like(image)   
    #Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    #We could have used fixed numbers as the vertices of the polygon,
    #but they will not be applicable to images with different dimesnions.
    rows, cols = image.shape[:2]
    bottom_left  = [cols * 0.1, rows * 0.95]
    top_left     = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right    = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit =[]
    try:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)  #getting two points for each line
            parameters = np.polyfit((x1, x2), (y1, y2), 1) #find slope and intercept by passing points and degree of the polynomial is 1
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:  #left_fit because as x increases y decreases(y increases from top to bottom)
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
            
        left_fit_average = np.average(left_fit, axis = 0)
        right_fit_average = np.average(right_fit, axis = 0)

        left_line = make_line(image, left_fit_average)
        right_line = make_line(image, right_fit_average)

        return np.array([left_line, right_line])
    except:
        return None


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255,0,0), 10)
    return line_image



#lane_image = cv2.imread('test_image.jpg') #read and store image as multidimensional numpy image
# canny_image = canny(lane_image)
# interest_image = region_of_interest(canny_image)

# #arguments - image, binsize rho, binsize theta, threshold(min no.of curves intersecting), placeholder(not important), min line length, max gap between lines that can be connected
# lines = cv2.HoughLinesP(interest_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, averaged_lines) #many lines which is nearby displays.. so take average to make them a single line

# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

# cv2.imshow('result', combo_image) #render image - ('name of the window', image to show)
# cv2.waitKey(0) #displays the image fro specified seconds (0 - for infinite time display)

vid_cap = cv2.VideoCapture("solidWhiteRight.mp4")
while(vid_cap.isOpened()):
    ret, frame = vid_cap.read()
    if not ret:
        break
    canny_image = canny(frame)
    interest_image = region_of_interest(canny_image)

    #arguments - image, binsize rho, binsize theta, threshold(min no.of curves intersecting), placeholder(not important), min line length, max gap between lines that can be connected
    lines = cv2.HoughLinesP(interest_image, 1, np.pi/180, 20, np.array([]), minLineLength=20, maxLineGap=500)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines) #many lines which is nearby displays.. so take average to make them a single line

    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    cv2.imshow('result', combo_image) #render image - ('name of the window', image to show)
    if cv2.waitKey(25) == ord('q'): #displays the image for specified (0 - for infinite time display)
        break

vid_cap.release()
cv2.destroyAllWindows()
