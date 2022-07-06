import cv2
import numpy as np
import matplotlib.pyplot as plt

####################################### LANE DETECTION ###########################################

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



#########################################  OBJECT DETECTION ########################################################

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg") #(model - binary file of trained weights, config - textfile with network configurations)


######## Loading classes ########
# classes = ["person", "bicycle"]
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

#print(classes)

######## Loading Layers ########
layer_names = net.getLayerNames()
#print(layer_names)
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
#print(output_layers)
colors = np.random.uniform(0, 255, size = (len(classes), 3))  #(low intensity, high intensity, (total_colors, channel_size))



#############################################  MERGE LANE OBJECT #######################################################
vid_cap = cv2.VideoCapture("solidWhiteRight.mp4")
while(True):
    ################################## lane detection ###########################################
    ret, frame = vid_cap.read()
    if not ret:
        break
    canny_image = canny(frame)
    interest_image = region_of_interest(canny_image)

    #arguments - image, binsize rho, binsize theta, threshold(min no.of curves intersecting), placeholder(not important), min line length, max gap between lines that can be connected
    lines = cv2.HoughLinesP(interest_image, 1, np.pi/180, 20, np.array([]), minLineLength=20, maxLineGap=500)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines) #many lines which is nearby displays.. so take average to make them a single line



    ######################################## object detection ####################################

    height, width, channels = frame.shape
    ######## Detecting objects #######
    #blobFromImage perform mean subtraction, scaling and color channel swapping
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    #(image, scale_factor usually 1/sigma (1/255 here), spatial_size for CNN, rgb mean subtraction values, swapRB(by default BGR), crop)

    # for b in blob:
    #     for n,img_blob in enumerate(b):
    #         cv2.imshow(str(n), img_blob)

    net.setInput(blob)
    outs = net.forward(output_layers)


    ######## Displaying information #########
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            #choosing the max score detected(confidence) and comparing if > 0.5
            #first four paramters are center_x, center_y, width and height of the object, remaining are the scores
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                #Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                #Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    #Non-max suppression(NMS) removes boxes with high overlapping
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) #(boxes, confidences, confidence_threshold, nms_threshold)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y - 8), font, 1, color, 2) #(image, text_to_display, where too start, font, font_size, font_color, font_thickness)
            #print(label)

    ######################################## merge frames lane object frames ###############################

    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('result', combo_image) #render image - ('name of the window', image to show)
    if cv2.waitKey(25) == ord('q'): #displays the image for specified (0 - for infinite time display)
        break

vid_cap.release()
cv2.destroyAllWindows()
