import cv2
import numpy as np

####### Load Yolo Model (Deep Neural Network)#########
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

####### Loading Image #########
img = cv2.imread("test_image1.jpg")
#height, width, channels = img.shape
#img = cv2.resize(img, None, fx = 0.4, fy = 0.4)

#Loading Video
cap = cv2.VideoCapture("solidWhiteRight.mp4")
while(True):

    _, frame = cap.read()
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
            if confidence > 0.5:
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
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 8), font, 2, color, 2) #(image, text_to_display, where too start, font, font_size, font_color, font_thickness)
            #print(label)

    cv2.imshow("Image", frame)
    if cv2.waitKey(1) == ord('q'): #displays the image for specified (0 - for infinite time display)
        break

cap.release()
cv2.destroyAllWindows()