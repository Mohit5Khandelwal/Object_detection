import cv2
import numpy as np
import time

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#loading the image
cap = cv2.VideoCapture(0)

starting_time = time.time()
frame_id = 0

while True:
    _, frame = cap.read()
    frame_id += 1

    height , width , channels = frame.shape


    # Detecting Objects extract 416 x 416 image
    blob = cv2.dnn.blobFromImage( frame , 0.00392 , ( 320 , 320 ) , ( 0 , 0 , 0 ) , True , crop=False )

    #Creating an three sperate  channel of image RGB
    # for b in blob:
    #     for n,img_blob in enumerate(b):
    #         cv2.imshow( str(n) , img_blob )

    #Passing these three into yolo
    net.setInput(blob)
    outs = net.forward( output_layers )

    #Showing the information on the screen
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scroes  = detection[ 5 : ]
            class_id = np.argmax( scroes )
            confidence = scroes[ class_id ]
            if confidence > 0.2:
                #Object detected height and width be original size of image
                center_x = int( detection[0] * width )
                center_y = int(detection[1] * height )
                w = int( detection[2] * width )
                h = int(detection[3] * height )

                #cv2.circle( img , ( center_x , center_y ) , 10 , ( 0 , 255 , 0 ) , 2 )

                #Creating an rectangle
                #Rectangle coordinate
                x = int( center_x - w / 2 ) #Top left point
                y = int( center_y - h / 2 )

                boxes.append( [ x , y , w , h ] )
                confidences.append( float( confidence ) )
                class_ids.append( class_id )

                # cv2.rectangle( img , ( x , y ) , ( x + w , y + h ) , ( 0 , 255 , 0 ) , 2 )

    indexes = cv2.dnn.NMSBoxes( boxes , confidences , 0.5 , 0.4 )
    print( indexes )

    number_object_detected = len( boxes )

    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range( len( boxes) ):
        x , y , w , h = boxes[i]

        if i in indexes:

            label = classes[ class_ids[i] ]
            color = colors[ class_ids[i] ]
            confidence = confidences[i]
            cv2.rectangle( frame, (x, y), (x + w, y + h), color , 2)
            cv2.putText(frame, label + " " + str( round( confidence , 2 )  ) , ( x , y + 30 ) , font , 1 , color , 3 )









    elapsed_time = time.time() - starting_time #time left from starting_time
    fps = frame_id / elapsed_time

    cv2.putText( frame , "FPS: " + str( round( fps , 2 ) ), (20 , 20) , font , 1 , ( 0 , 0 , 0 ) , 3 )

    cv2.imshow("Image" , frame )
    key = cv2.waitKey(1)
    if key == 27: #esc key
        break

cap.release()
cv2.destoryAllWindows()




