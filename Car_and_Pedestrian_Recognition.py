import cv2

# car image
car_img = 'car_traffic.jpg'
car_video = cv2.VideoCapture('tesla_car_crash.mov')

# pre-trained car classifying data
classifier= 'cars_haar_cascades.xml'
# pre-trained data for pedestrian recognition
pedestrian_data = 'pedestrian_haar_cascade.xml'

# create car classifier
car_tracker = cv2.CascadeClassifier(classifier)
# creating pedestrian classifier
pedestrian_track = cv2.CascadeClassifier('pedestrian_haar_cascade.xml')

# code to read car movement in streets via tesla accident model video
# run forever until vehicles stop
while True:

    # read current frame
    (read_successful, frame) = car_video.read()

    # converting video to grayscale if read
    if read_successful:
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detecting cars in video
    car_detector = car_tracker.detectMultiScale(grayscale_frame)
    # detect pedestrians
    pedestrians = pedestrian_track.detectMultiScale(grayscale_frame)
    print(pedestrians)

    # drawing boxes around cars and pedestrians that are detected
    for (x, y, w, h) in car_detector:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 2)

    # displays images with cars spotted
    cv2.imshow('Car and Pedestrian Recognition', frame)

    # waits till any key is pressed to close python program
    cv2.waitKey(1)

    # listens for 1ms to check if any key was pressed, then move on
    key = cv2.waitKey(1)

    # stop is Q key is pressed
    if key == 81 or key == 113:
        break
    
    # release video capture
    car_video.release()
print('successful')