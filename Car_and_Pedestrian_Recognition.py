import cv2


## Car Recognition 

# car image
car_img = 'car_traffic.jpg'
car_video = cv2.VideoCapture('tesla_car_crash.mov')

# pre-trained car classifying data
classifier= 'cars_haar_cascades.xml'

# create car classifier
car_tracker = cv2.CascadeClassifier(classifier)


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
    print(cars)

    # drawing boxes around cars that are detected
    for (x, y, w, h) in car_detector:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # displays images with cars spotted
    cv2.imshow('Car Recognition', frame)

    # waits till any key is pressed to close python program
    cv2.waitKey(1)

# creating opencv image
# img = cv2.imread(car_img)

# converting image to grayscale
# grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# displaying image with the cars spotted
# cv2.imshow('Car Recognition', img)

print('successful')


## Pedstrian Recognition

# pre-trained data for pedestrian recognition
pedestrian_data = 'pedestrian_haar_cascade.xml'

# creating pedestrian classifier
pedestrian_track = cv2.CascadeClassifier('pedestrian_haar_cascade.xml')

# run forever until pedestrian doesn't show up
while True:

    # read the current frame
    (read_successful, frame) = video.read()

    # conversion to grayscale
    if read_successful:
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break