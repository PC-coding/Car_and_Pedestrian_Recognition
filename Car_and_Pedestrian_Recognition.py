import cv2

# car image
car_img = 'car_traffic.jpg'
car_video = cv2.VideoCapture('tesla_car_crash.mov')

# pre-trained car classifying data
classifier= 'cars_haar_cascades.xml'

# creating opencv image
# img = cv2.imread(car_img)

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

    # displays images with cars spotted
    cv2.imshow('Car Recognition', grayscale_frame)

    # waits till any key is pressed to close python program
    cv2.waitKey(1)

# converting image to grayscale
# grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# creating car recognition
# car_tracker = cv2.CascadeClassifier(classifier)

# car detection
# car_detector = car_tracker.detectMultiScale(grayscale) 
# print(car_detector)

# drawing boxes around cars that are detected
# for (x, y, w, h) in car_detector:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# displaying image with the cars spotted
# cv2.imshow('Car Recognition', img)

# Waits till any key is pressed to close python program
# cv2.waitKey()

print('successful')