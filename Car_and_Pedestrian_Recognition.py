import cv2

# car image
car_img = 'car_traffic.jpg'

# pre-trained car classifying data
classifier= 'cars_haar_cascades.xml'

# creating opencv image
img = cv2.imread(car_img)

# converting image to grayscale
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# creating car recognition
car_tracker = cv2.CascadeClassifier(classifier)

# car detection
car_detector = car_tracker.detectMultiScale(grayscale) 
print(car_detector)

# drawing boxes around cars that are detected
for (x, y, w, h) in car_detector:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# displaying image with the cars spotted
cv2.imshow('Car Recognition', img)

# Waits till any key is pressed to close python program
cv2.waitKey()

print('successful')