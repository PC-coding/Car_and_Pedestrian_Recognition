import cv2

#car image
car_img = 'car_traffic.jpg'

# pre-trained car classifying data
classifier= 'cars_haar_cascades.xml'

# creating opencv image
img = cv2.imread(car_img)

# creating car recognition
car_detector = cv2.CascadeClassifier(classifier)

# converting image to grayscale
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# displaying image with the cars spotted
cv2.imshow('Car Recognition', img)

# Waits till any key is pressed to close python program
cv2.waitKey()

print('successful')