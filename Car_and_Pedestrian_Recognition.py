import cv2

#car image
car_img = 'car_traffic.jpg'

# pre-trained car classifying data
car_detector = 'cars_haar_cascades.xml'

# creating opencv image
img = cv2.imread(car_img)



print('successful')