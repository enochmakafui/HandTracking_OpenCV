
# this is the package the opens the webcam gets picture and refreshes to get the next one
# following how the loop runs but because it's a loop everything runs fast and it makes it looks
# like it is a video
# open cv draws the lines to connect the landmark when mediapipe detects the hands
import cv2

# this is a model trained by google that takes the image and detects the hands
# and also detects the landmarks,the 21 landmarks
import mediapipe as mp

# imported to help calculate the refresh rate or the pictures
import time

# Here we are setting up open cv to open the webcam , if you have am external web cam connected to the pc
# then you might pass 1 as the argument but if you dont have then the default webcam connected to the pc is 0
cap = cv2.VideoCapture(0)


# this gets the hands module
# gives me access to the tools that can help with detecting the hands
mpHands = mp.solutions.hands

# Inside the hands module there is a class Called hands
# so here we are creating an objects from that class
hands = mpHands.Hands()


mpDraw = mp.solutions.drawing_utils

p_time = 0
c_time = 0


#The loops that runs when open cv is getting the images to it makes everything look like a video

# About Images

# An image is a grid of color numbers.
# To a computer, a picture is nothing but a big table of pixels, and each pixel is just a few numbers saying how much blue, green and red it has.
# For example one pixel might be [200, 50, 90] meaning blue=200, green=50, red=90.
# A 640×480 frame is just a 640×480 grid of those triples.
# That grid of numbers is img. There’s no separate “image” object floating around — the image you see on screen is only what those numbers look like when a monitor lights up pixels for you.


while True:
    # the part where open cv is getting the images , if it gets an image it returns true which is stored in the success and
    # if it returns the image , it is stored in the img
    success, img = cap.read()

    # for open cv the pixels are in [B G R ] but in mediapipe the pixels are stored in [R G B]
    # so the cvtColor converts the [B G R ] into the [R G B ] format so that media pipe can work with
    # this is stored in the imgRGB

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    # The hands object now gives us access to the process method which takes the convert pixel color grid
    # to start the detection. 
    results = hands.process(imgRGB)



    # The multi hand landmarks returns a list when it detects a hand.
    # The list contains the x,y and z positions of the landmarks on the hand.
    # When it doesn't detect a hand, it returns none and when it does it returns the array which contains the landmarks
    # This block of code is checking whether there is an landmark or whether a hand is detected.
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # enumerate will give access to the index and the landmark
            for id, lm in enumerate(handLms.landmark):

                #  this is used to detect how tall, wide and the color channel in the image
                h, w, c = img.shape

                # this is used to get the actual position of the landmarks pixel , or
                # the pixel on which the landmark is

                cx, cy = int(lm.x * w), int(lm.y * h)

                # This draws a circle on the wrist landmark
                # if id == 0:
                #     cv2.circle(img, (cx,cy), 25, (255,0,255), cv2.FILLED)
                # draws a rectangle.
                # if id == 1:
                #     s = 25  # half the square's size
                #     cv2.rectangle(img, (cx - s, cy - s), (cx + s, cy + s), (255, 0, 0), cv2.FILLED)
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)



    c_time = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time

    # This is used to put the text unto the  image
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
