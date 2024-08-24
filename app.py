from flask import testing
import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2
import pandas

WINDOWSIZEX= 640
WINDOWSIZEY= 480

BOUNDRYINC= 5
WHITE= (255, 255, 255)
BLACK= (0, 0, 0)
RED= (255, 0, 0)

IMAGESAVE= False

MODEL= load_model("best_model.h5")

LABELS= {0:"Zero", 1:"One", 2:"Two", 3:"Three", 4:"Four", 5:"Five", 6:"Six", 7:"Seven", 8:"Eight", 9:"Nine"}



#initilize our pygame
pygame.init()

#Font= pygame.font.Font("freesansbold.tff", 18)   #optional
DISPLAYSURF= pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
#WHILE_INT= DISPLAYSURF.mp_rgb(WHITE)
#pygame.display.set_caption("Digit Board")

iswriting= False
number_xcod= []
number_ycod= []
img_cnt= 1
PREDICT =True

while True:
    for event in pygame.event.get():
        if event.type== QUIT:
            pygame.quit()
            sys.exit()

        if event.type==MOUSEMOTION and iswriting:
            xcod, ycod= event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcod, ycod), 4, 0)
            number_xcod.append(xcod)
            number_ycod.append(ycod)

        if event.type== MOUSEBUTTONDOWN:
            iswriting= True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            if number_xcod and number_ycod:
                number_xcod = sorted(number_xcod)
                number_ycod = sorted(number_ycod)

                rect_min_x = max(number_xcod[0] - BOUNDRYINC, 0)
                rect_max_x = min(WINDOWSIZEX, number_xcod[-1] + BOUNDRYINC)
                rect_min_y = max(number_ycod[0] - BOUNDRYINC, 0)
                rect_max_y = min(number_ycod[-1] + BOUNDRYINC, WINDOWSIZEY)

                number_xcod = []
                number_ycod = []

                img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x: rect_max_x, rect_min_y: rect_max_y].T.astype(np.float32)

                if IMAGESAVE:
                    cv2.imwrite("image.png", img_arr)
                    img_cnt += 1

                if PREDICT:
                    image = cv2.resize(img_arr, (28, 28))
                    image = np.pad(image, (10, 10), 'constant', constant_values=0)
                    image = cv2.resize(image, (28, 28)) / 255

                    label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])

                    textSurface = pygame.font.Font(None, 18).render(label, True, RED, WHITE)
                    textRecObj = textSurface.get_rect()
                    textRecObj.left, textRecObj.bottom = rect_min_x, rect_max_y

                    DISPLAYSURF.blit(textSurface, textRecObj)

        if event.type == KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)

    pygame.display.update()