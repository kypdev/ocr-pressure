import os
import shutil
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
from PIL import Image


# move file
def move_image():
    original = r'../media/p.jpg'
    target = r'images/p.jpg'
    shutil.move(original,target)


# Crop sys image
def crop_sys_image():
    im = Image.open("images/p.jpg")

    xcenter = im.width/2
    ycenter = im.height/2

    x1 = xcenter - 180
    y1 = ycenter - 375
    x2 = xcenter + 220
    y2 = ycenter - 180

    sys = im.crop((x1, y1, x2, y2))

    sys.save("images/sys" + ".jpg", "JPEG")
    # sys.show()


# Crop dia image
def crop_dia_image():
    im = Image.open("images/p.jpg")

    xcenter = im.width/2
    ycenter = im.height/2

    x1 = xcenter - 50
    y1 = ycenter - 190
    x2 = xcenter + 220
    y2 = ycenter + 15

    sys = im.crop((x1, y1, x2, y2))

    sys.save("images/dia" + ".jpg", "JPEG")


def main():
    move_image()
    crop_sys_image()
    crop_dia_image()


if __name__ == '__main__':
    main()