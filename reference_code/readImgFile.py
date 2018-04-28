import cv2
import numpy as np
from PIL import Image


def readimage(path):
    with open(path, "rb") as f:
        return bytearray(f.read())

bytes = readimage("images/flowers007.rgb")

print(len(bytes))
for x in bytes:
	print(x)

image = Image.frombytes('RGB', (352,288), bytes)
image.save("test.jpg")