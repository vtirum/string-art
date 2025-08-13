import cv2 
import numpy as np  
import requests


def nothing(x): pass

# read in image
'''
url = "https://photos.fife.usercontent.google.com/pw/AP1GczOnQp_fvF2jubmdnxONyOjp0UsaTJbzeyJzKhHE2Lvcn8NVRTVxlQSC=w519-h923-s-no-gm?authuser=1"
resp = requests.get(url)
img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
image = cv2.imdecode(img_array, cv2.IMREAD_COLOR_BGR)
'''
image = cv2.imread('PXL_20240601_044143906.MP.jpg') 

preview = None

if len(image.shape) == 3:
    preview = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
else:
    preview = image.copy()

height, width = preview.shape

# set window size
size = 600 
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', width, height)

max_radius = min(width, height) // 2
initial_radius = max_radius // 2
cv2.createTrackbar('Radius', 'image', initial_radius, max_radius, nothing)

cv2.createTrackbar('Center X', 'image', width // 2, width, nothing)

cv2.createTrackbar('Center Y', 'image', height // 2, height, nothing)

while True:
    radius = cv2.getTrackbarPos('Radius', 'image')
    center_x = cv2.getTrackbarPos('Center X', 'image')
    center_y = cv2.getTrackbarPos('Center Y', 'image')

    min_x = radius
    max_x = width - radius
    center_x = max(min_x, min(center_x, max_x))
    cv2.setTrackbarPos('Center X', 'image', center_x)

    min_y = radius
    max_y = height - radius
    center_y = max(min_y, min(center_y, max_y))
    cv2.setTrackbarPos('Center Y', 'image', center_y)

    display = preview.copy()

    mask = np.zeros_like(preview)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    result = cv2.bitwise_and(display, mask)

    cv2.circle(result, (center_x, center_y), radius, 2)
    
    cv2.imshow('image', result)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cropped = cv2.bitwise_and(preview, mask)
        cropped = cropped[center_y - radius:center_y + radius, 
                  center_x - radius:center_x + radius]
        cv2.imwrite('cookie.png', cropped)

cv2.destroyAllWindows()
        


