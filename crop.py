import cv2 
import numpy as np  


def nothing(x): pass

# read in image
image = cv2.imread('lucki.png') 

# set window size
size = 600 
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', size, size)

height, width = image.shape[:2]
r = min(height, width) // 2

# create trackbars for center position
cv2.createTrackbar('Center Y', 'image', (height - 2 * r) // 2, height - 2 * r, nothing)

while True:
    cx = r
    cy = cv2.getTrackbarPos('Center Y', 'image') + r

    # preview grayscale image with circle 
    preview = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    cv2.circle(preview, (cx, cy), r, 255, 2)
    
    cv2.imshow('image', preview)

    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # enter key
        break
    elif key == 27:  # esc key
        exit()
cv2.destroyAllWindows()

# create mask with circle
mask = np.zeros((height, width), dtype=np.uint8)
mask = cv2.circle(mask, (cx, cy), r, 255, -1)
mask = cv2.resize(mask, image.shape[1::-1])

# apply mask to image
result = cv2.bitwise_and(image, image, mask=mask)
result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

#crop the result to the circle area
result = result[cy - r:cy + r, cx - r:cx + r]

cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.resizeWindow('result', size, size)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save the cropped image
cv2.imwrite('cropped_image.png', result)