import matplotlib.pylab as plt
import cv2
import numpy as np


def interest(img,vertices):
    mask = np.zeros_like(img)
    # channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
def draw(img, lines):
    img = np.copy(img)
    line_img = np.zeros((img.shape[0],img.shape[1],3), dtype = np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1,y1), (x2,y2), (0, 255, 0), thickness=5)

    img = cv2.addWeighted(img, 0.8, line_img, 1, 0.0)
    return img





image = cv2.imread('SIH\Lane_detect\image.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def pro(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
            (100,539),(950,539),(480,290)
        ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 180, 240)

    cropped_image = interest(canny_image,
                    np.array([region_of_interest_vertices], np.int32),)
    lines = cv2.HoughLinesP(cropped_image,
                            rho=2,
                            theta=np.pi/180,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=100)
    image_with_lines = draw(image, lines)
    return image_with_lines

cap = cv2.VideoCapture('SIH\Lane_detect\solidYellowLeft.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    frame1 = pro(frame)
    cv2.imshow('org',frame)
    cv2.imshow('frame', frame1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()