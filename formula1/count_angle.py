from sample_bot_car_TI6 import ImageProcessor, cv2
import numpy as np
import math

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

white_pixel = (255, 255, 255)

def locate_white_pixels(ary_2d):
    return np.where(np.sum(ary_2d, axis=1) == sum(white_pixel))[0]

def get_target(img):
    img = img.copy()
    mask = np.zeros([img.shape[0] + 2, img.shape[1] + 2], np.uint8)
    cv2.floodFill(img, mask, (int(img.shape[1] / 2 + 10), img.shape[0] - 1), white_pixel)

    top_target = locate_white_pixels(img[0])
    if top_target.any():
        x, y = int(np.mean(top_target)), 0

    else:
        left_target, right_target = locate_white_pixels(img[:, 0]), locate_white_pixels(img[:, -1])
        if min(left_target) < min(right_target):
            x, y = 0, int(np.mean(left_target))
        else:
            x, y = img.shape[1] - 1, int(np.mean(right_target))

    return x, y

def put_meta(img, msg):
    target_x, target_y = get_target(img)
    half_x = int(img.shape[1]/2)

    cv2.line(img, (target_x, target_y), (half_x, img.shape[0]), white_pixel, 5)
    cv2.putText(img, msg, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    return img

def get_car_road_angle(img):
    target_x, target_y = get_target(img)
    source_x, source_y = int(img.shape[1]/2), img.shape[0]
    return math.atan2(abs(target_x-source_x), abs(target_y-source_y))

if __name__ == '__main__':
    import sys
    filenames = [line.strip('\n') for line in sys.stdin]
    for filename in filenames:
        draw_direction(filename)
        break