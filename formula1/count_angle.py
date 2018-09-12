from sample_bot_car_TI6 import ImageProcessor, cv2
import numpy as np
import math

def blur(x):
    return int(x/10)*10

blur = np.vectorize(blur)

def black(x):
    return 255 if x == 70 else x

def most_common(ary):
    count = np.bincount(ary.flatten())
    return np.argmax(count)


def find_target(ary, target_color):
    print(ary.shape)
    print(target_color)
    st, ed = -1, -1
    medians = []
    for i,val in enumerate(ary):
        if st < 0:
            if val == target_color:
                st = i
        else:
            if val == target_color:
                ed = i
            else:
                if ed - st > 2:
                    medians.append(int((st+ed)/2))
                st, ed = -1, -1

    return medians

def draw_direction(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    print(get_car_road_angle(img))
    if img is None:
        return
    blur_img = blur(img)
    target_color = most_common(blur_img)
    top_line = blur_img[0, :].flatten()
    medians = find_target(top_line, target_color)

    half_x = int(blur_img.shape[1]/2)
    target_x = min(medians, key=lambda x: abs(x-half_x))

    cv2.line(img, (target_x, 0), (half_x, blur_img.shape[0]), 255, 5)

    cv2.imwrite('%s_.jpg'%(filename), img)
    print('%s_.jpg'%(filename))

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def get_car_road_angle(track_img):
    track_img = rgb2gray(track_img)
    blur_img = blur(track_img)
    target_color = most_common(blur_img)
    medians = find_target(blur_img[0, :], target_color)

    half_x = int(blur_img.shape[1] / 2)
    target_x = min(medians, key=lambda x: abs(x-half_x))
    y, x = abs(half_x-target_x), blur_img.shape[0]
    return math.atan2(y, x)

#print(draw_direction('images/track_img-20180911-144456-955393current_angle_1.196496.jpg'))

if __name__ == '__main__':
    import sys
    filenames = [line.strip('\n') for line in sys.stdin]
    for filename in filenames:
        draw_direction(filename)
        break