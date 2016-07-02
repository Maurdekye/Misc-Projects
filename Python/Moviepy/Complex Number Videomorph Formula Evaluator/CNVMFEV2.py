import cv2
import numpy as np
import sys
import time
import copy

if len(sys.argv) <= 1:
    print("Give a video file to modify")
    sys.exit()

vidfile = sys.argv[1]
outputfile = "output/" + vidfile

video = cv2.VideoCapture(vidfile)

w, h = int(video.get(3)), int(video.get(4))
ratio = w / h
minr = -1
maxr = 1
mini = minr * ratio
maxi = maxr * ratio
r_range = maxr - minr
i_range = maxi - mini

def get_complex(x, y):
    small_x = (x/w) * r_range
    small_y = (y/h) * i_range
    r = small_x + minr
    i = small_y + mini
    return r + i*1j

def get_pixel(cmpx):
    corrected_r = cmpx.real - minr
    corrected_i = cmpx.imag - mini
    approx_x = (corrected_r / r_range) * w
    approx_y = (corrected_i / i_range) * h
    x = int(approx_x)
    y = int(approx_y)
    return x, y

def is_in_bounds(x, y):
    if x < 0 or x >= w:
        return False
    if y < 0 or y >= h:
        return False
    return True
    
equation = lambda x: x**2 + 1

fc = cv2.VideoWriter_fourcc(*' ???')
outputvideo = cv2.VideoWriter(outputfile, fc, 30.0, (w, h))
blank_frame = np.array([np.array([np.array([0, 0, 0], dtype="uint8") for x in range(w)], dtype="uint8") for y in range(h)], dtype="uint8")
fi = 1

print("Processing video...")

while True:
    has_frame, frame = video.read()
    if not has_frame:
        break

    print("Translating frame", fi)

    newframe = copy.deepcopy(blank_frame)
    for y in range(h):
        for x in range(w):
            comp = get_complex(x, y)
            modified = equation(comp)
            newx, newy = get_pixel(modified)
            if is_in_bounds(newx, newy):
                newframe[y][x] = frame[newy][newx]

    outputvideo.write(newframe)

    cv2.imshow(vidfile, newframe)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    fi += 1

video.release()
outputvideo.release()
cv2.destroyAllWindows()