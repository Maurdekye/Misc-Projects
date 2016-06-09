from moviepy.editor import *
import copy
import sys
print(sys.argv)
clipfile = sys.argv[1]

print("Loading clip " + clipfile + "...")

clip = VideoFileClip(clipfile)
modifiedclip = VideoFileClip(clipfile)

ratio = clip.w / clip.h

minr = -10
maxr = 10
mini = minr * ratio
maxi = maxr * ratio

def get_complex(x, y):
    r_range = maxr - minr
    i_range = maxi - mini
    small_x = x / r_range
    small_y = y / i_range
    r = small_x + minr
    i = small_y + mini
    return r + i*1j

def get_pixel(cmpx):
    corrected_r = cmpx.real - minr
    corrected_i = cmpx.imag - mini
    approx_x = corrected_r * clip.w
    approx_y = corrected_i * clip.h
    x = int(approx_x)
    y = int(approx_y)
    return x, y

def is_in_bounds(x, y):
    if x < 0 or x >= clip.w:
        return False
    if y < 0 or y >= clip.h:
        return False
    return True
    
equation = lambda x: x**2 + 1

numframes = int(clip.fps * clip.duration)
'''
print("Creating blank clip...")

for f, frame in enumerate(modifiedclip.iter_frames()):
    print("\t\tFrame " + str(f) + "/" + str(numframes))
    for y, row in enumerate(frame):
        for x, pixel in enumerate(row):
            frame[y][x] = [0, 0, 0]
'''
print("Evaluating clip function...")

for f, frame in enumerate(clip.iter_frames()):
    print("\t\tFrame " + str(f) + "/" + str(numframes))
    for y, row in enumerate(frame):
        for x, pixel in enumerate(row):
            cpos = get_complex(x, y)
            newcpos = equation(cpos)
            newx, newy = get_pixel(newcpos)
            if is_in_bounds(newx, newy):
                modifiedclip.get_frame(f)[newy][newx] = pixel

print("Saving clip...")

modifiedclip.write_videofile("output/" + clipfile)