white = (255, 255, 255)
light = (196, 196, 196)
grey = (128, 128, 128)
dark = (64, 64, 64)
black = (0, 0, 0)

red = (230, 40, 40)
dark_red = (120, 10, 10)
light_red = (255, 160, 160)
max_red = (255, 0, 0)

blue = (40, 40, 230)
dark_blue = (10, 10, 120)
light_blue = (160, 160, 255)
max_blue = (0, 0, 255)

green = (40, 230, 40)
dark_green = (10, 120, 10)
light_green = (160, 255, 160)
max_green = (0, 255, 0)

purple = (230, 40, 230)
cyan = (40, 230, 230)
yellow = (230, 230, 40)

orange = (255, 160, 0)

def combine(*colors):
    sums = [[col[i] for col in colors] for i in xrange(3)]
    return tuple([int(sum(code) / len(code)) for code in sums])

'''
for i in xrange(len(collas)):
    start = (i / float(len(collas))) * HIH
    end = ((i + 1) / float(len(collas))) * HIH
    pyg.draw.rect(scrn, collas[i], [0, start, WID, end])
'''
