from PIL import Image, ImageDraw

def make_triangle(pos, length, iterations=8, image_size=(2160, 2160)):
    im = Image.new("RGB", image_size, "white")
    g = ImageDraw.Draw(im)

    init_points = [pos]
    for i in range(iterations):
        print("Iteration {} of {}".format(i+1, iterations))
        next_points = []
        for x, y in init_points:
            g.line([
                (x, y),
                (x+length/2, y+length),
                (x+length, y),
                (x, y)
                ], fill="black")
            next_points.append((x, y))
            next_points.append((x+length/4, y+length/2))
            next_points.append((x+length/2, y))
        length /= 2
        init_points = next_points

    im.save("output.png", "PNG")

make_triangle((10, 10), 2140, iterations=12)

