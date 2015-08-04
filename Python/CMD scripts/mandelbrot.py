def mandelbrot(re1, im1, re2, im2, iters):
    to_sender = 0
    for i in range(iters):
        to_sender = i+1
        if re1**2 + im1**2 > 4: break
        re, im = re1, im1
        re1 = ( re**2 - im**2 ) + re2
        im1 = ( 2 * re * im ) + im2
    return to_sender

xRange = [(x/20.0)-1 for x in range(40)]
yRange = [(y/20.0)-1 for y in range(40)]
for r in xRange:
    for i in yRange:
        mandel = mandelbrot(i, r, 0.3, -0.49, 40)
        if  mandel < 40: print(" ", end="")
        else: print("X", end="")
    print()
