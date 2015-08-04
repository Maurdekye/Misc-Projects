def mandelbrot(re1, im1, re2, im2, iters):
    to_sender = 0
    for i in xrange(iters):
        to_sender = i+1
        if re1**2 + im1**2 > 4: break
        re, im = re1, im1
        re1 = ( re**2 - im**2 ) + re2
        im1 = ( 2 * re * im ) + im2
    return to_sender

if __name__ == '__main__':
    xRange = [(x/20.0)-1 for x in xrange(40)]
    yRange = [(y/20.0)-1.5 for y in xrange(40)]
    for r in xRange:
        for i in yRange:
            mandel = mandelbrot(i, r, 0.3, -0.49, 40)
            if  mandel < 40: print " ",
            else: print "X",
        print

