import math

class imaginary:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def __str__(self):
        realformat = str(int(self.real)) if type(self.real) == float and self.real % 1.0 == 0 else str(self.real)
        imagformat = str(int(self.imag)) if type(self.imag) == float and self.imag % 1.0 == 0 else str(self.imag)
        if self.real == 0:
            if self.imag == 0:
                return realformat
            else:
                return imagformat + "i"
        else:
            if self.imag == 0:
                return realformat
            else:
                if self.imag < 0:
                    return "{} - {}i".format(realformat, imagformat[1:] if abs(self.imag) != 1 else "")
                else:
                    return "{} + {}i".format(realformat, imagformat if self.imag != 1 else "")

    def __repr__(self):
        if self.imag < 0:
            return "{} - {}i".format(self.real, abs(self.imag))
        else:
            return "{} + {}i".format(self.real, self.imag)

    def __eq__(self, other):
        if self.imag == 0 and type(other) in [int, long, float]:
            return other == self.real
        elif self.real == other.real and self.imag == other.imag:
            return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __neg__(self):
        return imaginary(-self.real, -self.imag)

    def __abs__(self):
        return math.sqrt(self.real**2 + self.imag**2)

    def __round__(self, n):
        return imaginary(math.round(self.real, n), math.round(self.imag, n))

    def __floor__(self):
        return imaginary(math.floor(self.real), math.floor(self.imag))

    def __ceil__(self):
        return imaginary(math.ceil(self.real), math.ceil(self.imag))

    def __trunc__(self):
        return imaginary(math.trunc(self.real), math.trunc(self.imag))

    def __add__(self, other):
        if type(other) in [int, long, float]:
            return imaginary(self.real + other, self.imag)
        else:
            return imaginary(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other):
        if type(other) in [int, long, float]:
            return imaginary(self.real - other, self.imag)
        else:
            return imaginary(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other):
        if type(other) in [int, long, float]:
            return imaginary(self.real * other, self.imag)
        else:
            return imaginary((self.real * other.real) - (self.imag * other.imag),
                             (self.real * other.imag) + (self.imag * other.real))
    def __floordiv__(self, other):
        result = self.__div__(other)
        return imaginary(math.floor(result.real), math.floor(result.imag))

    def __div__(self, other):
        if type(other) in [int, long, float]:
            return imaginary(self.real / other, self.imag)
        else:
            stepone = self * other.conjugate()
            steptwo = other.conjugate_squared()
            return imaginary(stepone.real / float(steptwo), stepone.imag / float(steptwo))

    def __pow__(self, other):
        stand = imaginary(self.real, self.imag)
        for i in xrange(other):
            stand = stand * self
        return stand

    def __iadd__(self, other):
        return self.__add__(other)

    def __isub__(self, other):
        return self.__sub__(other)

    def __imul__(self, other):
        return self.__mul__(other)

    def __ifloordiv__(self, other):
        return self.__floordiv__(other)

    def __idiv__(self, other):
        return self.__div__(other)

    def __ipow__(self, other):
        return self.__pow__(other)

    def __int__(self):
        return int(self.real)

    def __long__(self):
        return long(self.real)

    def __float__(self):
        return float(self.real)

    def __copy__(self):
        return imaginary(self.real, self.imag)

    def __deepcopy__(self):
        return self.__copy__()

    def conjugate(self):
        return imaginary(self.real, -self.imag)
    
    def conjugate_squared(self):
        return self.real**2 + self.imag**2
    
    def distance_squared(self, other):
        return (other.real - self.real) * (other.imag - self.imag)

def mandelbrot(position, iterations):
    i = 0
    change = position
    while i < iterations:
        change = change * change + position
        i += 1
        if abs(change) > 2:
            return i
    return i

def julia(position, subposition, iterations):
    i = 0
    change = position
    while i < iterations:
        change = change * change + subposition
        i += 1
        if abs(change) > 2:
            return i
    return 
    
if __name__ == "__main__":

    print "\n----------------------------\n"
    
    compA = imaginary(2, 3)
    compB = imaginary(4, 5)
    resultA = compA * compB

    print "test one: ({}) * ({}) = {}".format(compA, compB, resultA)

    compC = imaginary(4, 2)
    compD = imaginary(3, -1)
    resultB = compC / compD

    print "test two: ({}) / ({}) = {}".format(compC, compD, resultB)

    print "test three:"
    
    mark = imaginary(-0.74434, -0.10772)
    for y in xrange(-25, 25):
        for x in xrange(-50, 50):
            view = imaginary(x/25.0, y/25.0)
            result = mandelbrot(view, 100)
            if result == 100:
                print "X",
            else:
                print " ",
        print

    

        
