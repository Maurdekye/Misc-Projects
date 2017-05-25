from numbers import Number
from copy import copy

sign_table = {True:1, False:-1}

class IndeterminateResultError(Exception):
  def __init__(self, message):
    super(IndeterminateResultError, self).__init__(message)

class Infinity(Number):
  def __init__(self, sign=True):
    self.sign = sign

  def __new__(self):
    return Infinity(self.sign)

  def __str__(self):
    f = u"\u221E"
    if not sign:
      f = "-" + f
    return f

  def __repr__(self):
    return str(self)

  def __bool__(self):
    return True

  def __len__(self):
    return Infinity()

  def __iter__(self):
    i = 0
    while True:
      yield i
      if self.sign:
        i += 1
      else:
        i -= 1

  def __reversed__(self):
    while True:
      yield copy(self)

  def __contains__(self):
    return True

  # Maths

  def __add__(self, other):
    if other is Infinity:
      if other.sign != self.sign:
        return 0
      else:
        return copy(self)
    if isinstance(other, Number):
      return copy(self)
    else:
      return str(self) + str(other)

  def __sub__(self, other):
    if isinstance(other, Number):
      return self + (other * (-1))
    else:
      raise TypeError()

  def __mul__(self, other):
    if isinstance(other, Number):
      nsign = self.sign
      if other < 0:
        nsign = not nsign
      if other is Infinitesimal:
        return sign_table[nsign]
      elif other == 0:
        return 0
      else:
        return Infinity(nsign)
    else:
      raise TypeError()

  def __truediv__(self, other):
    if isinstance(other, Number):
      nsign = self.sign
      if other < 0:
        nsign = not nsign
      if other is Infinity:
        return sign_table[nsign]
      else:
        return Infinity(nsign)
    else:
      raise TypeError()

  def __floordiv__(self, other):
    return self.__truediv__(other)

  def __mod__(self, other):
    if isinstance(other, Number):
      raise IndeterminateResultError("Unable to perform modulus on infinity")
    else:
      raise TypeError()

  def __divmod__(self, other):
    return self.__mod__(other)

  def __pow__(self, other):
    if isinstance(other, Number):
      if self.sign:
        if other > 0: return Infinity()
        elif other < 0: return 0
        else: return 1
      else:
        if other < 0: return Infinity()
        elif other > 0: return 0
        else: return 1
    else:
      raise TypeError()

  # Reversed Maths

  def __radd__(self, other):
    return self.__add__(other):

  def __rsub__(self, other):
    return self.__sub__(other) * (-1)

  def __rmul__(self, other):
    return self.__mul__(other)

  def __rtruediv__(self, other):
    if isinstance(other, Number):
      nsign = self.sign
      if other < 0:
        nsign = not nsign
      if other is Infinity:
        return Infinity(nsign)
      elif other == 0:
        return 0
      else:
        return Infinitesimal(nsign)
    else:
      raise TypeError()

  def __rfloordiv__(self, other):
    return self.__rtruediv__(self, other)

  def __rmod__(self, other):
    if isinstance(other, Number):
      return other
    else:
      raise TypeError()

  def __rdivmod__(self, other):
    if isinstance(other, Number):
      return abs(other)
    else:
      raise TypeError()

  def __neg__(self):
    return Infinity(not self.sign)

  def __abs__(self):
    return Infinity()

  def __lt__(self, other):
    if isinstance(other, Number):
      return not self.sign
    raise TypeError()

  def __


class Infinitesimal(Number):
  pass