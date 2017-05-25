import random

def to_tuplist(l):
  if l == []: return ()
  return l[0], to_tuplist(l[1:])

def tl_str(tl):
  def tl_str_internal(itl, carry=""):
    if itl == ():
      return carry
    head, tail = itl
    return tl_str_internal(tail, carry + str(head) + ", ")
  return "[" + tl_str_internal(tl)[:-2] + "]"

def tl_len(tl):
  def tl_len_internal(itl, carry=0):
    if itl == ():
      return carry
    _, tail = itl
    return tl_len_internal(tail, carry + 1)
  return tl_len_internal(tl)

def tl_add(tl, item):
  return item, tl

def tl_append(tl, item):
  if tl == ():
    return item, ()
  head, tail = tl
  return head, tl_append(tail, item)

def tl_get_front(tl, index):
  if tl == ():
    raise Exception("Tuplist index out of range")
  head, tail = tl
  if index == 0:
    return head
  return tl_get(tail, index - 1)

def tl_get_back(tl, index):
  if tl == ():
    return False, index
  head, tail = tl
  got, val = tl_get_back(tail, index)
  if got:
    return got, val
  if val == 0:
    return True, head
  return False, val - 1

def tl_get(tl, index):
  if index < 0:
    got, val = tl_get_back(tl, -index-1)
    if not got:
      raise Exception("Tuplist index out of range")
    return val
  return tl_get_front(tl, index)

def tl_set(tl, index, val):
  if tl == ():
    raise Exception("Tuplist index out of range")
  head, tail = tl
  if index == 0:
    return val, tail
  return head, tl_set(tail, index-1, val)

def tl_insert(tl, index, val):
  if tl == ():
    return val, ()
  if index == 0:
    return val, tl
  head, tail = tl
  return head, tl_set(tail, index-1, val)

def tl_remove(tl, val, amount = -1):
  if tl == ():
    return tl
  if amount == 0:
    return t1
  head, tail = tl
  if head == val:
    return tl_remove(tail, val, amount - 1)
  return head, tl_remove(tail, val, amount)

def tl_delete(tl, index):
  if tl == ():
    raise Exception("Tuplist index out of range")
  head, tail = tl
  if index == 0:
    return tail
  return head, tl_delete(tail, index-1)

def tl_contains(tl, val):
  if tl == ():
    return False
  head, tail = tl
  if head == val:
    return True
  return tl_contains(tail, val)

def tl_startswith(tl, stl):
  if stl == ():
    return True
  if tl == ():
    return False
  head, tail = tl
  shead, stail = stl
  if head != shead:
    return False
  return tl_startswith(tl, stl)

def tl_sublist(tl, stl):
  if tl == ():
    return False
  if tl_startswith(tl, stl):
    return True
  _, tail = tl
  return tl_sublist(tail, stl)

def tl_dropfront(tl, amount):
  if tl == ():
    return ()
  if amount == 0:
    return tl
  _, tail = tl
  return tl_dropfront(tail, amount)

def tl_dropend(otl, amnt):
  def tl_dropend_internal(tl, amount):
    if tl == ():
      return False, amount
    head, tail = tl
    got, val = tl_dropend_internal(tail, amount)
    if got:
      return True, (head, val)
    if val == 0:
      return True, ()
    return False, val - 1
  return tl_dropend_internal(otl, amnt)[0]

def tl_slice(tl, start, end=-1):
  l = tl_len(tl)
  if start < 0:
    start = l + start
  if start < 0:
    raise Exception("Tuplist index out of range")
  if end < 0:
    end = l + end
  if end < 0:
    raise Exception("Tuplist index out of range")
  return tl_dropend(tl_dropfront(tl, start), end)

def tl_map(tl, func):
  if tl == ():
    return ()
  head, tail = tl
  return func(head), tl_map(tail, func)

def tl_filter(tl, func):
  if tl == ():
    return ()
  head, tail = tl
  if func(head):
    return head, tl_filter(tail, func)
  return tl_filter(tail, func)

# l = [random.randrange(10) for i in range(random.randrange(0, 4))]

# print(l)
# tl = to_tuplist(l)
# print(tl)
# print(tuplist_str(tl))
# print(len(l))
# print(tuplist_len(tl))

a = to_tuplist([2**x for x in range(12)])
print()
print(a)
print(tl_str(a))
print()
print(tl_get_inverse(a, 0))
print(tl_get_inverse(a, 1))
print(tl_get_inverse(a, 3))
print(tl_get_inverse(a, 9))
print(tl_get_inverse(a, 10))
print(tl_get_inverse(a, 11))
print(tl_get_inverse(a, 12))