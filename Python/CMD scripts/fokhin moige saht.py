import random

def merge_sort(lst):
    l = len(lst)
    if l == 0: return []
    if l == 1: return lst
    if l == 2:
        if lst[0] > lst[1]: return lst[::-1]
        else: return lst
    mid = l//2
    left = lst[:mid]
    right = lst[mid:]
    left = merge_sort(left)
    right = merge_sort(right)
    fin = []
    for i in range(len(lst)):
        if len(left) == 0:
            fin += right
            break
        elif len(right) == 0:
            fin += left
            break
        elif left[0] < right[0]:
            fin += [left.pop(0)]
        else:
            fin += [right.pop(0)]
    return fin
            

def random_list(ln, mx=100):
    return [random.randrange(mx) for x in range(ln)]

rls = random_list(20)
print(rls)
print(merge_sort(rls))
