import random

def rnumrow(size, prow=[]):
  if size == 0:
    return prow
  return rnumrow(size - 1, prow + [random.randrange(10, 100)])

def rnumgrid(size, pgrid=[], csize=None):
  if csize == None:
    csize = size
  if csize == 0:
    return pgrid
  return rnumgrid(size, pgrid + [rnumrow(size)], csize-1)

def pgrid(grid, x=0, y=0):
  if x == len(grid):
    print
    x = 0
    y += 1
  if y == len(grid[0]):
    return
  print grid[x][y],
  pgrid(grid, x+1, y)
  
def shiftback(grid, x, y):
  if x == len(grid):
    return grid
  grid[x-1][y] = grid[x][y]
  return shiftback(grid, x+1, y)

def shiftrow(grid, row):
  first = grid[0][row]
  grid = shiftback(grid, 1, row)
  grid[len(grid[0])-1][row] = first
  return grid

def examinerow(grid, row, crow=0):
  if crow == row:
    return examinerow(grid, row, crow+1)
  if crow == len(grid):
    return False
  if comparerows(grid, row, crow):
    return True
  return examinerow(grid, row, crow+1)

def comparerows(grid, y1, y2, x=0):
  if x == len(grid[0]):
    return False
  if grid[x][y1] == grid[x][y2]:
    return True
  return comparerows(grid, y1, y2, x+1)

def examinegrid(grid, row=0):
  if row == len(grid):
    return grid, 0, True
  if examinerow(grid, row):
    return shiftrow(grid, row), row, False
  return examinegrid(grid, row+1)
  
def iterexamine(grid, lastr=-1, rcount=0):
  ngrid, shrow, isend = examinegrid(grid)
  if isend:
    return grid, True
  if shrow == lastr:
    rcount += 1
  else:
    rcount = 1
  if rcount == len(grid):
    return grid, False
  lastr = shrow
  return iterexamine(grid, lastr, rcount)
  

g = rnumgrid(5)
pgrid(g)
print
ng, slv = iterexamine(g)
if slv:
  pgrid(ng)
else:
  print "couldnt solve"
