import time

def printgrid(*grids):
    depth = max([len(grid) for grid in grids])
    interbuffer = [0 for _ in grids]
    for y in range(depth):
        for i, grid in enumerate(grids):
            if y == len(grid):
                interbuffer[i] = len(grid)
            elif y <= len(grid):
                for e in grid[y]: print(e, end="")
            print(' '*(interbuffer[i]*2), end="")
        print
    print

def gridclone(grid):
    new = []
    for r in grid:
        new += [list(r)]
    return new

def grid_equal(grid1, grid2):
    if len(grid1) != len(grid2):
        return False
    if len(grid1[0]) != len(grid2[0]):
        return False
    for x, row in enumerate(grid1):
        for y, item in enumerate(row):
            if item != grid2[x][y]:
                return False
    return True        

def plist_grid(pointlist):
    xs = [l[0] for l in pointlist]
    ys = [l[1] for l in pointlist]
    smallx, largex = min(xs), max(xs)
    smally, largey = min(ys), max(ys)
    rangex = abs(largex - smallx)
    rangey = abs(largey - smally)
    xs = [x - smallx for x in xs]
    ys = [y - smally for y in ys]
    final = [[0 for x in range(rangey+1)] for y in range(rangex+1)]
    for x, y in zip(xs, ys):
        final[x][y] = 1
    return final

def transverse(grid):
    l = min([len(x) for x in grid])
    return [[grid[x][y] for x in range(len(grid))] for y in range(l)]

def mirror(grid):
    newgrid = [[0 for x in range(len(grid[0]))] for y in range(len(grid))]
    for xr, x in zip(range(len(grid))[::-1], range(len(grid))):
        for yr, y in zip(range(len(grid[0])), range(len(grid[0]))):
            newgrid[x][y] = grid[xr][yr]
    return newgrid

def rotate(grid):
    return mirror(transverse(grid))

def shape_equal(grid1, grid2):
    for i in range(2):
        for x in range(4):
            if grid_equal(grid1, grid2):
                return True
            grid2 = rotate(grid2)
        grid2 = mirror(grid2)
    return False
            
def expand(expansion):
    final = []
    for x, y in expansion:
        checks = [(x+a, y) for a in [-1, 1]] + [(x, y+a) for a in [-1, 1]]
        for nx, ny in checks:
            cur = gridclone(expansion)
            if [nx, ny] in cur: continue
            cur += [[nx, ny]]
            final += [cur]
    return final

def clean_old(gridlist):
    if len(gridlist) <= 1: return gridlist
    for i, grid in enumerate(gridlist[1:]):
        cgrid1, cgrid2 = plist_grid(gridlist[0]), plist_grid(grid)
        if shape_equal(cgrid1, cgrid2):
            return clean(gridlist[1:])
    return [gridlist[0]] + clean(gridlist[1:])

def clean(pointlist, debug=False):
    gridlist = [plist_grid(pl) for pl in pointlist]
    final = []
    for x, grid1 in enumerate(gridlist):
        if debug: print('checked grid {}, was'.format(x+1), end="")
        for y, grid2 in enumerate(gridlist[x+1:]):
            if shape_equal(grid1, grid2):
                if debug: print('copy of grid ' + str(y))
                break
        else:
            if debug: print('by itself')
            final += [grid1]
    return final

if __name__ == '__main__':
    start = time.clock()
    genlaps = []
    grids = [[[0, 0]]]
    for i in range(5):
        adds = []
        for g in grids:
            adds += expand(g)
        grids += adds
        genlaps += [time.clock()]
        print('expanded stage ' + str(i + 1))

    print('finished expanding')

    print('cleaning {} grids...'.format(len(grids)))

    dirtylen = len(grids)
    grids = clean(grids, 1)
    end = time.clock()
    print('cleaned')

    for i, g in enumerate(grids):
        print(i)
        printgrid(g)

    with open('grid6.txt', 'w') as f:
        for g in grids:
            for row in g:
                for item in row:
                    f.write(str(item) + ' ')
                f.write('\n')
            f.write('\n')

    print('Total of {} grids in this set.'.format(len(grids)))
    print('It took a total of {} seconds to generate every solution, with the splits as follows:'.format(genlaps[-1] - start))
    print('Generation 1: {} seconds'.format(genlaps[0] - start))
    for i, t in enumerate(genlaps[1:]):
        print('Generation {}: {} seconds'.format(i + 2, genlaps[i] - genlaps[i-1]))
    print('It took a total of {} seconds to clean the grid set out: {} out of {} grids were removed in the process.'.format(end - start, dirtylen-len(grids), dirtylen))
