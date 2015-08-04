import pygame as pyg
import sys
import math
import re

class cached_search:
    def __init__(this, term, results):
        this.term = term
        this.results = results

    def __eq__(this, other):
        return this.term == other

def drawtext(scrn, text, pos, size=25, color=(0,0,0)):
    scrn.blit(pyg.font.SysFont("monospace", size).render(text, 1, color), pos)

def trim(text):
    return re.search("^\s*(.*)\s*$", text).group(0)

def search(criteria, collection):
    returning = []
    for item in collection:
        if criteria in item:
               returning += [item]
    return returning

def re_contains(pattern, check):
    items = re.findall(pattern, check)
    return len(items) != 0

def search_update():
    global term, cache, database, textclock
    textclock = 0
    searchby = trim(term).lower()
    results = []
    print("Checking search term '{}'".format(searchby))
    if searchby == "":
        print("term is blank")
        return results
    for ch in cache:
        if ch == searchby:
            print("found in cache")
            results = ch.results
            break
    else:
        for ch in reversed(cache):
            if re_contains("^" + re.escape(ch.term), searchby):
                print("extension of previous cache, '{}'".format(ch.term))
                results = search(searchby, ch.results)
                cache += [cached_search(searchby, results)]
                break
        else:
            print("not in cache, doing default search.")
            results = search(searchby, database)
            cache += [cached_search(searchby, results)]
    print("finished looking up term")
    return results
             
datafile = None if len(sys.argv) < 2 else sys.argv[1]
if datafile == None:
    input("No data file provided.")
    sys.exit()
    
pyg.init()

pressing = {i : False for i in xrange(512)}
clock = 0
WIDTH, HEIGHT = 800, 800
scrn = pyg.display.set_mode((WIDTH, HEIGHT))
errmsg = "Error!"
errlen = int((WIDTH / len(errmsg)) / 0.6)
textclock = 0
typerange = range(92, 123) + [32]
term = ""
results = []
cache = []
prereqs = [l for l in "abcdefghijklmnopqrstuvwxyz"]
topsize = 50

print("Loading database...")
database = []
with open(datafile, "r") as f:
    database = [word[:-1] for word in f]
print("Done; {} items.".format(len(database)))
print("Caching {} prerequesites...".format(len(prereqs)),)
for i, p in enumerate(prereqs):
    cache += [cached_search(p, search(p, database))]
    print(i+1,)
print("\nDone.")

try:
    while True:
        clock += 1
        textclock += 1
        for ev in pyg.event.get():
            if ev.type == 12:
                pyg.quit()
                sys.exit()
            elif ev.type == 2:
                if ev.key == 27:
                    pyg.quit()
                    sys.exit()
                elif ev.key in typerange:
                    if pressing[303] or pressing[304]:
                        term += chr(ev.key).upper()
                    else:
                        term += chr(ev.key)
                    results = search_update()
                elif ev.key == 8:
                    term = term[:-1] if len(term) > 0 else ""
                    results = search_update()
                pressing[ev.key] = True
            elif ev.type == 3: pressing[ev.key] = False
            elif ev.type == 5: pressing[ev.button] = True
            elif ev.type == 6: pressing[ev.button] = False
            
        scrn.fill((255, 255, 255))

        drawtext(scrn, term + ("_" if textclock % 1000 < 500 else ""), (10, 10), topsize)
        drawtext(scrn, str(len(results)), (WIDTH - (len(str(len(results)))*(0.6*topsize) + 10), 10), topsize)
        pyg.draw.rect(scrn, (192, 10, 10), (10, topsize+20, WIDTH-20, max(2, topsize/10)))
        for i, t in enumerate(results[:min(23, len(results))]):
            drawtext(scrn, t, (30, (topsize+40) + i * 30), 25)
        pyg.display.update()
except Exception as ex:
    import traceback
    traceback.print_exc()
    while True:
        for ev in pyg.event.get():
            if ev.type == 12:
                pyg.quit()
                sys.exit()
        clock += 1
        scrn.blit(pyg.font.SysFont("monospace", errlen).render(errmsg, 10, (math.sin(clock/40.0)*100 + 154, 0, 0)), (10, HEIGHT/2 - (errlen/2)))
        pyg.display.update()
