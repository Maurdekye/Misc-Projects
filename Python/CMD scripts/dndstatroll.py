from random import randrange
def dieroll(): return randrange(6) + 1

def rollstat_normal(null=0):
    return sum([dieroll() for d in xrange(3)])

def rollstat_high_die(die_per_stat=6, debug=False):
    roll = [dieroll() for d in xrange(3)]
    if debug: print 'start ' + str(roll)
    for _ in xrange(die_per_stat):
        die = dieroll()
        if debug: print die,
        for i, d in enumerate(roll):
            if die > d:
                roll[i] = die
                break
    if debug: print '\nend ' + str(roll) + '\n'
    return sum(roll)

def rollstats(method=rollstat_normal, arg=None):
    stats = [
        'STR', 'DEX',
        'CON', 'INT',
        'WIS', 'CHA'
    ]
    to_sender = {stat : 0 for stat in stats}
    for stat in stats:
        if arg == None:
            roll = method()
        else:
            roll = method(arg)
        to_sender[stat] = roll
    return to_sender

def rollstats_high_stat(stat_rolls=12):
    to_sender = rollstats()
    stat_rolls = max(stat_rolls - 6, 0)
    for _ in xrange(stat_rolls):
        roll = rollstat_normal()
        for name, stat in to_sender.iteritems():
            if roll > stat:
                to_sender[name] = roll
                break
    return to_sender

for i in xrange(3):
    print rollstats()
    print
print '---'
for i in xrange(3):
    print rollstats_high_stat()
    print
print '==='
for i in xrange(3):
    print rollstats(rollstat_high_die)
    print
