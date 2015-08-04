import sys
from datetime import datetime

# ----- Functions -----

def rlen(iterable): return range(len(iterable))
def xrlen(iterable): return xrange(len(iterable))

def trim(txt):
    for l in txt:
        if l not in [' ', '\n']: break
    else: return ''
    for i in xrlen(txt):
        if txt[i] not in [' ', '\n']:
            txt = txt[i:]
            break
    for i in rlen(txt)[::-1]:
        if txt[i] not in [' ', '\n']:
            txt = txt[:i+1]
            break
    return txt

def removefrom(txt, rid):
    if len(txt) < len(rid): return txt
    if len(txt) == len(rid):
        if txt == rid: return ''
        else: return txt
    while True:
        for l in xrange(len(txt)-len(rid) + 1):
            if rid == txt[l:l+len(rid)]:
                txt = txt[:l] + txt[l+len(rid):]
                break
        else: break
    return txt

def remall(txt, rids):
    for r in rids:
        txt = removefrom(txt, r)
    return txt

def cleanup(inline, caps=False):
    a = ''
    if caps: a = [remall(w, ['\n', '\t']) for w in trim(inline).split(' ')]
    else: a = [remall(w, ['\n', '\t']) for w in trim(inline.lower()).split(' ')]
    return [w for w in a if w != '']

def tabspace(txt, spacing, cutoff=False):
    txt = txt + ' '*(max(spacing - len(txt), 0))
    if spacing < len(txt) and cutoff: txt = txt[:spacing]
    return txt

def ifstring(inval):
    if type(inval) == type(''):
        return "'{}'".format(inval)
    return str(inval)

def repack(d):
    return [(a, d[a]) for a in d]

def numparse(inval):
    ret = inval
    for test in [int, float]:
        try:
            ret = test(inval)
            return ret
        except ValueError: pass
    return ret
            
def prettyprint(ls):
    if len(ls) == 0:
        return 'Nothing,'
    if len(ls) == 1:
        return str(ls[0]) + ','
    if len(ls) == 2:
        return str(ls[0]) + ' and ' + str(ls[1]) + ','
    fin = ''
    for w in list(ls)[:-1]:
        fin += str(w) + ', '
    return fin + 'and ' + str(ls[-1]) + ','

def YorN(mismessage="Sorry, I didn't understand that. Please put 'y' or 'n'."):
    while True:
        ans = trim(raw_input()).lower()
        if ans not in ['y', 'n']: print mismessage
        else: return {'y':True, 'n':False}[ans]

class Command:
    name = 'null'
    helptext = 'No helper text available.'
    def __init__(this, parentdata, parentcoms):
        this.parentdata = parentdata
        this.parentcoms = parentcoms
    
    def __str__(this):
        return this.helptext

    def activate(this, args, caps):
        pass

welcome_msg = """Welcome to Info Manager 2013!
Type 'help' for a list of commands.
"""

global stdin, STDIN, data, data_old, commit, commands
stdin = ['> ']
STDIN = ['> ']
commit = {}
data = {}
data_old = {}
commands = []

# ----- Commands -----

class helper(Command):
    name = 'help'
    helptext = """help - Gives help for all commands.
help <command> - Gives help for the specific command."""
    def activate(this, args, caps):
        if len(args) == 0:
            for c in this.parentcoms:
                print c
            print "\nUse the '-r' tag to perform it on the most recently returned dataset."
        else:
            for c in this.parentcoms:
                if c.name == args[0]:
                    print c
                    break
            else:
                print "No command by the name '{}' was found.".format(args[0])
commands += [helper(data, commands)]
class lister(Command):
    name = 'list'
    helptext = '''list - lists all commands.
list data -  lists all data stored.
list numerical - lists all data points with numerical values.
list text - lists all data points with text values.'''
    def activate(this, args, caps):
        if len(args) == 0:
            print 'Current commands are:'
            for c in this.parentcoms[:-1]:
                print c.name + ',',
            print 'and ' + this.parentcoms[-1].name
        elif args[0] in ['data', 'numerical', 'text']:
            if len(this.parentdata) == 0: print 'No data is currently stored.'
            else:
                for d in this.parentdata:
                    val = this.parentdata[d]
                    if args[0] == 'numerical':
                        if type(val) in [type(1), type(1.5)]:
                            print "{} : {}".format(d, val)
                            commit[d] = val
                    elif args[0] == 'text':
                        if type(val) == type(''):
                            print "{} : {}".format(d, val)
                            commit[d] = val
                    else:
                        print "{} : {}".format(d, val)
                        commit[d] = val
commands += [lister(data, commands)]
class getdata(Command):
    name = 'get'
    helptext = """get <datapoint> - Gets the data by the name of the point."""
    def activate(this, args, caps):
        if len(args) == 0: print 'Need at least one argument.'
        else:
            for d in this.parentdata:
                if d == args[0]:
                    print "Datapoint '{}' has a value of {}.".format(d, ifstring(this.parentdata[d]))
                    break
            else:
                print "Couldn't find a datapoint with the name '{}'.".format(args[0])
commands += [getdata(data, commands)]
class setdata(Command):
    name = 'set'
    helptext = "set <datapoint> <value> - Set the datapoint to the specified value."
    def activate(this, args, caps):
        if len(args) in [0, 1]: print 'Need at least two arguments.'
        else:
            for d in this.parentdata:
                if d == args[0]:
                    print """Datapoint '{}' already exists, and has a value of {}.
Do you want to overwrite it? (y/n)""".format(d, ifstring(this.parentdata[d]))
                    if YorN():
                        this.parentdata[d] = numparse(args[1])
                        print 'Overwritten.'
                    break
            else:
                args[1] = numparse(args[1])
                print "Creating new datapoint '{}', with value {}.".format(args[0], ifstring(args[1]))
                this.parentdata[args[0]] = args[1]
commands += [setdata(data, commands)]
class remdata(Command):
    name = 'remove'
    helptext = 'remove <datapoint> - Removes the datapoint from the list of data.'
    def activate(this, args, caps):
        if args == []: print 'Need an argument.'
        else:
            if args[0] in this.parentdata:
                del this.parentdata[args[0]]
                print "Datapoint '{}' removed.".format(args[0])
            else:
                print "Could not find a datapoint by the name '{}'.".format(args[0])
commands += [remdata(data, commands)]
class search(Command):
    name = 'search'
    helptext = """search <query> - Displays names of all datapoints that fit the query.
search -v <query> - Displays datapoints and their values with values that fit the query."""
    def activate(this, args, caps):
        if args == []: print "Need an argument."
        elif args[0] == '-v':
            if len(args) == 1:
                print "Need a second argument with the '-v' tag."
            else:
                if this.parentdata == []: print 'No data to search in.'
                points = []
                for d in this.parentdata:
                    val = str(this.parentdata[d])
                    for l in xrange(len(val)-len(args[1]) + 1):
                        if args[1] == val[l:l+len(args[1])]:
                            points += [d]
                            break
                if points == []: print 'No values were found to fit that search query.'
                else:
                    commit = {p : this.parentdata[p] for p in points}
                    print "The search returned the following:"
                    for p in points:
                        print p + " : " + str(this.parentdata[p])
        else:
            if this.parentdata == []: print 'No data to search in.'
            points = []
            for d in this.parentdata:
                for l in xrange(len(d)-len(args[0]) + 1):
                    if args[0] == d[l:l+len(args[0])]:
                        points += [d]
                        break
            if points == []: print 'No datapoints were found to fit that search query.'
            else:
                commit = {p : this.parentdata[p] for p in points}
                print type(points), points
                print "The search returned the following:"
                print prettyprint(points)[:-1] + '.'
commands += [search(data, commands)]
class average(Command):
    name = 'average'
    helptext = """average - Gets the average of every numerical data point.
average <datapoint a> <datapoint b> [etc.] - Gets the average of every datapoint listed if it is numerical."""
    def activate(this, args, caps):
        if len(this.parentdata) == 0: print 'No data to average.'
        roll = 0.0
        count = 0.0
        founddata = False
        if len(args) == 0:
            for d in this.parentdata:
                val = this.parentdata[d]
                if type(val) in [type(1), type(1.5)]:
                    founddata = True
                    roll += val
                    count += 1.0
            if not founddata:
                print 'No numerical data to average.'
            else:
                print 'Average of all numerical data is {}.'.format(roll / count)
        else:
            founds = []
            points = []
            for r in args:
                if r in this.parentdata:
                    founds += [r]
                    if type(this.parentdata[r]) in [type(1), type(1.5)]:
                        points += [r]
            if founds == []: print 'No given points were found in the data set.'
            elif points == []: print 'No given points were numerical.'
            else:
                pvals = [this.parentdata[p] for p in points]
                print 'Of the datapoints provided, the ones found to be numerical were',
                print prettyprint(points),
                print ' and their average is ' + str(sum(pvals) / float(len(points)))
commands += [average(data, commands)]
class save(Command):
    name = 'save'
    helptext = 'save <filename> - Saves all data to the desired filename.'
    def activate(this, args, caps):
        if args == []: print 'Need a filename.'
        else:
            args[0] += '.txt'
            exists = False
            try:
                with open(args[0]) as f: pass
                exists = True
            except IOError: pass
            if exists:
                print "'{}' already exists. Do you want to overwrite it? (y/n)".format(args[0])
                if not YorN(): return
            with open(args[0], 'w') as f:
                for d, v in repack(this.parentdata):
                    f.write(str(d) + ':' + str(v) + '\n')
            print "Data written to '{}'.".format(args[0])
commands += [save(data, commands)]
class load(Command):
    name = 'load'
    helptext = 'load <filename> - Replaces all data with information from the file.'
    def activate(this, args, caps):
        if args == []: print 'Need a filename.'
        else:
            args[0] += '.txt'
            try:
                with open(args[0]) as f: pass
            except IOError:
                print "File '{}' does not exist.".format(args[0])
                return
            if len(this.parentdata) > 0:
                print """Doing this will erase all your data; are you sure you want to do this?
(Hint: the command 'loadin' will not remove existing data.) y/n: """
                if not YorN(): return
            for i in this.parentdata.keys():
                del this.parentdata[i]
            with open(args[0], 'r') as f:
                for line in f:
                    k, v = line.split(':')
                    data[k] = numparse(v)
                    commit[k] = numparse(v)
            print "Data loaded."
commands += [load(data, commands)]
class loadin(Command):
    name = 'loadin'
    helptext = 'loadin <filename> - Adds all the data from the file to the current dataset.'
    def activate(this, args, caps):
        if args == []: print 'Need a filename.'
        else:
            args[0] += '.txt'
            try:
                with open(args[0]) as f: pass
            except IOError:
                print "File '{}' does not exist.".format(args[0])
                return
            if len(this.parentdata) > 0:
                print """Doing this may replace some current values with ones in the file.
Are you sure you want to do this? y/n: """
                if not YorN(): return
            with open(args[0], 'r') as f:
                commit = {}
                for line in f:
                    k, v = line.split(':')
                    data[k] = numparse(v)
                    commit[k] = numparse(v)
            print "Data loaded in."
commands += [loadin(data, commands)]
class changestdin(Command):
    name = 'stdin'
    helptext = """stdin <new stdin> - Change the look of the stdin CLI.
stdin - Resets the stdin to it's orignal look."""
    def activate(this, args, caps):
        if args == []: stdin[0] = STDIN[0]
        else:
            stdin[0] = ' '.join(caps) + ' '
commands += [changestdin(data, commands)]
class time(Command):
    name = 'time'
    helptext = 'time - Displays the time and date.'
    def activate(this, args, caps):
        print "The current time is " + str(datetime.now())
commands += [time(data, commands)]
class leave(Command):
    name = 'exit'
    helptext = 'exit - Closes the program.'
    def activate(this, args, caps):
        raw_input("Goodbye.")
        sys.exit()
commands += [leave(data, commands)]
class reset(Command):
    name = 'reset'
    helptext = 'reset - Resets the CLI and deletes all data.'
    def activate(this, args, caps):
        if len(this.parentdata) > 0:
            print """Warning - doing this will delete all your data.
Are you sure you want to do this? (y/n):"""
            if not YorN(): return
        print "Resetting Program...."
        for i in this.parentdata.keys():
            del this.parentdata[i]
        stdin[0] = STDIN[0]
        print '\n\n\n\n'
        print welcome_msg
commands += [reset(data, commands)]
class credit(Command):
    name = 'credits'
    helptext = 'credits - Displays the accreditations to everyone who worked on this.'
    def activate(this, args, caps):
        print 'Nicholas Colaprete did everything.'
commands += [credit(data, commands)]

# ----- Main Program -----

print welcome_msg
while True:
    userin = raw_input(stdin[0])
    lowers = cleanup(userin)
    uppers = cleanup(userin, True)
    if lowers == []: continue
    com = lowers[0]
    args = []
    if len(lowers) > 1:
        args = lowers[1:]
        uppers = uppers[1:]
    reuse = False
    if '-r' in args:
        reuse = True
        data_old = dict(data)
        data = dict(commit)
        del uppers[args.index('-r')]
        args.remove('-r')
    for c in commands:
        if c.name == com:
            commit = {}
            c.activate(args, uppers)
            break
    else: print "'{}' is not a valid command.".format(com)
    if reuse: data = dict(data_old)
    print
