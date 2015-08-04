import time
time.clock()

beats = ''
while type(beats) != type(0):
    try: beats = int(input('How many beats to check? '))
    except ValueError: print('Must be a number.')

timings = []
    
input('Press enter to begin recording the first beat.')
old = time.clock()
print('Continue pressing to check beats...')
for i in range(1, beats + 1):
    input()
    cur = time.clock()
    change = round(cur - old, 4)
    timings += [change]
    print('Beat ' + str(i) + ',', 'delay', change)
    print('Beat ' + str(i) + ',', 'inacuraccy', round( (sum(timings) / float(len(timings))) - change, 4 ))
    old = time.clock()

print
print('Average delay: ' + str(sum(timings) / float(len(timings))))
print('BPM: ' + str(round( 60.0 / (sum(timings) / float(len(timings))) , 2)))
print('Range of inaccuracy: ' + str(min(timings)), '-', str(max(timings)) + ',',)
print(str(max(timings) - min(timings)))
print('Range of BPM inacuraccy: ' + str(60.0 / max(timings)), '-', )
print(str(60.0 / min(timings)) + ',', str(60.0 / min(timings) - 60.0 / max(timings)))
print('Beats counted: ' + str(beats))
print

a = ''
while len(a) == 0:
    a = input('Type and hit enter to exit the program... ')
