import math
import random

# Begin magnitude definitions

magnitude_list = {
    "Minecraft Stacks" : {
        "item" : 1,
        "half stack" : 32,
        "stack" : 64,
        "row" : 576,
        "chestfull" : 1728,
        "inventory" : 2304,
        "double-chest full" : 3456
        },

    "Sheets of Paper" : {
        "sheet" : 1,
        "stack" : 20,
        "binder" : 100,
        "drawer" : 800,
        "filing cabinet" : 4800
        },

    "General Currency" : {
        "penny" : 1,
        "nickel" : 5,
        "dime" : 10,
        "quarter" : 25,
        "dollar bill" : 100,
        "five dollar bill" : 500,
        "ten dollar bill" : 1000,
        "twenty dollar bill" : 2000,
        "fifty dollar bill" : 5000,
        "one-hundred dollar bill" : 10000
        },
        
    "Wizarding Currency" : {
        "knut" : 1,
        "sickle" : 29,
        "galleon" : 493
        }
    }

mag_names_list = {}
i = 0
for name in magnitude_list:
    i+=1
    mag_names_list[i] = name


# End magnitude definitions
	
def recurnum(message):
	num = raw_input(message)
	try:
		num = int(num)
	except ValueError:
		print "Sorry, I didn't understand that."
		return recurnum(message)
	return num

def magnitude_arrange(amount, ord_mag):
    """Prints a string containing the amount given organized under the order of magnitude given."""
    def pluralize(string):
        if string[-1:].lower() in ["a", "i", "o", "u", "s"]:
            return string + "es"
        elif string[-1:].lower() == 'y':
            return string[:-1] + "ies"
        else:
            return string + "s"
    
    def swapped(lis):
        d = {}
        for i in lis:
            d[lis[i]] = i
        return d
    
    # Getting the amounts of items per currency
    ord_mag = swapped(ord_mag)
    title = "unit"
    if 1 in ord_mag:
        title = ord_mag[1]
        del ord_mag[1]
    if amount <= 0: amount = recurnum("How many "+pluralize(title)+" are there? ")
    orAmnt = amount
    magnitudes = []
    magTitles = []
    for v in reversed(sorted(ord_mag)):
        if v <= amount:
            magnitudes.append(math.floor(amount/v))
            magTitles.append(ord_mag[v])
            amount = amount%v
    # Putting the result into text
    state = str(orAmnt)+" "+pluralize(title)+" is "
    i = 0
    for nums in magnitudes:
        titles = magTitles[i]
        if nums != 1.0:
            titles = pluralize(titles)
        if i+1 == len(magnitudes) and amount == 0:
            if len(magnitudes) > 1:
                state += "and "
            state += "%d %s." % (nums,titles)
        else:
            state += "%d %s, " % (nums,titles)
        i+=1
    if amount != 1:
        title = pluralize(title)
    if len(magnitudes) == 0:
        state += " %d %s." % (amount,title)
    elif amount == 0:
        state += ""
    else:
        state += "and %d %s." % (amount,title)
    print ""
    print state

cmag = {}
print "What magnitude would you like to format under?"
for j in mag_names_list:
    print "\t%d. %s" % (j, mag_names_list[j])

endNum = str(len(magnitude_list))
inp = recurnum("1 - " + endNum + " please: ")
while not inp in range(1,len(mag_names_list)+1):
    inp = recurnum("1 - " + endNum + " please, no higher or lower: ")

cmag = magnitude_list[mag_names_list[inp]]
magnitude_arrange(0, cmag)
raw_input("\nPress enter to exit the program...")