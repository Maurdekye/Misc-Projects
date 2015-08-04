
class Command(object):
    def __init__(this, name, perm_types, activate):
        this.name= name # String
        this.perm_types = perm_types # Array
        this.activate = activate # Function

    def call(this, usr):
        if usr.perm_in(this):
            this.activate(usr)
            return True
        else: return False
        
class User(object):
    def __init__(this, name, perms):
        this.name = name # String
        this.perms = perms # Array

    def perm_in(this, com):
        for i in this.perms:
            if i in com.perm_types: return True
        return False
    
def a(ply):print "Eat Shit and Die"
def b(ply):print "Calling All Bitchasses, " + ply.name
def c(ply):print "You're going down."
Comms = [
    Command("eatshit", [0, 1, 2], a),
    Command("zombieland", [1, 2], b),
    Command("killall", [2], c)
    ]

def speech(usr, msg):
    if len(msg) > 0 and msg[0] == "/":
        com = ""
        for l in msg:
            if l == " ":
                break
            else: com += l
        for c in Comms:
            if com[1:] == c.name:
                if not usr.perm_in(c):
                    print "You don't have the perms."
                    break
                c.activate(usr)
                break
        else: print "Not a command: " + com[1:]
        return ""
    return msg

def printToChat(usr, msg):
    if msg != "":
        print "<{}> {}".format(usr.name, msg)

me = User("Cannibal Steve", [0, 1, 2])

print "Commands:"
for x in Comms:
    print "/" + x.name

while True:
    intext = raw_input()
    printToChat(me, speech(me, intext))
