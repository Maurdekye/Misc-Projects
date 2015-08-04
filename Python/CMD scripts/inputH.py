def recurnum(message, wrongmsg="Sorry, I didn't understand that."):
        num = raw_input(message)
        try:
                num = int(num)
        except ValueError:
                print wrongmsg
                return recurnum(message)
        return num

def clamp(minim, num, maxim):
    if num < minim:
        return minim
    if num > maxim:
        return maxim
    return num

def keyPress():
    import msvcrt
    return msvcrt.getch()
