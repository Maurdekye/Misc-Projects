def spaceout(text):
    return " ".join(list(text.upper()))

def corner(text):
    f = spaceout(text) + "\n"
    for l in text[1:]:
        f += l.upper() + "\n"
    return f
    
def square(text):
    text = text.upper()
    f = " ".join(list(text)) + "\n"
    spacing = " " * (2 * len(text) - 3)
    for l, r in zip(text[1:-1], text[1:-1][::-1]):
        f += l + spacing + r + "\n"
    f += " ".join(list(text[::-1])) + "\n"
    return f
    
def cross(text):
    text = text.upper()
    f = ""
    lim = len(text) // 2
    if len(text) % 2 == 1:
        for i, l in enumerate(text[:lim]):
            ows = (2 * i) * " "
            iws = (4 * (lim - i) - 1) * " "
            f += ows + l + iws + l + "\n"
        f += (2 * lim) * " " + text[lim] + "\n"
        for i, l in enumerate(text[lim+1:]):
            iws = (4 * i + 3) * " "
            ows = (2 * (lim - i) - 2) * " "
            f += ows + l + iws + l + "\n"
    else:
        for i, l in enumerate(text[:lim]):
            ows = (2 * i) * " "
            iws = (4 * (lim - i) - 3) * " "
            f += ows + l + iws + l + "\n"
        for i, l in enumerate(text[lim:]):
            iws = (4 * i + 1) * " "
            ows = (2 * (lim - i) - 2) * " "
            f += ows + l + iws + l + "\n"
    return f
    
def squarecross(text):
    if len(text) <= 2:
        return square(text)
    text = text.upper()
    f = spaceout(text) + "\n"
    lim = len(text) // 2
    if len(text) % 2 == 1:
        for i, (l, r) in list(enumerate(zip(text[:lim], text[::-1][:lim])))[1:]:
            ows = (2 * i - 1) * " "
            iws = (4 * (lim - i) - 1) * " "
            f += l + ows + l + iws + l + ows + r + "\n"
        spc = (2 * lim - 1) * " "
        f += text[lim] + spc + text[lim] + spc + text[lim] + "\n"
        for i, (l, r) in list(enumerate(zip(text[lim+1:], text[::-1][lim+1:])))[:-1]:
            iws = (4 * i + 3) * " "
            ows = (2 * (lim - i) - 3) * " "
            f += l + ows + l + iws + l + ows + r + "\n"
    else:
        for i, (l, r) in list(enumerate(zip(text[:lim], text[::-1][:lim])))[1:]:
            ows = (2 * i - 1) * " "
            iws = (4 * (lim - i) - 3) * " "
            f += l + ows + l + iws + l + ows + r + "\n"
        for i, (l, r) in list(enumerate(zip(text[lim:], text[::-1][lim:])))[:-1]:
            iws = (4 * i + 1) * " "
            ows = (2 * (lim - i) - 3) * " "
            f += l + ows + l + iws + l + ows + r + "\n"
    f += spaceout(text)[::-1] + "\n"
    return f

def block(text):
    f = ""
    for i in range(len(text)):
        f += spaceout(text) + "\n"
        text = text[1:] + text[0]
    return f

def diamond(text):
    f = ""
    text = text.upper()
    for i, _ in enumerate(text):
        f += spaceout(text[:i]) + "\n"
    for i, _ in enumerate(text):
        spc = (2 * i) * " "
        f += spc + spaceout(text[i:]) + "\n"
    return f
    
def crossdiamond(text):
    f = ""
    text = text.upper()
    text = text[-1] + text[:-1]
    for i, _ in enumerate(text):
        f += spaceout(text[:i]) + "\n"
        text = text[1:] + text[0]
    for i, _ in enumerate(text):
        spc = (2 * i) * " "
        f += spc + spaceout(text[i:]) + "\n"
        text = text[1:] + text[0]
    return f

def star(text):
    text = text.upper()
    f = ""
    for i, c in enumerate(text[:-1]):
        iws = ( 2 * (len(text) - i - 2) + 1) * " "
        ows = ( 2 * i ) * " "
        f += ows + c + iws + c + iws + c + "\n"
    f += " ".join(list(text)) + " "
    f += " ".join(list(text[:-1][::-1])) + "\n"
    for i, c in enumerate(text[:-1][::-1]):
        ows = ( 2 * (len(text) - i - 2)) * " "
        iws = ( 2 * i + 1) * " "
        f += ows + c + iws + c + iws + c + "\n"
    return f
    
if __name__ == "__main__":
  print(crossdiamond(input()))
