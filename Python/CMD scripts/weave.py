def weave(texta, textb):
    fintext = ""
    for a, b in zip(texta, textb):
        fintext += a + b
    if len(texta) > len(textb):fintext += texta[len(textb):]
    if len(textb) > len(texta):fintext += textb[len(texta):]
    return fintext

def wordWeave(texta, textb):
    fintext = ""
    splita, splitb = texta.split(" "), textb.split(" ")
    for i in range(max(len(splita), len(splitb))):
        if i < len(splita): fintext += " " + splita[i]
        if i < len(splitb): fintext += " " + splitb[i]
    return fintext[1:]

print(weave("sandwiches taste nice", "your mother eats turnips"))
print(wordWeave("can't eat canned casserole without water spouts",
                "sounds like you enjoy new york nuggets"))
