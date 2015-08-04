alph = [
  'a', 'b', 'c', 'd', 'e', 'f',
  'g', 'h', 'i', 'j', 'k', 'l',
  'm', 'n', 'o', 'p', 'q', 'r',
  's', 't', 'u', 'v', 'w', 'x',
  'y', 'z', ' ', '.', '1', '2',
  '3', '4', '5', '6', '7', '8',
  '9', '0', 'A', 'B', 'C', 'D',
  'E', 'F', 'G', 'H', 'I', 'J',
  'K', 'L', 'M', 'N', 'O', 'P',
  'Q', 'R', 'S', 'T', 'U', 'V',
  'W', 'X', 'Y', 'Z', '#', '!',
  '?', "'", '*', ';', '\n', '-',
  ','] 

def simplefy(intext):
  ret = ""
  for l in intext:
    if l in alph: ret += l
  return ret

def DAScrypt(text, key):
  text = simplefy(text)
  cAlph = []
  for i in xrange(len(key)):
    cAlph.append(alph[alph.index(key[i]):] + 
                 alph[:alph.index(key[i])])
  newtext = ""
  for i in xrange(len(text)): 
    newtext += alph[cAlph[i%len(key)].index(text[i])]
  return newtext 

def recrypt(text, key):
  for x in xrange(len(key)**2):
    text = DAScrypt(text, key)
  return text

key = raw_input("Please enter key: ")
print "Encrypting..."
with open("text.txt", "r") as off:
  with open("Output.txt", "w") as on:
    on.write(recrypt(off.read(), key))
raw_input("Text encrypted to 'Output.txt'.")
