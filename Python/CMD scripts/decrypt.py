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
 
def deDAScrypt(text, key):
  text = simplefy(text)
  cAlph = []
  for i in xrange(len(key)):
    cAlph.append(alph[alph.index(key[i]):] + 
                 alph[:alph.index(key[i])])
  newtext = ""
  for i in xrange(len(text)): 
    newtext += cAlph[i%len(key)][alph.index(text[i])]
  return newtext

def recrypt(text, key):
  for x in xrange(len(key)**2):
    text = deDAScrypt(text, key)
  return text

def paraphrase(text, charlimit=60):
  charcount = 0
  output = ""
  for l in text:
    charcount += 1
    output += l
    if charcount > charlimit:
      output = output[::-1]
      charcount = 0
      for i in xrange(len(output)):
        if output[i] == " ":
          output = output[:i] + "\n" + output[i:]
          output = output[::-1]
          break
  return output

key = raw_input("Please enter key: ")
print "Decrypting..."
output = ""
with open("Output.txt", "r") as off:
  output = recrypt(off.read(), key)
print "Displaying decrypted text:\n"
raw_input(output)
