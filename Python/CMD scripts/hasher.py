import math

fileName = "tohash.txt" # name of file to be hashed

filesize = 0
hashnum = 0
with open("tohash.txt", "r") as f:
    for line in f:
        for letter in line:
            filesize += 1
            n = ord(letter)
            hashnum += n**hashnum
            hashnum %= n**2
            hashnum *= hashnum*n
            hashnum -= n*hashnum

hashstr = ""

for i in xrange(32):
    
