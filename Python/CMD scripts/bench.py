import time

for i in range(10):
  s = time.clock()
  c = 0
  while time.clock() < s+1:
    c+=1
  print(c)
input("done")