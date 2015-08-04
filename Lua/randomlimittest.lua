limit = 30

print('warming up...')
for x=1,3 do
	c = os.time()
	i = 0
	f = i+x
	while os.time() < c + 1 do
		i = i + 1
	end
	print(x, i)
end
print('Committing Averages')
avg = 0
for x=1,limit do
	c = os.time()
	i = 0
	f = i+x
	while os.time() < c + 1 do
		i = i + 1
	end
	avg = avg + i
	print(x..'/'..limit, i)
end
print('Average', avg/limit)
