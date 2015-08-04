function euler(n)
	return (1 + 1/n)^n
end

for i=1,12 do
	print(i, 10^i, euler(10^i))
end

print(math.e)
