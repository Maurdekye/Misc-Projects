function ack(m, n)
	if m == 0 then return n + 1 end
	if n == 0 then return ack(m - 1, 1) end
	return ack(m - 1, ack(m, n - 1))
end

for i=0,4 do
	for j=0,4 do
		print(i, j, ack(i, j))
	end
end
