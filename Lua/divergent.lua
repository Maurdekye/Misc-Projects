function diverge(equation, start, sample)
	sumall = 0
	for generation=1,sample do
		sumall = sumall + start
		if (generation%(sample/100) == 0) then print(generation, start, sumall) end
		start = equation(generation)
	end
end

diverge(function(n) return n end, 1, 100000000)
