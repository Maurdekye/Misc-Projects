for i=1,30 do
	fstr = ""
	if i%5 == 0 then fstr = "buzz " .. fstr end
	if i%3 == 0 then fstr = "fizz " .. fstr end
	if i%5 ~= 0 and i%3 ~= 0 then fstr = i .. fstr end
	print(fstr)
end
