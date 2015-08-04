function tab_to_str(tab, keys)
	if type(tab) ~= "table" then
		return "" .. tab
	end
	ret = "{"
	for k, v in pairs(tab) do
		if keys then ret = ret .. tab_to_str(k) .. " : " end
		ret = ret .. tab_to_str(v) .. ", "
	end
	return ret:sub(1, ret:len()-2) .. "}"
end
function plist(tab) print(tab_to_str(tab, false)) end
function ptable(tab) print(tab_to_str(tab, true)) end

function slice(list, start, last)
	i = 0
	ret = {}
	for k, v in ipairs(list) do
		i = i + 1
		if i >= start and i <= last then
			ret[k] = v
		end
	end
	return ret
end

function copy(tab)
	if type(tab) ~= "table" then
		return "" .. tab
	end
	ret = {}
	for k, v in pairs(tab) do
		ret[copy(k)] = copy(v)
	end
	return ret
end

function factorial_combos(list)
	if #list == 2 then
		return {list, {list[2], list[1]}}
	end
	if #list < 2 then
		return {list}
	end
	ret = {}
	for i, v in ipairs(list) do
		ret[i] = {v, factorial_combos(slice(list, 2, #list))}
	end
	return ret
end

function expand(coefficients, test)
	coeffs = {}
	for i=1,coefficients do coeffs[i] = 2 end
	while true do
		for i, tab in ipairs(factorial_combo(coeffs)) do
			if test(coeffs) then return coeffs end
		end
		coeffs[1] = coeffs[1] + 1
		for i=2, coefficients do
			if coeffs[i-1] >= coeffs[i] then
				coeffs[i-1] = 0
				coeffs[i] = coeffs[i] + 1
			end
		end
	end
end

sample = {1, 2, 3, 4, 5}
plist(factorial_combos(sample))
