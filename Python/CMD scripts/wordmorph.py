import sys
import copy

startword = "beach"
endword = "shore"

if len(startword) != len(endword):
	raw_input("Words must be the same length.")
	sys.exit()

dictionary = []
with open("thesaurus.txt", "r") as f:
	for line in f.read().split("\n"):
		if len(line) == len(startword):
			dictionary += [line]

startword = [l for l in startword]
endword = [l for l in endword]

steps = set([startword])

def move(wordbase, goal, steps):
	if wordbase in steps:
		return ""
	if wordbase != goal:
		moves = []
		changewords = []
		for w in dictionary:
			sames = 0
			cmove = []
			for i in xrange(len(w)):
				if w[i] != wordbase[i]:
					sames += 1
					if sames > 1:
						break
					cmove = [i, w[i]]
			else:
				if sames == 1:
					moves += [cmove]
					changewords += [[l for l in w]]
					print w, cmove

		for w in changewords:
			result = move(w, goal, steps + [cmove])
			if result != "":
				return steps + [cmove]
	else:
		return wordbase


raw_input(move(startword, endword, []))