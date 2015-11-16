import re
import sys
import traceback

toptagpat = re.compile("<([a-zA-Z0-9]+)(?: ([^>/]+))?>")
flagtagpat = re.compile("<([a-zA-Z0-9]+)(?: ([^>]+))?/>")
metainfpat = re.compile("([a-zA-Z0-9]+)=\"([^\"]*)\"")
notemptypat = re.compile("[^\s]")

metakeyname = "_meta"
valkeyname = "_value"

def indenttext(text, indent):
	newtext = ""
	for l in str(text).split("\n"):
		newtext += "\t"*indent + l + "\n"
	return newtext[:-1]

def parseanyvalue(v):
	try:
		return int(v)
	except ValueError: pass
	try:
		return float(v)
	except ValueError: pass
	try:
		return {"true" : True, "false" : False}[v.strip().lower()]
	except KeyError: pass
	try:
		return str(v)
	except Exception: pass
	return v

def printdict(dct, indentAmount=0, newlineBracket=False, jsonFormat=False, indentSpaces=None):
	tb = "\t"
	if indentSpaces:
		tb = " "*indentSpaces
	indent = tb*indentAmount
	s = ""
	if type(dct) != dict:
		if type(parseanyvalue(dct)) == str:
			s += "\"" + dct + "\""
		else:
			s += str(dct)
	elif len(dct) > 0:
		if newlineBracket:
			s += "\n" + indent
		s += "{\n"
		if metakeyname in dct:
			s += indent + tb + metakeyname + " : " + printdict(dct[metakeyname], indentAmount + 1, newlineBracket, jsonFormat, indentSpaces) + ",\n"
		for key in dct:
			val = dct[key]
			if key == metakeyname:
				continue
			if jsonFormat and val == True:
				s += indent + tb + key + ",\n"
			else:
				s += indent + tb + key + " : " + printdict(val, indentAmount + 1, newlineBracket, jsonFormat, indentSpaces) + ",\n"
		s = s[:-2] + "\n"
		s += indent + "}"
	else:
		s += "{ }"
	return s


def pullmetainfo(text):
	meta = {}
	if not text:
		return meta
	match = re.search(metainfpat, text)
	while match:
		val = parseanyvalue(match.group(2))
		meta[match.group(1)] = val
		text = text[match.span()[1]:]
		match = re.search(metainfpat, text)
	return meta

def toxml(dct, nident=0, spaces=None):
	tb = "\t" if not spaces else " "*spaces
	ident = tb*nident
	s = ""
	if type(dct) == dict:
		for key in sorted(dct):
			val = dct[key]
			if key in [metakeyname, valkeyname]:
				continue
			kmat = re.search("^(.*)_[0-9]+$", key)
			pkey = key
			if kmat: pkey = kmat.group(1)
			s += ident + "<" + pkey
			noitemcount = 0
			if type(val) == dict:
				if metakeyname in val:
					noitemcount = 1
					for mkey in sorted(val[metakeyname]):
						mval = val[metakeyname][mkey]
						s += ' ' + mkey + '="' + str(mval) + '"'
				if len(val) == noitemcount:
					s += "/>\n"
			if type(val) != dict or key == valkeyname:
				s += ">" + str(val) + "</" + pkey + ">\n"
			elif len(val) != noitemcount:
				s += ">\n"
				s += toxml(val, nident+1, spaces)
				s += ident + "</" + pkey + ">\n"
	else:
		print(str(val))
		s = ident + str(dct) + "\n"
	return s

def todict(text):
	findict = {}
	while text != "":
		tag = re.search(toptagpat, text)
		if not tag:
			tag = re.search(flagtagpat, text)
			if not tag:
				if not re.search(notemptypat, text):
					return findict
				else:
					togive = parseanyvalue(text)
					if findict != {}:
						findict[togive] = True
						return findict
					else:
						return togive
			else:
				value = tag.group(1)
				metainf = tag.group(2)
				meta = pullmetainfo(metainf)
				if value in findict:
					if value + "_1" in findict:
						maxd = 1
						for t in [tx for tx in findict if re.match(value + "_[0-9]+", tx)]:
							maxd = max(maxd, int(re.search("_([0-9]+)", t).group(1)))
						value += "_" + str(maxd+1)
					else:
						value += "_1"
				findict[value] = {}
				if meta:
					findict[value][metakeyname] = meta
				text = text[tag.span()[1]:]
		else:
			title = tag.group(1)
			metainf = tag.group(2)
			bottomtag = re.search("</" + title + ">", text)
			content = text[tag.span()[1]:bottomtag.span()[0]]
			text = text[bottomtag.span()[1]:]
			if title in findict:
				if title + "_1" in findict:
					maxd = 1
					for t in [tx for tx in findict if re.match(title + "_[0-9]+", tx)]:
						maxd = max(maxd, int(re.search("_([0-9]+)", t).group(1)))
					title += "_" + str(maxd+1)
				else:
					title += "_1"
			findict[title] = todict(content)
			meta = pullmetainfo(metainf)
			if meta:
				if type(findict[title]) != dict:
					value = findict[title]
					findict[title] = {}
					findict[title][valkeyname] = value
				findict[title][metakeyname] = meta
	return findict

try:
	buff = ""
	if len(sys.argv) >= 2:
		fname = sys.argv[1]
		with open(fname, "r") as f:
			buff = f.read()
		print("printing dict:")
		dic = todict(buff)
		xml = toxml(dic, spaces=3)
		with open(fname + ".out", "w") as f:
			f.write(printdict(dic, newlineBracket=True, indentSpaces=3))
		with open(fname + ".out.back", "w") as f:
			f.write(xml)
	print("finished with no errors")
except:
	print("an error occured:")
	print(indenttext(traceback.format_exc(), 1))
	input("Press enter to exit...")

