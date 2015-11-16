import pyperclip
import time
import re
import os
import sys
import traceback

MasterFilePath = "V:\\Transfer Items\\nc\\~clipsync.txt"

def fmat(text, maxwid=80, indent=0):
	ftext = ""
	m = re.search("[\n\r]", text[:maxwid])
	while len(text) > maxwid or m:
		m = re.search("[\n\r]", text[:maxwid])
		if not m:
			ftext += "\t"*indent + text[:maxwid] + "\n"
			text = text[maxwid:]
		else:
			ftext += "\t"*indent + text[:m.span()[1]]
			text = text[m.span()[1]:]
	return ftext + "\t"*indent + text
	
# init
lastClip = ""
if pyperclip.paste() == None:
	pyperclip.copy("")
try:
	pt = os.path.split(MasterFilePath)[0]
	os.makedirs(pt)
except:
	pass
try:
	if not os.path.isfile(MasterFilePath):
		with open(MasterFilePath, "w") as f:
			pass
except:
	print(traceback.format_exc())
	input("Error: File path cannot be written/read.\nPlease change path.\nPress enter to exit...\n\n")
	sys.exit()

while True:

	# Check for new clipboard on local computer
	newLocalClip = pyperclip.paste().replace("\r\n", "\n")
	if newLocalClip != lastClip:
		with open(MasterFilePath, "w") as f:
			f.write(newLocalClip)
		lastClip = newLocalClip
		print("Local clip pushed:\n", fmat(newLocalClip, 60, 1))
	
	# Check for new clipboard on master file
	try:
		with open(MasterFilePath, "r") as f:
			newMasterClip = f.read()
			if newMasterClip != lastClip:
				pyperclip.copy(newMasterClip)
				lastClip = newMasterClip
				print("New Master clip:\n", fmat(newMasterClip, 60, 1))
	except:
		pass

	time.sleep(0.5)
