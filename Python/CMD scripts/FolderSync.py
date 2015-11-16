import os
import copy
import shutil
import time

def filePaths(path):
	for dirpath, _, files in os.walk(path):
		if path != dirpath:
			yield dirpath
		for fname in files:
			yield os.path.join(dirpath, fname)

def computeCache(path):
	cache = {}
	for fpath in filePaths(path):
		if os.path.isfile(fpath):
			cache[fpath.split(path)[1]] = os.stat(fpath).st_mtime
		else:
			cache[fpath.split(path)[1]] = 0
	return cache

def getDiff(oldcache, newcache):
	added, removed = copy.copy(oldcache), copy.copy(newcache)
	changed = {}
	for key in oldcache:
		if key in newcache:
			del removed[key]
			del added[key]
			changed[key] = newcache[key]
	for kx in copy.copy(removed):
		for ky in copy.copy(removed):
			if ky.startswith(os.path.abspath(kx) + "\\"):
				del removed[ky]
	for key in copy.copy(changed):
		if oldcache[key] == newcache[key]:
			del changed[key]
	return removed, added, changed

def copypath(fr, to):
	if os.path.isfile(fr):
		try:
			dirs = os.path.split(to)[0]
			os.makedirs(dirs)
		except FileExistsError:
			pass
		shutil.copy2(fr, to)
	else:
		try:
			os.makedirs(to)
		except FileExistsError:
			pass

def rempath(path):
	if os.path.isdir(path):
		shutil.rmtree(path)
	else:
		os.remove(path)

def update(fr, to, add, remove):
	for p in [fr, to]:
		if not os.path.isdir(p):
			os.mkdir(p)
	for newitem in add:
		copypath(fr + newitem, to + newitem)
	for olditem in remove:
		rempath(to + olditem)

RemoteDir = "V:\\Transfer Items\\nc\\~syncstore"

SyncDir = os.getcwd()

lastCache = {}

while True:

	# check for diff between local and last folder
	localCache = computeCache(SyncDir)
	if lastCache != localCache:
		print("\nPushing local changes:")
		adds, removes, changes = getDiff(lastCache, localCache)
		update(SyncDir, RemoteDir, {**adds, **changes}, removes)
		lastCache = localCache
		for f in adds:
			print('\tSent new file "' + f + '"')
		for f in changes:
			print('\tUpdated file "' + f + '"')
		for f in removes:
			print('\tDeleted file "' + f + '"')

	# check for diff between remote and last folder
	remoteCache = computeCache(RemoteDir)
	if lastCache != remoteCache:
		print("\nRecieving remote changes:")
		adds, removes, changes = getDiff(lastCache, remoteCache)
		update(RemoteDir, SyncDir, {**adds, **changes}, removes)
		lastCache = remoteCache
		for f in adds:
			print('\tRetrieved new file "' + f + '"')
		for f in changes:
			print('\tUpdated file "' + f + '"')
		for f in removes:
			print('\tDeleted file "' + f + '"')

	time.sleep(2)