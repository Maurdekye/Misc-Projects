def framarang(imgs):
    frames = {}
    for i in imgs:
        order = -1
        name = i
        for x in xrange(0, len(i), -1):
            if i[x] == "_":
                name = i[:x]
                order = int(i[x+1:])
                break
        if order == -1:
            frames[name] = [0]
        else:
            frames[name] = list(frames[name])
            frames[name].append(order)
    for i in frames:
        frames[i].sort()
