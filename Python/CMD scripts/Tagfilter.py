class tagged(object):
    def __init__(this, name, tags):
        this.name = name
        this.tags = tags

Search = [
    tagged("Someshit", ['tag1', 'foo', 'bar', 'baz', 'bing']),
    tagged("Othashit", ['tag1', 'baz', 'cake']),
    tagged("Tableshit", ['foo', 'bing', 'bar']),
    tagged("Fab Dapp", ['candle', 'bing', 'cake']),
    tagged("Schindler", ['tag1', 'baz', 'bing', 'cake']),
    tagged("Chromeh", ['bar', 'bing', 'baz'])
    ]

def filterTags(toSearch, tags):
    to_sender = []
    for item in toSearch:
        for tag in tags:
            if tag not in item.tags: break
        else: to_sender.append(item.name)
    return to_sender

def getAllTags(toSearch):
    to_sender = []
    for item in toSearch:
        for tag in item.tags:
            if tag not in to_sender: to_sender.append(tag)
    return to_sender

for i in filterTags(Search, ['cake', 'baz']): print i
print getAllTags(Search)
