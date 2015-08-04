import os

def sqlize(infileDir, outfileDir=None, tablename=None, columname="items"):
    if infileDir == None: return False
    if not os.path.exists(infileDir):
        return None

    if not os.path.exists("Sql Dumps"):
        os.makedirs("Sql Dumps")

    if tablename == None:
        tablename = infileDir[:infileDir.rindex(".")]

    if outfileDir == None:
        outfileDir = infileDir[:infileDir.rindex(".")] + ".sql"
    
    itemlist = []
    with open(infileDir, 'r') as infile:
        for item in infile: itemlist += [item.replace("\n", "")]

    maxlen = 0
    for item in itemlist:
        maxlen = len(item) if len(item) > maxlen else maxlen

    toWrite = """DROP TABLE If EXISTS `{0}`;
CREATE TABLE `{0}` (
  `{1}` varchar({2}) NOT NULL,
  PRIMARY KEY(`{1}`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

INSERT INTO `{0}` VALUES """.format(tablename, columname, maxlen)

    for item in itemlist:
        item = item.replace("'", "\\'")
        toWrite += "('{}'),".format(item)

    toWrite = toWrite[:-1] + ";"
        
    with open(outfileDir, 'w') as outFile:
        outFile.write(toWrite)

    return outfileDir

if __name__ == "__main__":
    in_f = raw_input("Which file to translate? ")
    result = sqlize(in_f, ("Sql Dumps\\" + in_f[:in_f.rindex(".")] + ".sql"))
    while result == None:
        print "That file doesn't exist."
        in_f = raw_input("Which file to translate? ")
        result = sqlize(in_f, ("Sql Dumps\\" + in_f[:in_f.rindex(".")] + ".sql"))
    raw_input("Finished; File copied to '{}'.".format(result))
        
