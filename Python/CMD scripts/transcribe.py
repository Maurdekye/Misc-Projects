infile = raw_input("What file would you like to transcribe? ")
with open(infile + ".txt", "r") as tran:
    with open("Output.txt", "w") as scribe:
        for line in tran:
            if line[-1:] == "\n":
                line = line[:-1]
            if line == "slave":
                scribe.write("-\n")
            else:
                scribe.write(line + "\n")
