ones = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen"
    ]

tens = [
    "zeroith",
    "eleventy",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety"
    ]

hundred = "hundred"

magnitudes = [
    "",
    " thousand",
    " million",
    " billion",
    " trillion",
    " quadrillion",
    " quintillion",
    " sextillion",
    " septillion"
    ]

def say_number(number):
    maxdigits = 3*(len(magnitudes)-1)
    maxnum = 10**maxdigits
    if number < 0:
        return "negative " + say_number(-number)
    elif isinstance(number, float):
        back = str(number).rsplit(".", 1)[-1]
        backstr = " ".join([ones[int(d)] for d in back])
        return say_number(int(number)) + " point " + backstr
    elif number < 20:
        return ones[number]
    elif number < 100:
        front = tens[number // 10]
        rest = number % 10
        if rest == 0:
            return front
        else:
            return front + "-" + say_number(rest)
    elif number < 1000:
        front = ones[number // 100]
        rest = number % 100
        if rest == 0:
            return front + " " + hundred
        else:
            return front + " " + hundred + " and " + say_number(rest)
    elif number < maxnum:
        peices = []
        smallend = False
        for i, magname in reversed(list(enumerate(magnitudes))):
            magnitude = 10**(3*i)
            if number >= magnitude:
                front = number // magnitude
                number = number % magnitude
                if i == 1 and number < 100 and number > 0:
                    smallend = True
                peices.append(say_number(front) + magname)
        if smallend:
            return ", ".join(peices[:-1]) + " and " + str(peices[-1])
        else:
            return ", ".join(peices)
    else:
        return str(number // maxnum) + "*10^" + str(maxdigits) + " plus " + say_number(number % maxnum)

def getnumstr(nstr):
    try:
        return int(nstr)
    except ValueError:
        return float(nstr)

if __name__ == "__main__":
    while True:
        given = input(" -> ")
        if given == "":
            break
        try:
            number = getnumstr(given)
            print(say_number(number))
        except ValueError:
            print("nonono, give number.")
