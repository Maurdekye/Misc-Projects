import Data.Char
doubleMe x = x * x
doubleUs x y = doubleMe x + doubleMe y
doubleSmallNumber x = if x > 100 then x else x*2
prependFive list = 5:list
factorial x = product [2..x]
squares = [ x*x | x <- [1..]]
fibonacci n
	| n < 0     = 0
	| n == 1    = 1
	| otherwise = fibonacci (n - 1) + fibonacci (n - 2)
pascal a b
	| a <= b 	= 1
	| b <= 0 	= 1
	| otherwise = pascal (a - 1) b + pascal (a - 1) (b - 1)
fibs = [fibonacci n | n <- [1..]]
isprime n = if null [0 | i <- init [2..n], mod n i == 0] then True else False
primes = 1:[i | i <- [2..], isprime i]
addNotIn item list =  if item `elem` list then list else item:list
bmi height weight = weight / (height * height)
e = sum [1 / factorial n | n <- [0..170]]
phi = fibonacci 25 / fibonacci 24
absolute n
	| n < 0 	= n * (-1)
	| otherwise = n
mandel re im cRe cIm iters
    | iters <= 0 = 0
    | (re * re) + (im * im) > 4 = iters
    | otherwise = 
	let nRe = ( (re * re) - (im * im) ) + cRe;
		nIm = ( 2 * re * im ) + cIm 
	in mandel nRe nIm cRe cIm (iters - 1)
collatz 1 = [1]
collatz n
	| even n = n:collatz (n `div` 2)
	| odd n = n:collatz ((n * 3) + 1)

-- Vignere Ciphering

alphaBet = ['a'..'z'] ++ [' ']
noRepeat word list
	| null word = list
	| head word `elem` list = noRepeat (tail word) list
	| otherwise = noRepeat (tail word) (list ++ [head word])
lower word = [toLower l | l <- word]
cipherBet key = noRepeat alphaBet (noRepeat key [])
cipher key text
	| null text = "Need to cipher something!"
	| null key  = "No key to cipher with!"
	| otherwise = backCipher (lower text) (lower key) []
	where backCipher text key outText
			| null text = outText
			| otherwise = backCipher (tail text) key ( outText ++ [((cipherBet key) !! ( literatrans (head text) ))] )
			where literatrans l 
					| null finl = 27
					| otherwise = finl !! 0
					where finl = [n | n <- [0..26], ( alphaBet !! n ) == l]
decipher key text
	| null text = "Need to decipher something!"
	| null key  = "No key to decipher with!"
	| otherwise = frontCipher (lower text) (lower key) []
	where frontCipher text key outText
			| null text = outText
			| otherwise = frontCipher (tail text) key ( outText ++ [(alphaBet !! ( ciphertrans (head text) key ))] )
			where ciphertrans l key
					| null finl = 27
					| otherwise = finl !! 0
					where finl = [n | n <- [0..26], ( (cipherBet key) !! n ) == l]
