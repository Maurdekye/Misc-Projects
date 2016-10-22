separateVowels :: [Char] -> [[Char]]
separateVowels "" = [""]
separateVowels (c:r) = 
  let listed = separateVowels r in
    if elem c "aeiou" then  
      "":(c:head listed):tail listed
    else 
      (c:head listed):tail listed

intersperse :: a -> [a] -> [a]
intersperse _ [] = []
intersperse _ [a] = [a]
intersperse sep (h:t) = h:sep:intersperse sep t

flatten :: [[a]] -> [a]
flatten [] = []
flatten (h:t) = h ++ flatten t

gibberish :: [Char] -> [Char]
gibberish str = flatten $ intersperse "igit" $ separateVowels str