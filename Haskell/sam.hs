splitClusterBefore :: (a -> Bool) -> Bool -> [a] -> [[a]]
splitClusterBefore _ _ [] = [[]]
splitClusterBefore inCluster filt (c:r) = 
    if filt c then
      let listed = splitClusterBefore True filt r in
        if not inCluster then
          []:(c:head listed):tail listed
        else
          (c:head listed):tail listed
    else 
      let listed = splitClusterBefore False filt r 
      in (c:head listed):tail listed

intersperse :: a -> [a] -> [a]
intersperse _ [] = []
intersperse _ [a] = [a]
intersperse sep (h:t) = h:sep:intersperse sep t

flatten :: [[a]] -> [a]
flatten [] = []
flatten (h:t) = h ++ flatten t

gibberish :: [Char] -> [Char]
gibberish = flatten . (intersperse "igit") . (splitClusterBefore False $ not . (`elem` "aeiou"))

gibberish' :: Bool -> [Char] -> [Char]
gibberish' _ [] = []
gibberish' inCluster (c:r) = 
    if elem c "aeiou" then
      if not inCluster then
        "idig" ++ c:gibberish' True r
      else c:gibberish' True r
    else c:gibberish' False r