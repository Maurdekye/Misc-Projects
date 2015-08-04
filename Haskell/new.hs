rev :: [a] -> [a]
rev [] = []
rev (h:t) = rev t ++ [h]

headTail :: [a] -> (a, [a])
headTail (h:t) = (h, t)

part :: (a -> Bool) -> [a] -> ([a], [a])
part _ [] = ([], [])
part f (h:t) = let joinpair (a,b) (c,d) = (a ++ c, b ++ d)
                   pairing = if f h then ([h], []) else ([], [h])
               in  joinpair pairing $ part f t

qsort :: (Ord a) => [a] -> [a]
qsort [] = []
qsort (h:t) = let (a,b) = part (<h) t
              in  qsort a ++ h:qsort b

transpose :: [[a]] -> [[a]]
transpose l
    | any null l = []
    | otherwise = map head l:(transpose $ map tail l)

filter' :: (a -> Bool) -> [a] -> [a]
filter' f l = [i | i <- l, f i]

map' :: (a -> b) -> [a] -> [b]
map' f l = [f i | i <- l]

fold :: (a -> a -> a) -> [a] -> a
fold _ [] = error "can't fold empty list"
fold _ [i] = i
fold f (h:t) = f h $ fold f t

flatten :: [[a]] -> [a]
flatten [] = []
flatten (h:t) = h ++ flatten t

mymult :: (Num a) => Int -> a -> a
mymult n = sum . replicate n

unique :: (Eq a) => [a] -> [a]
unique [] = []
unique (h:t)
    | elem h t  = unique t  
    | otherwise = h:unique t

fixl :: Int -> a -> [a] -> [a]
fixl i fill list@(_:t)
    | diff > 0  = fixl i fill $ fill:list
    | diff < 0  = fixl i fill t
    | otherwise = list
    where diff = i - length list

fixr :: Int -> a -> [a] -> [a]
fixr i fill list
    | diff > 0  = fixr i fill $ list ++ [fill]
    | diff < 0  = fixr i fill $ init list
    | otherwise = list
    where diff = i - length list

fixl1 :: Int -> [a] -> [a]
fixl1 i list@(h:_) = fixl i h list

fixr1 :: Int -> [a] -> [a]
fixr1 i list = fixr i (last list) list

fluff :: a -> [a] -> [a]
fluff _ [] = []
fluff _ [e] = [e]
fluff i (h:t) = h:i:fluff i t

applyAll :: [a -> a] -> [a] -> [a]
applyAll [] l = l
applyAll (hf:tf) l = map hf $ applyAll tf l

applyUnto :: [[a] -> [a]] -> [a] -> [a]
applyUnto [] l = l
applyUnto (hf:tf) l = hf $ applyUnto tf l

applyMultiple :: (a -> a) -> Int -> a -> a
applyMultiple _ 0 val = val
applyMultiple f c val = f $ applyMultiple f (c-1) val

takeAfter :: Int -> [a] -> [a]
takeAfter 0 l = l
takeAfter i (_:t) = flip takeAfter t $ i - 1

split :: Int -> [a] -> ([a], [a])
split i l = (take i l, takeAfter i l)

shuffle :: [a] -> [a]
shuffle l = let (a,b) = flip split l $ flip quot 2 $ length l 
            in  flatten . transpose $ [a, b]

sampleMatrix :: Int -> a -> [[a]]
sampleMatrix s = take s . repeat . take s . repeat

prime :: Int -> Bool
prime n = any (\a -> mod n a /= 0) [2..(quot n 2 - 1)]

devoidOf :: (Eq a) => [a] -> a -> [a]
devoidOf [] _ = []
devoidOf (h:t) e
    | h == e    = devoidOf t e
    | otherwise = h:devoidOf t e

without :: [a] -> Int -> [a]
without l@(h:t) ind
    | ind >= length l = error "Index out of range"
    | ind < 1         = t
    | otherwise       = h:without t (ind - 1)

shuffleMovement :: Int -> [Int]
shuffleMovement (-1) = []
shuffleMovement 0 = []
shuffleMovement n
    | odd n     = 1:shuffleMovement ((n - 1) `div` 2)
    | otherwise = 0:shuffleMovement (n `div` 2)

getShuffle :: Int -> [Int]
getShuffle n = rev . shuffleMovement $ n - 1