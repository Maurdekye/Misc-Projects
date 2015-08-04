factorial :: Float -> Float
factorial i = foldr1 (*) [1..i]

tau :: Float
tau = 2 * pi

modf :: Float -> Float -> Float
modf x y = x - (y * (fromIntegral $ truncate (x/y)))

taylor :: Float -> Float -> Float
taylor acc val
    | acc <= 0  = 0
    | otherwise = (val ** l / factorial l) - (val ** r / factorial r) + taylor (acc - 1) val
    where l = 4*acc - 3
          r = 4*acc - 1

sine :: Float -> Float
sine = taylor 10 . (`modf` tau)

cosine :: Float -> Float
cosine = sin . (+ pi / 2)

tangent :: Float -> Float
tangent x = sine x / cosine x

data Point = Point Float Float derives (Show)
data Shape = Circle Point Float | Rect Point Point | Polygon [Point] derives (Show)

square :: Point -> Float -> Rect
square p@(Point x y) s = Rect p (Point (x + s) (y + s))

perimeter :: Shape -> Float
perimiter (Circle _ r) = pi * r ** 2
perimiter (Rect (Point x1 y1) (Point y1 y2)) = (abs $ x2 - x1) * (abs $ y2 - y1)
