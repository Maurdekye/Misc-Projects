import qualified Data.Set as Set
import qualified Data.List as ListUtils
import qualified Data.Char as StringUtils
import qualified Text.Read as Parser
import qualified Data.List as ListUtils

-- prolog environment emulation

data Object = Zero | One | Two | Three | Out | In | Dorm_Room | Bus_Stop | Bus | Classroom | Book | TV | Alarm_Clock | Exam deriving (Eq, Ord, Enum, Show, Read)
data Clause = I_Am_At | At | Holding | Knowledge | Path | Late | Slept | Alarm_Set deriving (Eq, Ord)
data Fact = Flag Clause | Tag Clause Object | Link Clause Object Object | ThreeLink Clause Object Object Object deriving (Eq, Ord)
type ProlEnviron = Set.Set Fact

define :: Fact -> ProlEnviron -> ProlEnviron
define rel env = Set.insert rel env

assert :: Fact -> ProlEnviron -> Bool
assert rel env = Set.member rel env

retract :: Fact -> ProlEnviron -> ProlEnviron
retract rel env = Set.delete rel env

tagValue :: Clause -> ProlEnviron -> Maybe Object
tagValue cls env = case [v | Tag c v <- Set.toList env, c == cls] of 
  [] -> Nothing
  (v:_) -> Just v

linkValue :: Clause -> ProlEnviron -> Maybe Object
linkValue cls frst env = case [s | Link c f s <- Set.toList env, c == cls, f == frst] of 
  [] -> Nothing
  (v:_) -> Just v

threeLinkValue :: Clause -> Object -> Object -> ProlEnviron -> Maybe Object
threeLinkValue env cls here dir = case [t | ThreeLink c h d t <- Set.toList env, c == cls, h == here, d == dir] of 
  [] -> Nothing
  (v:_) -> Just v

-- starting constants

initialState :: ProlEnviron
initialState = Set.fromList [
  Tag I_Am_At Dorm_Room,
  Tag Knowledge Zero,
  Tag Mounted TV,
  ThreeLink Path Dorm_Room Out Bus_Stop,
  ThreeLink Path Bus_Stop In Dorm_Room,
  ThreeLink Path Bus Out Classroom,
  Link At Book Dorm_Room,
  Link At TV Dorm_Room,
  Link At Alarm_Clock Dorm_Room,
  Link At Exam Classroom]

startDescription = "You are in your dorm room. Your math final exam is\n" ++
  "tomorrow, and you haven't studied at all. You need\n" ++
  "to pass! What are you going to do?"

instructions = "Enter commands using standard Prolog syntax.\n" ++
  "Available commands are:\n" ++
  "main         -- to start the game.\n" ++
  "in, out       -- to go in that direction.\n" ++
  "take <item>.      -- to pick up an object.\n" ++
  "drop <item>.      -- to put down an object.\n" ++
  "use <item>.       -- to use an object.\n" ++
  "look.          -- to look around you again.\n" ++
  "instructions.  -- to see this message again.\n" ++
  "sleep          -- to get some rest before the exam.\n" ++
  "halt.          -- to end the game and quit.\n"

endGameMessage = "Game Over!"

-- io operations

main = do
  putStrLn "Welcome to the game!"
  putStrLn instructions
  putStrLn startDescription
  repl initialState

repl :: ProlEnviron -> IO ()
repl env = do
  print "> "
  command <- getLine
  let (msg, newEnv) = processCommand command
  if ListUtils.isSuffixOf endGameMessage msg then return ()
  else do
    putStrLn msg
    repl newEnv

-- command processing

splitToArgs :: String -> [String]
splitToArgs = map toLower . filter (/="") . foldl (\(h:t) c -> if c == ' ' then []:t else (c:h):t) [""]

processCommand :: String -> ProlEnviron -> (String, ProlEnviron)
processCommand cstr env
    | null comms  = ("Please type a command.", env)
    | otherwise   = callCommand env (head comms) tail comms
    where comms = splitToArgs cstr

-- commands 

dontUnderstand = "I don't understand that argument."

callCommand :: ProlEnviron -> String -> [String] -> (String, ProlEnviron)
callCommand env "take" (astr:_) = case Parser.readMaybe astr::Object of 
  Just v -> cTake env v
  Nothing -> (dontUnderstand, env)
callCommand env "drop" (astr:_) = case Parser.readMaybe astr::Object of 
  Just v -> cDrop env v
  Nothing -> (dontUnderstand, env)
callCommand env "use" (astr:_) = case Parser.readMaybe astr::Object of 
  Just v -> cUse env v
  Nothing -> (dontUnderstand, env)
callCommand env "in" _ = cIn env
callCommand env "out" _ = cOut env
callCommand env "look" _ = cLook env
callCommand env "sleep" _ = cSleep env
callCommand env "instructions" _ = cInstructions env
callCommand env _ _ = ("I don't recognize that command.", env)

cTake :: ProlEnviron -> Object -> (String, ProlEnviron)
cTake env item
    | item == Bus && assert (Tag I_Am_At Bus) env         = go In env
    | item == Exam && assert (Tag I_Am_At Classroom) env  = examResult env
    | otherwise                                           = if assert (Tag Holding item) env then ("You're already holding it!", env)
      else if (case tagValue I_Am_At env of 
        Nothing -> False
        Just myloc -> assert (Link At item myloc) env) then
        (if assert (Tag Mounted item) env then ("That item is secured to the floor.", env)
        else ("Okay.", define (Tag Holding item) $ retract (Link At item myloc) env))
      else ("I don't see it here.", env)

cDrop :: ProlEnviron -> Object -> (String, ProlEnviron)
cDrop env item = 
  if assert (Tag Holding item) env then
    let Just myloc = tagValue I_Am_At env
    in ("Okay.", define (Link At item myloc) $ retract (Tag Holding item) env)
  else ("You're not holding that.", env)

cUse :: ProlEnviron -> Object -> (String, ProlEnviron)
cUse env item
    | not (assert (Link At item myloc) env || assert (Tag Holding item) env)  = ("I don't see that anywhere.", env)
    | item == Book && assert (Tag Holding Book) env                           = useBook env
    | item == Book                                                            = ("Maybe you should pick the book up?", env)
    | item == TV && assert (Flag Slept) env                                   = ("You have to catch the bus!", env)
    | item == TV && assert (Flag Late) env                                    = ("You don't feel like watching any more tv.", env)
    | item == TV                                                              = ("You watch six episodes of 'Keeping Up with the Kardashians.'\nYou think it would have probably been a better idea to study.", define (Flag Late) env)
    | item == Alarm_Clock && not $ assert (Flag Alarm_Set) env                = ("You set your alarm for the morning.", define (Flag Alarm_Set) env)
    | item == Exam                                                            = (examResult, env)
    | otherwise                                                               = ("You can't use that right now.", env)
    where Just myloc = tagValue (Tag I_Am_At) env

cIn :: ProlEnviron -> (String, ProlEnviron)
cIn env = go In env

cOut :: ProlEnviron -> (String, ProlEnviron)
cOut env = go Out env

cSleep :: ProlEnviron -> (String, ProlEnviron)
cSleep evn
    | not $ assert (Tag I_Am_At Dorm_Room) env  = ("You can't sleep here!", env)
    | assert (Flag Slept) env                   = ("You're already well rested.", env)
    | not $ assert (Flag Late) env              = ("You don't feel tired yet, there's still stuff to do.", env)
    | not $ assert (Flag Alarm_Set) env         = ("Oh no, you forgot to set your alarm and slept right through the test!\nYou'll flunk for sure.\n" ++ endGameMessage, env)
    | otherwise                                 = ("You decide to get in bed and go to sleep for the night.", define (ThreeLink Path Bus_Stop In Bus) $ retract (ThreeLink Path Bus_Stop In Dorm_Room) $ define (Flag Slept) env)

cInstructions :: ProlEnviron -> (String, ProlEnviron)
cInstructions env = (instructions, env)

cLook :: ProlEnviron -> (String, ProlEnviron)
cLook env = let Just myloc = tagValue Tag I_Am_At env
            in ((getDescription myloc env) ++ "\n\nYou see " ++ (prettyPrintList $ map show $ getItemsNearby myloc env) ++ " around you.", env)

-- other functions

getDescription :: Object -> ProlEnviron -> String
getDescription myloc env
    | myloc == Dorm_Room                          = "You are in your dorm room."
    | myloc == Bus_Stop && assert (Tag Slept) env = "You are at the bus stop, the bus driver is waiting patiently for you to get in."
    | myloc == Bus_Stop                           = "You are at the bus stop, but there's not much to do here. You got off school hours ago."
    | myloc == Bus                                = "You are in the bus, waiting to be delivered to your classroom. There is still some time before you arrive."
    | myloc == Classroom                          = "You are in the classroom, ready to take your exam. Hopefully you've prepared."
    | otherwise                                   = "I don't know where you are."

getItemsNearby :: Object -> ProlEnviron -> [Object]
getItemsNearby loc env = [i | Link At i l <- Set.toList env, l == loc]

applyPreposition :: String -> String
applyPreposition "" = ""
applyPreposition str@(c:_) = (if elem c "aeiou" then "an " else "a ") ++ str

prettyPrintList :: [String] -> String
prettyPrintList _ = "nothing"
prettyPrintList [e] = applyPreposition e
prettyPrintList lst = intercalate ", " (map applyPreposition $ init lst) ++ " and " ++ applyPreposition $ last list

examResult :: ProlEnviron -> (String, ProlEnviron)
examResult env = ((case tagValue Knowledge env of 
  Nothing -> "You didn't even study for the exam."
  Just k -> if k < Three then "Sorry, you flunked the exam.\n" ++ endGameMessage 
  else if k == Three then "Whew! You passed the exam! Good going!"
  else "Wow, you aced the exam! I didn't think that was possible!") ++ "\n" ++ endGameMessage, env)

go :: ProlEnviron -> Object -> (String, ProlEnviron)
go env direction = case tagValue I_Am_At env of 
  Nothing -> ("where are you?", env)
  Just here -> case threeLinkValue Path here direction env of
    Nothing -> ("You can't go that way.", env)
    Just there -> let newEnv = define (Tag I_Am_At there) $ retract (Tag I_Am_At here) env
                  in (fst $ cLook newEnv, newEnv)

useBook :: ProlEnviron -> (String, ProlEnviron)
useBook env
    | k == Zero                           = ("Hey, this is actually interesting!\nI wish I had looked at it before.", learned)
    | k == One                            = ("This is hard, but I'm starting to understand it.", learned)
    | k == Two                            = ("My brain is full...", define (Flag Late) learned)
    | assert (Flag Late) env              = ("(Yawn) I'm too sleepy to see straight", env)
    | assert (Tag I_Am_At Bus) env        = ("You review all the things that you studied last night.\nFairly soon, you get to your stop, and exit the bus.", )
    | assert (Tag I_Am_At Classroom) env  = let (movesay, newenv) = go learned Out in ("You're not allowed to use that during the test.\n" ++ movesay, newenv)
    where Just k = tagValue Knowledge env
          learned = let Just k = tagValue Knowledge env in define (Tag Knowledge $ succ k) $ retract (Tag Knowledge k) env