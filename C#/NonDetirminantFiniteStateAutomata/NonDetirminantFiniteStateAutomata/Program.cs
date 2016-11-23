using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace NonDetirminantFiniteStateAutomata
{
    class Program
    {
        static void Main(string[] args)
        {
            List<Tuple<NDFA, List<string>>> NDFAs = new List<Tuple<NDFA, List<string>>>()
            {
                Tuple.Create(
                    new NDFA("A", "D", new List<Tuple<string, char, string>>()
                    {
                        Tuple.Create("A", '0', "A"),
                        Tuple.Create("A", '1', "A"),
                        Tuple.Create("A", '0', "B"),
                        Tuple.Create("B", '0', "D"),
                        Tuple.Create("A", '1', "C"),
                        Tuple.Create("C", '1', "D"),
                        Tuple.Create("D", '0', "D"),
                        Tuple.Create("D", '1', "D")
                    }),
                    new List<string>()
                    {
                        "101010",
                        "101101",
                        "10",
                        "00101010",
                        "0101",
                        ""
                    }
                ),
                Tuple.Create(
                    new NDFA("A", "D", new List<Tuple<string, char, string>>()
                    {
                        Tuple.Create("A", '0', "A"),
                        Tuple.Create("A", '1', "A"),
                        Tuple.Create("A", '1', "B"),
                        Tuple.Create("B", '0', "C"),
                        Tuple.Create("C", '1', "D")
                    }),
                    new List<string>()
                    {
                        "101010",
                        "101101",
                        "10",
                        "00101010",
                        "0101",
                        "00000101",
                        "10110110",
                        "00000000000",
                        "111111111",
                        "0000000001",
                        "00000101000",
                        "0000010101"
                    }
                ),
                Tuple.Create(
                    new NDFA("B", new List<string>() { "A", "D" }, new List<Tuple<string, char, string>>()
                    {
                        Tuple.Create("B", 'ε', "A"),
                        Tuple.Create("A", '1', "A"),
                        Tuple.Create("B", 'ε', "C"),
                        Tuple.Create("C", '0', "C"),
                        Tuple.Create("C", '1', "C"),
                        Tuple.Create("C", '0', "D")
                    }),
                    new List<string>()
                    {
                        "1",
                        "11",
                        "1111",
                        "11111111",
                        "111101111",
                        "11111110",
                        "01010001010",
                        "11010111011"
                    }
                )
            };
            foreach (var t in NDFAs)
            {
                Console.WriteLine();
                foreach (string inp in t.Item2)
                {
                    Tuple<bool, List<string>> result = t.Item1.EvaluateAsSetOfStates(inp);
                    Console.WriteLine($"{inp}: {result.Item1}, <-> {TextUtils.Lists.Printable(result.Item2)}");
                }
            }
            Console.Read();
        }
    }

    class NDFA
    {
        public string StartState;
        public List<string> GoalStates;
        public List<Tuple<string, char, string>> Transitions;

        public NDFA(string StartState, string GoalState, List<Tuple<string, char, string>> Transitions)
        {
            this.StartState = StartState;
            this.GoalStates = new List<string> { GoalState };
            this.Transitions = Transitions;
        }

        public NDFA(string StartState, List<string> GoalStates, List<Tuple<string, char, string>> Transitions)
        {
            this.StartState = StartState;
            this.GoalStates = GoalStates;
            this.Transitions = Transitions;
        }

        public static List<Tuple<NDFA, List<string>>> FromString(string InString)
        {
            bool givenStartState = false;
            bool givenGoalStates = false;
            bool givenTransitions = false;
            NDFA CurrentNDFA = new NDFA("", new List<string>(), new List<Tuple<string, char, string>>());
            List<string> CurrentInputs = new List<string>();

            List<Tuple<NDFA, List<string>>> NDFAs = new List<Tuple<NDFA, List<string>>>();

            foreach (string line in TextUtils.SplitBy.Lines(InString))
            {
                List<string> words = TextUtils.SplitBy.Words(line);
                string command = words.First();
                List<string> args = words.Skip(1).ToList();
                if (command == "START" && args.Count > 0)
                {
                    CurrentNDFA.StartState = words[1];
                    givenStartState = true;
                } 
                else if (command == "FINAL" && args.Count > 0)
                {
                    CurrentNDFA.GoalStates.AddRange(args);
                    givenGoalStates = true;
                }
                else if (command == "TRANSITIONS")
                {
                    string EntryState = "";
                    char Rule = ' ';
                    int Progress = 0;
                    foreach (string w in args)
                    {
                        if (Progress == 0)
                        {
                            EntryState = w;
                        }
                        else if (Progress == 1)
                        {
                            Rule = w.First();
                        }
                        else
                        {
                            CurrentNDFA.Transitions.Add(Tuple.Create(EntryState, Rule, w));
                            givenTransitions = true;
                            Progress = -1;
                        }
                        Progress += 1;
                    }
                }
                else if (command == "INPUT" && args.Count > 0)
                {
                    CurrentInputs.Add(string.Join("", args));
                }
                else if (command == "END")
                {
                    if (givenStartState && givenGoalStates && givenTransitions)
                    {
                        NDFAs.Add(Tuple.Create(CurrentNDFA, CurrentInputs));
                    }
                    givenStartState = false;
                    givenGoalStates = false;
                    givenTransitions = false;
                    CurrentNDFA = new NDFA("", new List<string>(), new List<Tuple<string, char, string>>());
                    CurrentInputs = new List<string>();
                }
            }

            return NDFAs;
        }

        public bool Evaluate(string Input)
        {
            return EvaluateWithSolutionTrail(Input).Item1;
        }

        public Tuple<bool, List<string>> EvaluateWithSolutionTrail(string Input)
        {
            return EvaluateWithSolutionTrail(Input, StartState, new List<string>());
        }

        private Tuple<bool, List<string>> EvaluateWithSolutionTrail(string Input, string State, List<string> Trail)
        {
            List<string> newTrail = new List<string>(Trail) { State };
            if (Input.Length == 0)
            {
                return Tuple.Create(GoalStates.Contains(State), newTrail);
            }

            List<Tuple<string, bool>> AvailableActions = GetActions(State, Input);
            if (AvailableActions.Count == 0)
            {
                return Tuple.Create(false, new List<string>());
            }

            foreach (var t in AvailableActions)
            {
                string ToGive = Input;
                if (t.Item2)
                {
                    ToGive = ToGive.Substring(1);
                }
                Tuple<bool, List<string>> evaluation = EvaluateWithSolutionTrail(ToGive, t.Item1, newTrail);
                if  (evaluation.Item1)
                {
                    return evaluation;
                }
            }
            return Tuple.Create(false, new List<string>());
        }

        public Tuple<bool, List<string>> EvaluateAsSetOfStates(string input)
        {
            HashSet<Tuple<string, string, List<string>>> statesToExplore = new HashSet<Tuple<string, string, List<string>>>() { Tuple.Create(StartState, input, new List<string>()) };
            while (true)
            {
                HashSet<Tuple<string, string, List<string>>> newStates = new HashSet<Tuple<string, string, List<string>>>();
                foreach (var stackframe in statesToExplore)
                {
                    if (stackframe.Item2 == "")
                    {
                        if (GoalStates.Contains(stackframe.Item1))
                        {
                            return Tuple.Create(true, stackframe.Item3);
                        }
                        else
                        {
                            continue;
                        }
                    }

                    List<Tuple<string, bool>> AvailableActions = GetActions(stackframe.Item1, stackframe.Item2);
                    if (AvailableActions.Count == 0)
                    {
                        continue;
                    }

                    foreach (var action in AvailableActions)
                    {
                        string ToGive = stackframe.Item2;
                        if (action.Item2)
                        {
                            ToGive = ToGive.Substring(1);
                        }
                        Tuple<string, string, List<string>> newItem = Tuple.Create(action.Item1, ToGive, new List<string>(stackframe.Item3) { action.Item1 });
                        newStates.Add(newItem);
                    }
                }
                if (newStates.Count == 0)
                {
                    return Tuple.Create(false, new List<string>());
                }
                statesToExplore = newStates;
            }
        }

        public List<Tuple<string, bool>> GetActions(string curState, string curInput)
        {
            List<Tuple<string, bool>> AvailableMovements = new List<Tuple<string, bool>>();
            foreach (var t in Transitions)
            {
                if (t.Item1 == curState)
                {
                    if (t.Item2 == 'ε')
                    {
                        AvailableMovements.Add(Tuple.Create(t.Item3, false));
                    }
                    else if (t.Item2 == curInput.First())
                    {
                        AvailableMovements.Add(Tuple.Create(t.Item3, true));
                    }
                }
            }
            return AvailableMovements;
        }
    }
}

namespace TextUtils
{
    public static class SplitBy
    {
        public static List<string> Lines(string text)
        {
            return text.Split('\n').Where(w => w != "").ToList();
        }

        public static List<string> Words(string text)
        {
            return Regex.Split(text, @"\s").Where(w => w != "").ToList();
        }
    }

    public static class Lists
    {
        public static string Printable(List<string> textlist)
        {
            return "[" + string.Join(", ", textlist.Select(s => "\"" + s + "\"")) + "]";
        }
    }
}
