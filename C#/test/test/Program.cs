using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace test
{
    class Program
    {
        static void Main(string[] args)
        {
            string input = "hello i am string\nnice to meet you on this fine evening\nyes\n\nhello\n";
            Console.WriteLine(input);
            List<List<string>> segments = lines(input).Select(l => words(l)).ToList();
            Console.WriteLine(ShowList(segments));
            Console.Read();
        }

        static string ShowList<T>(IEnumerable<T> iterable)
        {
            return "[" + string.Join(", ", iterable.Select(e => ShowList(e))) + "]";
        }

        static string ShowList(string str)
        {
            return $"\"${str}\"";
        }

        static string ShowList<T>(T elem)
        {
            return elem.ToString();
        }

        static List<string> lines(string text)
        {
            return text.Split('\n').Where(w => w != "").ToList();
        }

        static List<string> words(string text)
        {
            return Regex.Split(text, @"\s").Where(w => w != "").ToList();
        }
    }
}
