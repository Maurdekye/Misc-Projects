using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace monopoly
{
    static class Dice
    {
        public static Random RNG = new Random();

        public static int Roll()
        {
            return Roll(1);
        }

        public static int Roll(int amount)
        {
            int dieSum = 0;
            for (int i = 0; i < amount; i++)
            {
                dieSum += RNG.Next(6);
            }
            return dieSum;
        }
    }
}
