using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace monopoly
{
    static class Util
    {
        public static IEnumerable<U> FilterSubclass<T, U>(IEnumerable<T> list) where U : T
        {
            return list.Where(e => e is U) as IEnumerable<U>;
        }
    }
}
