using System.Collections.Generic;

namespace monopoly
{
    interface Proprietor<T>
    {
        int GetCapital();
        IEnumerable<T> GetProperties();
        void ModifyCapital(int amount);
    }
}