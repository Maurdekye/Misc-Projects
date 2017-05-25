#include <stdio.h>

int isprime(int n)
{
    for (int i = 2; i <= n/2; i = i + 1)
    {
        if (n % i == 0)
        {
            return 0;
        }
    } 
    return 1;
}

int main()
{
    int numfound = 0;
    int num = 2;
    while (1)
    {
        if (isprime(num))
        {
            numfound = numfound + 1;
            if (numfound % 1000 == 0)
            {
                printf("prime number #%d: %d\n", numfound, num);
            }
        }
        num = num + 1;
    }
}
