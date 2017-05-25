#include <stdio.h>
#include <time.h>

#define NUM_PRIMES 1000

int isprime(int n, int * primelist)
{
    for (int i = 0; i < NUM_PRIMES; i = i + 1)
    {
        int prime = primelist[i];
        if (prime == 0)
        {
            break;
        }
        if (prime >= n)
        {
            return 1;
        }
        else if (n % prime == 0)
        {
            return 0;
        }
    }
    return 1;
}

int main()
{
    int primes[NUM_PRIMES];
    for (int i = 1; i < NUM_PRIMES; i = i + 1)
    {
        primes[i] = 0;
    }

    primes[0] = 2;

    int cur_index = 1;

    int i = 3;
    
    long unsigned int starttime = GetTickCount();

    while (1)
    {
        if (cur_index >= NUM_PRIMES)
        {
            break;
        }
        if (isprime(i, primes))
        {
            primes[cur_index] = i;
            cur_index = cur_index + 1;
        }
        i = i + 1;
    }

    printf("First 1000 primes;\n");
    for (int j = 0; j < 1000; j = j + 1)
    {
        printf("%d ", primes[j]);
    }

    long unsigned int endtime = GetTickCount();
    
    printf("\nTook %d seconds\n", endtime - starttime);
}
