#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int isprime(int n)
{
  for (int i=2;i<=n/2;i=i+1)
  {
    if (n % i == 0)
    {
      return 0;
    }
  }
  return 1;
}

int shareitem(int * lA, int * lB)
{
  for (int i = 0; i < 20; i = i + 1)
  {
    for (int j = 0; j < 20; j = j + 1)
    {
      if (lA[i] == lB[j] && lA[i] != 0 && lB[j] != 0)
      {
        return 1;
      }
    }
  }
  return 0;
}

int dispArr(int * arr, int size)
{
  printf("[");
  int ind = 0;
  while (ind < size - 1)
  {
    printf("%d, ", arr[ind]);
    ind = ind + 1;
  }
  printf("%d]", arr[ind]);
}

int doFactors(int n, int * arr, int primes[1000])
{
  for (int i = 0; i < 20; i = i + 1)
  {
    arr[i] = 0;
  }

  int arrindex = 0;
  int cont = 1;
  while (cont)
  {
    //getchar();
    //printf("Current array: ");
    //dispArr(arr, 20);
    //printf("\nCurrent num: %d\n", n);
    for (int cprime = 0; cprime < 1000; cprime = cprime + 1)
    {
      //printf("\tchecking prime %d\n", primes[cprime]);
      if (primes[cprime] >= n)
      {
        //printf("num reached smallest size of %d, leaving\n", n);
        cont = 0;
        break;
      }
      if (n % primes[cprime] == 0)
      {
        //printf("found divisible prime %d, adding to list at position %d\n", primes[cprime], arrindex);
        arr[arrindex] = primes[cprime];
        arrindex = arrindex + 1;
        n = n / primes[cprime];
        break;
      }
    }

  }
  arr[arrindex] = n;
  return 0;
}

int main() 
{
  printf("initializing program\n");
  int primenumbers[1000];
  int index = 0;
  int cnum = 2;

  srand((unsigned)time(NULL));

  printf("generating primes numbers\n");
  // create prime list
  while (1)
  {
    if (isprime(cnum))
    {
      primenumbers[index] = cnum;
      index = index + 1;
      if (index == 1000)
      {
        break;
      }
    }
    cnum = cnum + 1;
  }
  
  printf("finished, generating number pairs\n");
  // generate coprimes and cofactors
  int pfactsA[64]; 
  int pfactsB[64];

  int cofacts = 1;
  int coprimes = 1;

  int iteration = 0;

  while (1)
  {
    int numA = rand()%1000;
    int numB = rand()%1000;
    //printf("numbers: %d and %d\n", numA, numB);
    doFactors(numA, pfactsA, primenumbers);
    doFactors(numB, pfactsB, primenumbers);
    /*printf("factors: ");
    dispArr(pfactsA, 20);
    printf(" and ");
    dispArr(pfactsB, 20);
    printf(".\n");
    printf("Do they share a common factor? ");*/
    if (shareitem(pfactsA, pfactsB))
    {
      //printf("Yes.");
      coprimes = coprimes + 1;
    }
    else
    {
      //printf("No. ");
      cofacts = cofacts + 1;
    }
    //printf("\n");
    //getchar();
    iteration = iteration + 1;
    if (iteration % 1000000 == 0)
    {
      float result = ((float)coprimes) / ((float)cofacts);
      printf("%d, %d: ratio of %f\n", coprimes, cofacts, result);
    }
  }

  return 0;
}
