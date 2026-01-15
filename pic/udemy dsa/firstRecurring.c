#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

#define EMPTY_SLOT INT_MIN

int findRecurring(int *arr, int length)
{
    int *hashmap = (int *)calloc(length, sizeof(int));
    if (!hashmap) {
        printf("\nMemory allocation failed!");
        exit(1);
    }

    for (int i = 0; i < length; i++) hashmap[i] = EMPTY_SLOT;

    for (int i = 0; i < length; i++)
    {
        unsigned int hashVal = arr[i] % length;
        while (hashmap[hashVal] != EMPTY_SLOT)
        {
            if (hashmap[hashVal] == arr[i])
                {
                    free(hashmap);
                    return arr[i];}
            hashVal = (hashVal + 1) % length;
        }
        hashmap[hashVal] = arr[i];
    }

    free(hashmap);
    return -1;
}

int main()
{
    int testArray1[] = {1, 2, 4};
    printf("\nTest Array 1: %d", findRecurring(testArray1, 3)); // Expected: -1

    int testArray2[] = {2, 5, 1, 2, 3};
    printf("\nTest Array 2: %d", findRecurring(testArray2, 5)); // Expected: 2

    int testArray3[] = {2, 1, 1, 2, 3, 5, 1, 2, 4};
    printf("\nTest Array 3: %d", findRecurring(testArray3, 9)); // Expected: 1

    int testArray4[] = {10, 20, 30, 10, 40};
    printf("\nTest Array 4: %d", findRecurring(testArray4, 5)); // Expected: 10
    
    int testArray5[] = {0, 1, 2, 0, 3};
    printf("\nTest Array 5: %d", findRecurring(testArray5, 5)); // Expected: 0

    return 0;
}