#include <stdlib.h>
#include <stdio.h>

int min(int num1, int num2) {
    return num1 < num2 ? num1 : num2;
}

int max(int num1, int num2) {
    return num1 > num2 ? num1 : num2;
}

int* createArray(int length) {
    int *arr = (int*)malloc(sizeof(int) * length);
    if(!arr) {
        printf("\nNo space left on device!");
        exit(1);
    }

    printf("\nEnter %d items for the array: \n", length);
    for (int i = 0; i < length; i++) scanf("%d", arr + i);

    return arr;
}

void printArray(int* arr, int length) {
    for(int i = 0; i < length; i++) {
        printf("%d ", *(arr + i));
    }
}

int* mergeArraySorted(int* arr1, int len1, int* arr2, int len2) {
    int* mergedArr = (int*)malloc(sizeof(int) * (len1 + len2));
    if(!mergedArr) {
        printf("No space left!");
        exit(1);
    }

    int i = 0, j = 0, k = 0;

    while (i < len1 && j < len2) mergedArr[k++] = arr1[i] < arr2[j] ? arr1[i++] : arr2[j++];
    while (i < len1) mergedArr[k++] = arr1[i++];
    while (j < len2) mergedArr[k++] = arr2[j++];

    return mergedArr;
}

int main() {
    int maxLen1, maxLen2;

    printf("\nEnter the length of the first array: ");
    scanf("%d", &maxLen1);
    int *arr1 = createArray(maxLen1);

    printf("\nEnter the length of the second array: ");
    scanf("%d", &maxLen2);
    int *arr2 = createArray(maxLen2);

    printf("\nArray 1:\n");
    printArray(arr1, maxLen1);
    printf("\nArray 2:\n");
    printArray(arr2, maxLen2);

    printf("\nMerged Array:\n");
    printArray(mergeArraySorted(arr1, maxLen1, arr2, maxLen2), maxLen1 + maxLen2);
}