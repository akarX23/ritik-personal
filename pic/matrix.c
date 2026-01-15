#include <stdio.h>

void printArray(int *arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            printf("%d ", *(arr + i * size + j));
        }
        printf("\n");
    }
}

void multiply(int *a, int *b, int *c, int size)
{
    int sum = 0;
    for (int i = 0, ; i < size; ) {
        for (int j = 0; j < size; j++) {
            sum += *(a + i * size + j) * *(b + j * size + i);
        }

    }
}

int main()
{
    int size;

    printf("Enter the size of matrix: \n");
    scanf("%d", &size);

    int a[2][size][size], product[size][size];
    for (int mat = 0; mat < 2; mat++)
    {
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                printf("Enter the %d %d element for Matrix %d\n", (i + 1), (j + 1), (mat + 1));
                scanf("%d", (*(*(a + mat) + i) + j));
            }
        }
    }

    printArray(**a, size);
    printf("\n");
    printArray(**(a + 1), size);
}
