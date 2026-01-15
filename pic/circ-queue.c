#include <stdio.h>

void printQueue(int *q, int front, int rear, int size)
{
    while (front != rear)
    {
        printf("%d ", q[front]);
        front = (front + 1) % size;
    }
    printf("%d\n", q[front]);
}

void push(int *q, int *front, int *rear, int size, int data)
{
    if ((*rear + 1) % size == *front)
    {
        printf("verflow");
        return;
    }

    if (*front == -1)
        *front = 0;

    *rear = (*rear + 1) % size;
    *(q + *rear) = data;
}

int pop(int *q, int *front, int *rear, int size)
{
    if (*front == -1)
    {
        printf("Underflow");
        return -1;
    }

    int temp = *(q + *front);
    if (*front == *rear)
    {
        *front = -1;
        *rear = -1;
    }
    else
        *front = (*front + 1) % size;
}

int main()
{
    int size = 5;
    int queue[size], front = -1, rear = -1;

    push(queue, &front, &rear, size, 10);
    push(queue, &front, &rear, size, 20);
    push(queue, &front, &rear, size, 30);
    push(queue, &front, &rear, size, 40);
    push(queue, &front, &rear, size, 50);

    printQueue(queue, front, rear, size);
    push(queue, &front, &rear, size, 60);

    pop(queue, &front, &rear, size);
    printQueue(queue, front, rear, size);
}