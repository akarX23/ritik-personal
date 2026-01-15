#include <stdio.h>
#include <stdlib.h>

typedef struct node
{
    int data;
    struct node *prev;
    struct node *next;
} Node;

Node *createNewNode(int data)
{
    Node *newNode = (Node *)malloc(sizeof(Node));
    if (!newNode)
    {
        printf("Not enough space!");
        exit(1);
    }

    newNode->data = data;
    newNode->prev = NULL;
    newNode->next = NULL;

    return newNode;
}

void printList(Node *head)
{
    while (head)
    {
        printf("%d->", head->data);
        head = head->next;
    }
}

void createNodes(Node **head, int n)
{
    Node *p = *head;

    for (int i = 0; i < n; i++)
    {
        int data;
        printf("\nEnter data for node %d:\n", (i + 1));
        scanf("%d", &data);
        Node *newNode = createNewNode(data);

        if (!*head)
        {
            *head = newNode;
            p = newNode;
        }
        else
        {
            p->next = newNode;
            newNode->prev = p;
            p = p->next;
        }
    }
}

void deleteNode(Node *head, int pos)
{
    int curPos = 0;
    while (curPos < pos - 1 && head)
    {
        head = head->next;
        curPos++;
    }

    if (!head->next)
    {
        printf("Position Not found");
        return;
    }

    Node *temp = head->next;
    head->next = head->next->next;
    head->next->prev = head;
    free(temp);
    temp = NULL;
}

int main()
{
    int n;

    printf("Enter number of nodes to be inserted: \n");
    scanf("%d", &n);

    Node *head = NULL;
    createNodes(&head, n);
    printList(head);

    deleteNode(head, 2);
    printList(head);
}