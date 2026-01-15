#include <stdio.h>
#include <stdlib.h>

typedef struct node
{
    int data;
    struct node *next;
} Node;

Node *createNode(int data)
{
    Node *newNode = (Node *)malloc(sizeof(Node));
    if (!newNode)
    {
        printf("Memory not enough");
        exit(1);
    }

    newNode->data = data;
    newNode->next = NULL;

    return newNode;
}

void printList(Node *head)
{
    if (!head)
        printf("No nodes present");

    Node *p = head;
    while (p)
    {
        printf("%d -> ", p->data);
        p = p->next;
    }
}

Node *insertElements(Node *head, int n)
{
    Node *p = head;
    while (p && p->next)
        p = p->next;

    for (int i = 0; i < n; i++)
    {
        int data;
        printf("Enter data for node %d\n", (i + 1));
        scanf("%d", &data);
        Node *newNode = createNode(data);

        if (!head)
        {
            head = newNode;
            p = head;
        }
        else
        {
            p->next = newNode;
            p = p->next;
        }
    }

    p = NULL;
    return head;
}

void insertNodeAtPos(Node *head, int pos)
{
    if (pos == -1)
    {
        while (head->next)
            head = head->next;
    }

    else
    {
        int curPos = 0;
        while (curPos < pos - 1 && head->next)
        {
            head = head->next;
            curPos++;
        }

        if (curPos < pos)
        {
            printf("Position not found\n");
        }
    }

    printf("Enter data for new node: \n");
    int data;
    scanf("%d", &data);
    Node *newNode = createNode(data);

    newNode->next = head->next;
    head->next = newNode;
}

int countNodes(Node *head)
{
    int count = 0;
    while (head)
    {
        count++;
        head = head->next;
    }

    return count;
}

void deleteNode(Node *head, int pos)
{
    int curPos = 0;
    while (curPos < pos - 1 && head->next)
    {
        head = head->next;
        curPos++;
    }

    if (head->next)
    {
        Node *temp = head->next;
        head->next = head->next->next;
        free(temp);
        temp = NULL;
    }
    else
    {
        printf("Position not found");
        return;
    }
}

int main()
{
    int n;

    printf("Enter number of nodes in the Linked List: \n");
    scanf("%d", &n);

    Node *head = NULL;

    head = insertElements(head, n);
    printList(head);

    // insertNodeAtPos(head, 2);
    // printList(head);

    deleteNode(head, 2);
    printList(head);
}