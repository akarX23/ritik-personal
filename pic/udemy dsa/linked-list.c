#include <stdlib.h>
#include <stdio.h>

struct Node
{
    char* name;
    struct Node *next;
};

typedef struct Node Node;

char *inputName()
{
    char *name = (char *)malloc(sizeof(char) * 256);
    if (!name)
    {
        printf("\nNo memory can be allocated!");
        exit(1);
    }
    printf("\nEnter name: ");
    scanf("%256s", name);

    return name;
}

Node *createNewNode()
{
    Node *node = (Node *)calloc(1, sizeof(Node));
    if (!node)
    {
        printf("\nNo memory can be allocated!");
        exit(1);
    }
    char *name = inputName();
    node->name = name;
    node->next = NULL;

    return node;
}

void prependList(Node **head)
{
    Node *node = createNewNode();
    if (!*head)
    {
        *head = node;
        return;
    }

    node->next = *head;
    *head = node;
}

void insertAtPos (Node **head, int pos) {

    if (pos == 0) {
        prependList(head);
        return;
    }

    int currPos = 0;
    Node* p = *head;
    while (currPos < pos - 1 && p) {
        p = p->next;
        currPos++;
    }

    if (!p) {
        printf("\nPosition not found in the list");
        return;
    }

    Node *temp = createNewNode();
    temp->next = p->next;
    p->next = temp;
}

void deleteAtPos (Node** head, int pos) {
    if (!*head) return;

    Node *temp = *head;
    if(pos == 0) {        
        *head = temp->next;
        free(temp->name);
        free(temp);
        return;
    }

    int currPos = 0;
    Node* prev = temp;
    while (temp && currPos < pos) {
        prev = temp;
        temp = temp->next;
        currPos++;
    }

    if (!temp) {
        printf("\nNode not found!");
        return;
    }

    prev->next = temp->next;
        free(temp->name);
    free(temp);
    return;
}

void printList(Node **head) {

    if (!*head) return;

    Node *p = *head;
    printf("\n");
    while(p->next) {
        printf("%s ---> ", p->name);
        p = p->next;
    }
    printf("%s\n", p->name);
}

void reverseList (Node** head) {
    if (!*head) return;

    Node *reversed = *head;
    *head = (*head)->next;
    reversed->next = NULL;

    while (*head) {
        Node* nextNode = (*head)->next;
        (*head)->next = reversed;
        reversed = *head;
        *head = nextNode;
    }

    *head = reversed;
}

int main()
{
    Node *head = NULL;

    prependList(&head);
    prependList(&head);
    printList(&head);
    insertAtPos(&head, 1);
    printList(&head);

    reverseList(&head);
    printList(&head);
}