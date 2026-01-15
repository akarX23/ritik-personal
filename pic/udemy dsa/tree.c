#include <stdio.h>
#include <stdlib.h>

struct BstNode
{
    struct BstNode *left;
    int data;
    struct BstNode *right;
};

typedef struct BstNode Node;

int max(int a, int b) {
    return a > b ? a : b;
}

Node *createNewNode()
{
    Node *node = (Node *)calloc(1, sizeof(Node));
    if (!node)
    {
        printf("\nNot enough memory.");
        exit(1);
    }

    printf("\nEnter data: ");
    scanf("%d", &(node->data));
    node->left = NULL;
    node->right = NULL;

    return node;
}

void inOrderTraversal (Node** root) {
    if (!*root) return;

    inOrderTraversal(&((*root)->left));
    printf("%d ", (*root)->data);
    inOrderTraversal(&((*root)->right));
}

void insertNode(Node **root, Node *node)
{
    if (!node)
        node = createNewNode();

    if (!*root)
    {
        *root = node;
        return;
    }

    if ((*root)->data < node->data)
        insertNode(&((*root)->right), node);
    else
        insertNode(&((*root)->left), node);
}

void printTree (Node** root) {
    int height = getHeight(root);
    int lstHeight = getHeight((*root)->left);

    printf("\n");
    for (int i = 0; i < height; i++) {
        for (int spaces = 1; spaces < lstHeight; spaces++) printf(" ");

    }
}

int removeNode (Node** root, int dataToDelete) {
    
}

int getHeight (Node **root) {
    if (!*root) return 0;

    return 1 + max(getHeight((*root)->left), getHeight((*root)->right));
}

int main()
{
    Node *root = NULL;

    insertNode(&root, NULL);
    insertNode(&root, NULL);
    insertNode(&root, NULL);
    insertNode(&root, NULL);
    inOrderTraversal(&root);
}