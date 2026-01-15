#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

struct Array {
    int maxLength;
    int *arr;
    int currLen;
};

void initArray(struct Array *store) {
    printf("Enter length of array: \n");
    scanf("%d", &store->maxLength);
    store->currLen = -1;

    store->arr = (int *)malloc(sizeof(int) * store->maxLength);
    if (!store->arr) {
        printf("No space available to allocate memory!");
        exit(1);
    }
}

void push(struct Array *store) {
    if(store->currLen == store->maxLength - 1) {
        printf("Overflow");
        return;
    }

    int ele;
    printf("\nEnter element to push in the array: ");
    scanf("%d", &ele);

    (store->currLen)++;
    *(store->arr + store->currLen) = ele;
}

void pop(struct Array *store) {
    if (store->currLen == -1) {
        printf("Underflow!");
        return;
    }

    store->currLen--;
}

void printArray(struct Array *store) {
    if (store->currLen == -1) {
        printf("\nEmpty Array!");
        return;
    }

    for (int i = 0; i < store->currLen; i++) {
        printf("%d ---> ", *(store->arr + i));
    }
    printf("%d", *(store->arr + store->currLen));
    printf("\n");
}

void deleteAtPos(struct Array *store) {
    if (store->currLen == -1) {
        printf("\nUnderflow");
        return;
    }

    int delPos;
    printf("\nEnter the position to delete the element from: ");
    scanf("%d", &delPos);

    if(delPos > store->currLen || delPos < 0) {
        printf("Invalid position entered!");
        return;
    }

    while (delPos < store->currLen) {
        *(store->arr + delPos) = *(store->arr + delPos + 1);
        delPos++;
    }

    store->currLen--;
}

void insertAtPos(struct Array *store) {
    if (store->currLen == store->maxLength - 1) {
        printf("\nOverflow!");
        return;
    }

    int pos;
    printf("\nEnter the position to insert the element in: ");
    scanf("%d", &pos);
    if (pos >= store->maxLength || pos > (store->currLen + 1)) {
        printf("\nInvalid Position");
        return;
    }

    int ele;
    printf("\nEnter element to insert: ");
    scanf("%d", &ele);

    store->currLen++;
    for (int i = store->currLen; i > pos; i--) {
        *(store->arr + i) = *(store->arr + i - 1);
    }

    *(store->arr + pos) = ele;
}

int pickMenuOption() {
    int select;

    printf("\n\nPick one option from below:\n");
    printf("1) Insert element at the end\n");
    printf("2) Insert element at a position\n");
    printf("3) Delete element from the end\n");
    printf("4) Delete element at a position\n");
    printf("5) Print the array\n");
    printf("6) Exit\n\n");

    scanf("%d", &select);
    return select;
}

int main() {
   struct Array store;

    initArray(&store);

    bool userActive = true;
    int option;
    while(userActive) {
        option = pickMenuOption();

        switch (option) {
            case 1:
                push(&store);
                break;
            case 2:
                insertAtPos(&store);
                break;
            case 3:
                pop(&store);
                break;
            case 4:
                deleteAtPos(&store);
                break;
            case 5:
                printArray(&store);
                break;
            case 6:
                userActive = false;
                break;
            default:
                printf("\nInvalid Option. Please choose again!");
        }
    }
}