#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void reverse(char* s) {
    char* begin = s;
    char* end = s;

    while (*end != '\0') end++;
    end--;

    while (begin <= end ) {
        char temp = *begin;
        *begin = *end;
        *end = temp;
        begin++;
        end--;
    }
}

int length(char* s) {
    int len = 0;

    char* p = s;
    while (*p != '\0') {
        len++;
        p++;
    }

    return len;
}

void copyStr(char* target, char* source) {
    char* p = source;

    while (*p != '\0') {
        *target = *p;
        target++;
        p++;
    }

    *target = '\0';
}

char* concatenate(char* s1, char* s2) {
    char* concat = (char*)malloc(length(s1) + length(s2) + 1);
    char* temp = concat;

    while ((*temp++ = *s1++) != '\0');
    temp--;

    while ((*temp++ = *s2++) != '\0');
   
    return concat;
}

void sortStringPointers(char **strings, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1; j++) {
            if (strcmp(*(strings + j), *(strings + j + 1)) > 0) {
                printf("Found unsorted");
                char* temp = *(strings + j);
                *(strings + j) = *(strings + j + 1);
                *(strings + j + 1) = temp;    
            }
        }
    }
}

void storeStrings(int n) {
    char *arr[n];

    printf("Enter %d strings:\n", n);
    for (int i = 0; i < n; i++) {
        *(arr + i) = (char*)malloc(100);
        scanf("%99s", *(arr + i));
    }

    return;
}

void printArgs(char **argv) {
    
}

int main() {
    int n;
    printf("Enter number of strings: \n");
    scanf("%d", &n);
    storeStrings(n);
}

    // char s[20], cpy[20] = " world";

    // printf("Enter a string to be used\n");
    // scanf("%19s", s);

    // printf("%s", concatenate(s, cpy));

    // char *s[] = {"mango", "banana", "aali", "nelson"};
    // int n = sizeof(s) / sizeof(*s);
    // sortStringPointers(s, n);

    // for (int i = 0; i < n; i++) {
    //     printf("%s\n", *(s + i));
    // }