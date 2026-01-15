#include <stdio.h>
#include <stdlib.h>

int strLen(char* str) {
    int len;
    for(len = 0; *(str + len) != '\0'; len++);
    return len;
}

char* rev_string(char* str) {
    int length = strLen(str);
    char* rev = (char*)malloc(sizeof(char) * length);
    if (!rev) {
        printf("\nNo space left on device!");
        exit(1);
    }

    for(int i = 0; i < length; i++) {
        *(rev + i) = *(str + length - i - 1);
    }

    return rev;
}

int main() {
    int maxLen;
    printf("\nEnter maximum length of string: ");
    if(scanf("%d", &maxLen) != 1) {
        printf("Invalid length!");
        exit(1);
    }
    while (getchar() != '\n' && getchar() != EOF);

    char str[maxLen + 1];

    char format[20];
    sprintf(format, "%%%ds", maxLen);
    printf("\nEnter your string: ");
    scanf(format, str);

    printf("\nLength of string: ");
    printf("%d", strLen(str));

    char* rev = rev_string(str);
    printf("\nReversed String: %s", rev);
}