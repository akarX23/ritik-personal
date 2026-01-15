#include <stdio.h>

int factorial(int n) {
    if(!n) return 1;

    return n * factorial(n - 1);
}

int printFibonacci(int n) {
    if(n  <= 1) {
        // printf("%d", n); 
        return n;
    }

    int seq = printFibonacci(n-1) + printFibonacci(n-2);
    printf("%d", (seq));

    return seq;
}

int sumOfDigits(int n) {
    if (n == 0) return 0;

    return n % 10 + sumOfDigits(n / 10);
}

void reverseString(char* s) {
    if(!*(s + 1)) {
        printf("%c", *s);
        return;
    } 

    reverseString(s + 1);
    printf("%c", *s);
}

int main() {
    char s[20];

    printf("Enter a word: \n");
    scanf("%19s", s);

    reverseString(s);
    // printf("Sum of Digits = %d", sumOfDigits(n));

}