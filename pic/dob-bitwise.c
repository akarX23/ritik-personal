#include <stdio.h>
#include <stdlib.h>

int yearBits = 23, monthBits = 4, dateBits = 5;

void printBits(int n) {
    for (int i = (int)sizeof(int) * 8 - 1; i >= 0; i--) 
        printf("%d", (n >> i) & 1);
    printf("\n");
}

char* unpackDOB(int packed) {
    int year = packed & 0x7FFFFF;
    int month = (packed >> yearBits) & 0xF;
    int date = packed >> (yearBits + monthBits);

    char* dob = (char*)malloc(20);
    snprintf(dob, 20, "%02d-%02d-%04d", date, month, year);
    
    return dob;
}

int main() {
    printf("Size of Int: %d\n", (int)sizeof(int));
    int date, month, year, combined = 0;

    printf("Enter Date\n");
    scanf("%d", &date);
    printf("Enter Month\n");
    scanf("%d", &month);
    printf("Enter Year\n");
    scanf("%d", &year);

    combined = combined | date;
    combined <<= (monthBits);

    combined |= month;
    combined <<= yearBits;

    combined |= year;
    
    printf("%s\n", unpackDOB(combined));
}