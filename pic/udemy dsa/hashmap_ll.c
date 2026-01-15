#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define DEFAULT_HASHMAP_SIZE 3

struct Person
{
    char name[256];
    int age;
    struct Person *next;
};

typedef struct Person person;

struct HashObject
{
    size_t size;
    size_t maxChainLength;
    person **hashmap;
    double maxLoadFactor;
    size_t threshold;
    size_t capacity;
};

typedef struct HashObject hashObject;

unsigned int hash(char *name, int map_capacity)
{
    unsigned int sum = 0;
    for (int i = 0; name[i] != '\0'; i++)
    {
        sum += name[i];
        sum = (sum * name[i]) % map_capacity;
    }

    return sum;
}

void resizeHashMap(hashObject *obj)
{
    int longestChain = 0;
    for (int i = 0; i < obj->capacity; i++)
    {
        int currLength = 0;
        person *currLink = obj->hashmap[i];
        while (currLink)
        {
            currLength++;
            currLink = currLink->next;
        }
        if (currLength > longestChain)
            longestChain = currLength;
    }

    if (obj->size < obj->threshold && longestChain < obj->maxChainLength)
        return;

    int resizedCapacity = (int)(obj->capacity * 2);
    person **resizedMap = (person **)calloc(resizedCapacity, sizeof(person *));
    if (!resizedMap)
    {
        printf("\nNo memory available to resize the HashMap!!");
        exit(1);
    }

    for (int i = 0; i < obj->capacity; i++)
    {
        person *entry = obj->hashmap[i];
        while (entry)
        {
            int newHash = hash(entry->name, resizedCapacity);
            person *nextEntry = entry->next;
            entry->next = resizedMap[newHash];
            resizedMap[newHash] = entry;
            entry = nextEntry;
        }
    }

    free(obj->hashmap);
    obj->capacity = resizedCapacity;
    obj->threshold = resizedCapacity * obj->maxLoadFactor;
    if (longestChain > obj->maxChainLength)
        obj->maxChainLength += 2;
    obj->hashmap = resizedMap;
}

person *createNewPerson()
{
    person *newPerson = (person *)malloc(sizeof(person));

    printf("\nEnter name for new person: ");
    scanf("%256s", newPerson->name);

    printf("Enter age: ");
    scanf("%d", &(newPerson->age));

    newPerson->next = NULL;

    return newPerson;
}

void pushIntoMap(hashObject *obj)
{
    person *new_person = createNewPerson();
    unsigned int hash_val = hash(new_person->name, obj->capacity);

    person *current = obj->hashmap[hash_val];
    while (current)
    {
        if (strcmp(current->name, new_person->name) == 0)
        {
            current->age = new_person->age;
            free(new_person);
            return;
        }
        current = current->next;
    }

    new_person->next = obj->hashmap[hash_val];
    obj->hashmap[hash_val] = new_person;
    obj->size++;

    resizeHashMap(obj);
}

void printMap(hashObject *obj)
{
    printf("\nHASHMAP Start-----------------------------\n\n");

    for (int i = 0; i < obj->capacity; i++)
    {
        if (!obj->hashmap[i])
            printf("\t%d) -------", i + 1);
        else
        {
            person *curr = obj->hashmap[i];
            printf("\t%d) ", i + 1);
            while (curr->next)
            {
                printf("%s ---> ", curr->name);
                curr = curr->next;
            }
            // Print last element without link
            printf("%s", curr->name);
        }
        printf("\n");
    }

    printf("\nHASHMAP End-------------------------------\n");
}

char *acceptName()
{
    char *name = (char *)malloc(sizeof(char) * 256);
    printf("\nEnter name to perform operation: ");
    scanf("%256s", name);

    return name;
}

person *deleteElement(hashObject *obj)
{
    char *name = acceptName();
    unsigned int hash_val = hash(name, obj->capacity);

    person *prev = NULL;
    person *curr = obj->hashmap[hash_val];

    while (curr)
    {
        if (strncmp(name, curr->name, 256) == 0)
        {
            // delete the head
            if (!prev)
                obj->hashmap[hash_val] = curr->next;
            // delete from middle
            else
                prev->next = curr->next;
            obj->size--;
            free(name);
            return curr;
        }
        prev = curr;
        curr = curr->next;
    }

    free(name);
    printf("\n%s not found in the Hashmap!!\n", name);
    return NULL;
}

person *search(hashObject *obj)
{
    char *name = acceptName();

    int hash_val = hash(name, obj->capacity);
    person *curr = obj->hashmap[hash_val];

    while (curr)
    {
        if (strncmp(name, curr->name, 256) == 0)
            return curr;
        curr = curr->next;
    }

    printf("\n%s not found in the Hashmap!!\n", name);
    return NULL;
}

int pickMenuOption()
{
    int select;

    printf("\n\nPick one option from below:\n");
    printf("1) Insert element\n");
    printf("2) Delete element\n");
    printf("3) Search for an element\n");
    printf("4) Print the HashMap\n");
    printf("5) Exit\n\n");

    if (scanf("%d", &select) != 1)
    {
        int c;
        while ((c = getchar()) != '\n' && c != EOF)
            ;
        return 10;
    }
    return select;
}

int main()
{
    hashObject obj = {.capacity = DEFAULT_HASHMAP_SIZE, .size = 0, .maxLoadFactor = 0.75, .maxChainLength = 5};
    obj.threshold = obj.capacity * obj.maxLoadFactor;
    obj.hashmap = (person **)calloc(obj.capacity, sizeof(person *));
    if (!obj.hashmap)
    {
        printf("\nNo space left!!");
        exit(1);
    }

    bool userActive = true;
    int option;
    while (userActive)
    {
        option = pickMenuOption();

        switch (option)
        {
        case 1:
            pushIntoMap(&obj);
            break;
        case 2:
            person *deleted = deleteElement(&obj);
            if (deleted)
                free(deleted);
            break;
        case 3:
            person *found = search(&obj);
            if (found)
                printf("\nFound with age: %d", found->age);
            break;
        case 4:
            printMap(&obj);
            break;
        case 5:
            userActive = false;
            break;
        default:
            printf("\nInvalid Option. Please choose again!");
            break;
        }
    }
}