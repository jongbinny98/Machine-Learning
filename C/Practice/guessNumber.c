#include <stdio.h>
#include <stdlib.h> // generate random number
#include <time.h> //provide the seed for the randome number using the current time 

int main()
{
    int randomNumber = 0;
    int guess = 0;
    time_t t;

    //initialization of random number generator 
    srand((unsigned) time (&t));

    //get the random number
    randomNumber = rand() % 21;

    printf("This is the guessing game.\n");
    printf("I have chosen the number from 0 to 20.\n");

    for (int numberOfGuesses = 5; numberOfGuesses> 0; --numberOfGuesses)
    {
        printf("you have %d tr%s", numberOfGuesses, numberOfGuesses == 1 ? "y" : "ies\n");
        printf("Enter a guess: ");
        scanf("%d", &guess);

        if(guess == randomNumber)
        {
            printf("\ncongratulations. You guessed it!\n");
            break;
        }
        else if (guess < 0 || guess > 20)
            printf("I said the number is between 0 and 20.\n");
        else if (randomNumber > guess)
            printf("Sorry, %d is wrong. My number is greater than that.\n", guess);
        else if (randomNumber < guess)
            printf("Sorry, %d is wrong. My number is less than that.\n", guess);    
    }
    printf("you have had five tries and failed. The number was %d.", randomNumber);

    return 0;
}   
