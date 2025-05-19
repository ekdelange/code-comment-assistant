#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#define MAX_HISTORY 100

char *history[MAX_HISTORY];
int history_count = 0;

void add_history(const char *entry) {
    if (history_count < MAX_HISTORY) {
        history[history_count++] = strdup(entry);
    }
}

double add(double a, double b) {
    double result = a + b;
    char buffer[100];
    snprintf(buffer, sizeof(buffer), "Added %.2f + %.2f = %.2f", a, b, result);
    add_history(buffer);
    return result;
}

bool divide(double a, double b, double *result) {
    if (b == 0) {
        return false;
    }
    *result = a / b;
    char buffer[100];
    snprintf(buffer, sizeof(buffer), "Divided %.2f / %.2f = %.2f", a, b, *result);
    add_history(buffer);
    return true;
}

void fibonacci(int n) {
    if (n <= 0) return;

    long long a = 0, b = 1;
    printf("Fibonacci sequence: ");
    for (int i = 0; i < n; i++) {
        printf("%lld ", a);
        long long temp = a;
        a = b;
        b = temp + b;
    }
    printf("\n");
}

bool is_prime(int number) {
    if (number < 2) return false;
    for (int i = 2; i <= (int)sqrt(number); i++) {
        if (number % i == 0) return false;
    }
    return true;
}

int main() {
    printf("Add: %.2f\n", add(4.5, 2.3));

    double result;
    if (divide(10.0, 2.0, &result)) {
        printf("Divide: %.2f\n", result);
    } else {
        printf("Cannot divide by zero.\n");
    }

    fibonacci(10);

    int number = 29;
    printf("%d is %s\n", number, is_prime(number) ? "prime" : "not prime");

    printf("\nOperation history:\n");
    for (int i = 0; i < history_count; i++) {
        printf("%s\n", history[i]);
        free(history[i]);
    }

    return 0;
}
