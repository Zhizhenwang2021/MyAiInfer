//
// Created by lx186 on 2024/9/17.
//
#include <stdio.h>
#include "head.h"

int main()
{
    int a = 20;
    int b = 30;
    printf("a =%d, b = %d\n", a, b);
    printf("a + b = %d\n", add(a, b));
    printf("a - b = %d\n", subtract(a, b));
    return 0;
}