#include <stdio.h>
#include <unistd.h>

int main() {
    printf("Hello world!\n");
    char dir[1000];
    getcwd(dir, sizeof(dir));
    fprintf(stdout, "current dir: %s\n", dir);
}
