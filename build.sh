gcc -s -static -DNDEBUG -O2 -m64 -march=haswell -mtune=haswell -Wno-unused-result -no-pie -nostdlib -fno-unwind-tables -fno-asynchronous-unwind-tables syscall.S transpose.c -o transpose

