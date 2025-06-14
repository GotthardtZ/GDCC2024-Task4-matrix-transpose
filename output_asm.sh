gcc -S -masm=intel -DNDEBUG -O2 -m64 -march=haswell -mtune=haswell -Wno-unused-result -no-pie -nostdlib -fno-unwind-tables -fno-asynchronous-unwind-tables transpose.c

