@echo off
set path=%path%;c:/mingw/winlibs-x86_64-posix-seh-gcc-9.3.0-llvm-10.0.0-mingw-w64-7.0.0-r4/bin

del _error.txt >nul 2>&1
del transpose.exe >nul 2>&1

g++.exe -static -s -O3 -march=haswell -mtune=haswell transpose.cpp -o transpose.exe 2>_error.txt
pause
