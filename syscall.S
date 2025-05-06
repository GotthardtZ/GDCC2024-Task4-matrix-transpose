.intel_syntax noprefix
.text
    .globl _start
    .globl syscall0, syscall1, syscall2, syscall3, syscall4, syscall5, syscall6

    _start:
        xor rbp, rbp
        pop rdi
        mov rsi, rsp
        and rsp, -16
        call main
        mov rdi, rax
        mov rax, 60       # SYS_exit
        syscall
        ret

    syscall0:
        mov rax, rdi      # syscall number
        syscall
        ret

    syscall1:
        mov rax, rdi      # syscall number
        mov rdi, rsi      # arg1
        syscall
        ret

    syscall2:
        mov rax, rdi      # syscall number
        mov rdi, rsi      # arg1
        mov rsi, rdx      # arg2
        syscall
        ret

    syscall3:
        mov rax, rdi      # syscall number
        mov rdi, rsi      # arg1
        mov rsi, rdx      # arg2
        mov rdx, rcx      # arg3
        syscall
        ret

    syscall4:
        mov rax, rdi      # syscall number
        mov rdi, rsi      # arg1
        mov rsi, rdx      # arg2
        mov rdx, rcx      # arg3
        mov r10, r8       # arg4
        syscall
        ret

    syscall5:
        mov rax, rdi      # syscall number
        mov rdi, rsi      # arg1
        mov rsi, rdx      # arg2
        mov rdx, rcx      # arg3
        mov r10, r8       # arg4
        mov r8, r9        # arg5
        syscall
        ret

    syscall6:
        mov rax, rdi      # syscall number
        mov rdi, rsi      # arg1
        mov rsi, rdx      # arg2
        mov rdx, rcx      # arg3
        mov r10, r8       # arg4
        mov r8, r9        # arg5
        mov r9, [rsp+8]   # arg6
        syscall
        ret
