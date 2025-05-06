#define PRINT_ERRORS false

#include <stdint.h> // for uint16_t, uint32_t, uintptr_t

#define false 0
#define true  1

typedef unsigned long int size_t;
typedef long int ssize_t;
typedef long int off_t;

// External syscall functions provided by syscall.S
extern ssize_t syscall1(int syscall_number, uintptr_t arg1);
extern ssize_t syscall2(int syscall_number, uintptr_t arg1, uintptr_t arg2);
extern ssize_t syscall3(int syscall_number, uintptr_t arg1, uintptr_t arg2, uintptr_t arg3);
extern ssize_t syscall4(int syscall_number, uintptr_t arg1, uintptr_t arg2, uintptr_t arg3, uintptr_t arg4);
extern ssize_t syscall5(int syscall_number, uintptr_t arg1, uintptr_t arg2, uintptr_t arg3, uintptr_t arg4, uintptr_t arg5);
extern ssize_t syscall6(int syscall_number, uintptr_t arg1, uintptr_t arg2, uintptr_t arg3, uintptr_t arg4, uintptr_t arg5, uintptr_t arg6);

// Constants for mmap
#define NULL 0
#define PROT_READ  0x1
#define PROT_WRITE 0x2
#define MAP_PRIVATE 0x02
#define MAP_SHARED  0x01
#define MAP_FAILED  ((void*)-1)

// Constants for open
#define O_RDONLY 0x0
#define O_RDWR   0x2
#define O_CREAT  0x40

// Syscall functions

// open
static int open(const char* pathname, int flags, int mode) {
  const int SYS_open = 2;  // SYS_open syscall number on Linux
  return (int)syscall3(SYS_open, (uintptr_t)pathname, (uintptr_t)flags, (uintptr_t)mode);
}

// read
static ssize_t read(int fd, void* buf, uintptr_t count) {
  const int SYS_read = 0;  // SYS_read syscall number on Linux
  return syscall3(SYS_read, (uintptr_t)fd, (uintptr_t)buf, count);
}

// write
static ssize_t write(int fd, const void* data, uintptr_t nbytes) {
  const int SYS_write = 1; // SYS_write syscall number on Linux
  return syscall3(SYS_write, fd, (uintptr_t)data, nbytes);
}

// close
static int close(int fd) {
  const int SYS_close = 3;  // SYS_close syscall number on Linux
  return (int)syscall1(SYS_close, (uintptr_t)fd);
}

// mmap
static void* mmap(void* addr, uintptr_t length, int prot, int flags, int fd, off_t offset) {
  const int SYS_mmap = 9;  // SYS_mmap syscall number on Linux
  return (void*)syscall6(SYS_mmap, (uintptr_t)addr, length, (uintptr_t)prot, (uintptr_t)flags, (uintptr_t)fd, (uintptr_t)offset);
}

// ftruncate
static int ftruncate(int fd, off_t length) {
  const int SYS_ftruncate = 77;  // SYS_ftruncate syscall number on Linux
  return (int)syscall2(SYS_ftruncate, (uintptr_t)fd, (uintptr_t)length);
}

// munmap
static int munmap(void* addr, uintptr_t length) {
  const int SYS_munmap = 11;  // SYS_munmap syscall number on Linux
  return (int)syscall2(SYS_munmap, (uintptr_t)addr, length);
}

// sbrk
static void* sbrk(intptr_t increment) {
  const int SYS_brk = 12;  // SYS_brk syscall number on Linux
  uintptr_t current_brk = (uintptr_t)syscall1(SYS_brk, 0); // Get current program break (top of heap)
  if (current_brk != (uintptr_t)-1 && increment != 0) {
    uintptr_t new_brk = current_brk + increment;
    uintptr_t current_brk = (uintptr_t)syscall1(SYS_brk, new_brk);
  }
  return (void*)current_brk;
}

// _exit
__attribute__((noreturn)) 
static void _exit(int status) {
  const int SYS_exit = 60;  // SYS_exit syscall number on Linux
  syscall1(SYS_exit, (uintptr_t)status);
  __builtin_unreachable();  // Inform the compiler this function never returns
}

// Implementations from <emmintrin.h>

typedef long long __v2di __attribute__((__vector_size__(16)));
typedef int __v4si __attribute__((__vector_size__(16)));
typedef short __v8hi __attribute__((__vector_size__(16)));

typedef long long __m128i __attribute__((__vector_size__(16), __may_alias__));
typedef long long __m128i_u __attribute__((__vector_size__(16), __may_alias__, __aligned__(1)));

static __inline __m128i __attribute__((__gnu_inline__, __always_inline__, __artificial__))
_mm_load_si128(__m128i const* __P)
{
  return *__P;
}

static __inline __m128i __attribute__((__gnu_inline__, __always_inline__, __artificial__))
_mm_loadu_si128(__m128i_u const* __P)
{
  return *__P;
}

static __inline void __attribute__((__gnu_inline__, __always_inline__, __artificial__))
_mm_store_si128(__m128i* __P, __m128i __B)
{
  *__P = __B;
}

static __inline void __attribute__((__gnu_inline__, __always_inline__, __artificial__))
_mm_storeu_si128 (__m128i_u *__P, __m128i __B)
{
  *__P = __B;
}

static __inline __m128i __attribute__((__gnu_inline__, __always_inline__, __artificial__))
_mm_unpacklo_epi16(__m128i __A, __m128i __B)
{
  return (__m128i)__builtin_ia32_punpcklwd128((__v8hi)__A, (__v8hi)__B);
}

static __inline __m128i __attribute__((__gnu_inline__, __always_inline__, __artificial__))
_mm_unpackhi_epi16(__m128i __A, __m128i __B)
{
  return (__m128i)__builtin_ia32_punpckhwd128((__v8hi)__A, (__v8hi)__B);
}

static __inline __m128i __attribute__((__gnu_inline__, __always_inline__, __artificial__))
_mm_unpacklo_epi32(__m128i __A, __m128i __B)
{
  return (__m128i)__builtin_ia32_punpckldq128((__v4si)__A, (__v4si)__B);
}

static __inline __m128i __attribute__((__gnu_inline__, __always_inline__, __artificial__))
_mm_unpackhi_epi32(__m128i __A, __m128i __B)
{
  return (__m128i)__builtin_ia32_punpckhdq128((__v4si)__A, (__v4si)__B);
}

static __inline __m128i __attribute__((__gnu_inline__, __always_inline__, __artificial__))
_mm_unpacklo_epi64(__m128i __A, __m128i __B)
{
  return (__m128i)__builtin_ia32_punpcklqdq128((__v2di)__A, (__v2di)__B);
}

static __inline __m128i __attribute__((__gnu_inline__, __always_inline__, __artificial__))
_mm_unpackhi_epi64(__m128i __A, __m128i __B)
{
  return (__m128i)__builtin_ia32_punpckhqdq128((__v2di)__A, (__v2di)__B);
}

static void swap(uint16_t* a, uint16_t* b) {
  uint16_t temp = *a;
  *a = *b;
  *b = temp;
}

static void transpose8x8SSE2_inplace(uint16_t* matrix_src, uint16_t* matrix_dst, size_t width) {
  // Load rows of the matrix into SSE registers
  __m128i row0 = _mm_loadu_si128((__m128i*) & matrix_src[0]);  // Row 0
  __m128i row1 = _mm_loadu_si128((__m128i*) & matrix_src[width]);  // Row 1
  __m128i row2 = _mm_loadu_si128((__m128i*) & matrix_src[width*2]); // Row 2
  __m128i row3 = _mm_loadu_si128((__m128i*) & matrix_src[width*3]); // Row 3
  __m128i row4 = _mm_loadu_si128((__m128i*) & matrix_src[width*4]); // Row 4
  __m128i row5 = _mm_loadu_si128((__m128i*) & matrix_src[width*5]); // Row 5
  __m128i row6 = _mm_loadu_si128((__m128i*) & matrix_src[width*6]); // Row 6
  __m128i row7 = _mm_loadu_si128((__m128i*) & matrix_src[width*7]); // Row 7

  // Transpose step 1: Unpack 16-bit elements (interleave within pairs of rows)
  __m128i t0 = _mm_unpacklo_epi16(row0, row1);
  __m128i t1 = _mm_unpackhi_epi16(row0, row1);
  __m128i t2 = _mm_unpacklo_epi16(row2, row3);
  __m128i t3 = _mm_unpackhi_epi16(row2, row3);
  __m128i t4 = _mm_unpacklo_epi16(row4, row5);
  __m128i t5 = _mm_unpackhi_epi16(row4, row5);
  __m128i t6 = _mm_unpacklo_epi16(row6, row7);
  __m128i t7 = _mm_unpackhi_epi16(row6, row7);

  // Transpose step 2: Unpack 32-bit elements (interleave pairs of 16-bit results)
  __m128i tt0 = _mm_unpacklo_epi32(t0, t2);
  __m128i tt1 = _mm_unpackhi_epi32(t0, t2);
  __m128i tt2 = _mm_unpacklo_epi32(t1, t3);
  __m128i tt3 = _mm_unpackhi_epi32(t1, t3);
  __m128i tt4 = _mm_unpacklo_epi32(t4, t6);
  __m128i tt5 = _mm_unpackhi_epi32(t4, t6);
  __m128i tt6 = _mm_unpacklo_epi32(t5, t7);
  __m128i tt7 = _mm_unpackhi_epi32(t5, t7);

  // Transpose step 3: Unpack 64-bit elements (final interleave step)
  row0 = _mm_unpacklo_epi64(tt0, tt4);
  row1 = _mm_unpackhi_epi64(tt0, tt4);
  row2 = _mm_unpacklo_epi64(tt1, tt5);
  row3 = _mm_unpackhi_epi64(tt1, tt5);
  row4 = _mm_unpacklo_epi64(tt2, tt6);
  row5 = _mm_unpackhi_epi64(tt2, tt6);
  row6 = _mm_unpacklo_epi64(tt3, tt7);
  row7 = _mm_unpackhi_epi64(tt3, tt7);

  __m128i row0x = _mm_loadu_si128((__m128i*) & matrix_dst[0]);  // Row 0
  __m128i row1x = _mm_loadu_si128((__m128i*) & matrix_dst[width]);  // Row 1
  __m128i row2x = _mm_loadu_si128((__m128i*) & matrix_dst[width * 2]); // Row 2
  __m128i row3x = _mm_loadu_si128((__m128i*) & matrix_dst[width * 3]); // Row 3
  __m128i row4x = _mm_loadu_si128((__m128i*) & matrix_dst[width * 4]); // Row 4
  __m128i row5x = _mm_loadu_si128((__m128i*) & matrix_dst[width * 5]); // Row 5
  __m128i row6x = _mm_loadu_si128((__m128i*) & matrix_dst[width * 6]); // Row 6
  __m128i row7x = _mm_loadu_si128((__m128i*) & matrix_dst[width * 7]); // Row 7

  // Transpose step 1: Unpack 16-bit elements (interleave within pairs of rows)
  __m128i t0x = _mm_unpacklo_epi16(row0x, row1x);
  __m128i t1x = _mm_unpackhi_epi16(row0x, row1x);
  __m128i t2x = _mm_unpacklo_epi16(row2x, row3x);
  __m128i t3x = _mm_unpackhi_epi16(row2x, row3x);
  __m128i t4x = _mm_unpacklo_epi16(row4x, row5x);
  __m128i t5x = _mm_unpackhi_epi16(row4x, row5x);
  __m128i t6x = _mm_unpacklo_epi16(row6x, row7x);
  __m128i t7x = _mm_unpackhi_epi16(row6x, row7x);

  // Transpose step 2: Unpack 32-bit elements (interleave pairs of 16-bit results)
  __m128i tt0x = _mm_unpacklo_epi32(t0x, t2x);
  __m128i tt1x = _mm_unpackhi_epi32(t0x, t2x);
  __m128i tt2x = _mm_unpacklo_epi32(t1x, t3x);
  __m128i tt3x = _mm_unpackhi_epi32(t1x, t3x);
  __m128i tt4x = _mm_unpacklo_epi32(t4x, t6x);
  __m128i tt5x = _mm_unpackhi_epi32(t4x, t6x);
  __m128i tt6x = _mm_unpacklo_epi32(t5x, t7x);
  __m128i tt7x = _mm_unpackhi_epi32(t5x, t7x);

  // Transpose step 3: Unpack 64-bit elements (final interleave step)
  row0x = _mm_unpacklo_epi64(tt0x, tt4x);
  row1x = _mm_unpackhi_epi64(tt0x, tt4x);
  row2x = _mm_unpacklo_epi64(tt1x, tt5x);
  row3x = _mm_unpackhi_epi64(tt1x, tt5x);
  row4x = _mm_unpacklo_epi64(tt2x, tt6x);
  row5x = _mm_unpackhi_epi64(tt2x, tt6x);
  row6x = _mm_unpacklo_epi64(tt3x, tt7x);
  row7x = _mm_unpackhi_epi64(tt3x, tt7x);

  // Store the transposed rows back into the matrix
  _mm_storeu_si128((__m128i*) & matrix_dst[0], row0);
  _mm_storeu_si128((__m128i*) & matrix_dst[width], row1);
  _mm_storeu_si128((__m128i*) & matrix_dst[width*2], row2);
  _mm_storeu_si128((__m128i*) & matrix_dst[width*3], row3);
  _mm_storeu_si128((__m128i*) & matrix_dst[width*4], row4);
  _mm_storeu_si128((__m128i*) & matrix_dst[width*5], row5);
  _mm_storeu_si128((__m128i*) & matrix_dst[width*6], row6);
  _mm_storeu_si128((__m128i*) & matrix_dst[width*7], row7);

  _mm_storeu_si128((__m128i*) & matrix_src[0], row0x);
  _mm_storeu_si128((__m128i*) & matrix_src[width], row1x);
  _mm_storeu_si128((__m128i*) & matrix_src[width * 2], row2x);
  _mm_storeu_si128((__m128i*) & matrix_src[width * 3], row3x);
  _mm_storeu_si128((__m128i*) & matrix_src[width * 4], row4x);
  _mm_storeu_si128((__m128i*) & matrix_src[width * 5], row5x);
  _mm_storeu_si128((__m128i*) & matrix_src[width * 6], row6x);
  _mm_storeu_si128((__m128i*) & matrix_src[width * 7], row7x);
}

static void transpose8x8SSE2_outofplace(uint16_t* matrix_src, uint16_t* matrix_dst, size_t width, size_t height) {
  // Load rows of the matrix into SSE registers
  __m128i row0 = _mm_loadu_si128((__m128i*) & matrix_src[0]);  // Row 0
  __m128i row1 = _mm_loadu_si128((__m128i*) & matrix_src[width]);  // Row 1
  __m128i row2 = _mm_loadu_si128((__m128i*) & matrix_src[width * 2]); // Row 2
  __m128i row3 = _mm_loadu_si128((__m128i*) & matrix_src[width * 3]); // Row 3
  __m128i row4 = _mm_loadu_si128((__m128i*) & matrix_src[width * 4]); // Row 4
  __m128i row5 = _mm_loadu_si128((__m128i*) & matrix_src[width * 5]); // Row 5
  __m128i row6 = _mm_loadu_si128((__m128i*) & matrix_src[width * 6]); // Row 6
  __m128i row7 = _mm_loadu_si128((__m128i*) & matrix_src[width * 7]); // Row 7

  // Transpose step 1: Unpack 16-bit elements (interleave within pairs of rows)
  __m128i t0 = _mm_unpacklo_epi16(row0, row1);
  __m128i t1 = _mm_unpackhi_epi16(row0, row1);
  __m128i t2 = _mm_unpacklo_epi16(row2, row3);
  __m128i t3 = _mm_unpackhi_epi16(row2, row3);
  __m128i t4 = _mm_unpacklo_epi16(row4, row5);
  __m128i t5 = _mm_unpackhi_epi16(row4, row5);
  __m128i t6 = _mm_unpacklo_epi16(row6, row7);
  __m128i t7 = _mm_unpackhi_epi16(row6, row7);

  // Transpose step 2: Unpack 32-bit elements (interleave pairs of 16-bit results)
  __m128i tt0 = _mm_unpacklo_epi32(t0, t2);
  __m128i tt1 = _mm_unpackhi_epi32(t0, t2);
  __m128i tt2 = _mm_unpacklo_epi32(t1, t3);
  __m128i tt3 = _mm_unpackhi_epi32(t1, t3);
  __m128i tt4 = _mm_unpacklo_epi32(t4, t6);
  __m128i tt5 = _mm_unpackhi_epi32(t4, t6);
  __m128i tt6 = _mm_unpacklo_epi32(t5, t7);
  __m128i tt7 = _mm_unpackhi_epi32(t5, t7);

  // Transpose step 3: Unpack 64-bit elements (final interleave step)
  row0 = _mm_unpacklo_epi64(tt0, tt4);
  row1 = _mm_unpackhi_epi64(tt0, tt4);
  row2 = _mm_unpacklo_epi64(tt1, tt5);
  row3 = _mm_unpackhi_epi64(tt1, tt5);
  row4 = _mm_unpacklo_epi64(tt2, tt6);
  row5 = _mm_unpackhi_epi64(tt2, tt6);
  row6 = _mm_unpacklo_epi64(tt3, tt7);
  row7 = _mm_unpackhi_epi64(tt3, tt7);
    
  _mm_storeu_si128((__m128i*) & matrix_dst[0], row0);
  _mm_storeu_si128((__m128i*) & matrix_dst[height], row1);
  _mm_storeu_si128((__m128i*) & matrix_dst[height * 2], row2);
  _mm_storeu_si128((__m128i*) & matrix_dst[height * 3], row3);
  _mm_storeu_si128((__m128i*) & matrix_dst[height * 4], row4);
  _mm_storeu_si128((__m128i*) & matrix_dst[height * 5], row5);
  _mm_storeu_si128((__m128i*) & matrix_dst[height * 6], row6);
  _mm_storeu_si128((__m128i*) & matrix_dst[height * 7], row7);
}

static void transpose8x8SSE2_diagonal(uint16_t* matrix_src, size_t width) {
  // Load rows of the matrix into SSE registers
  __m128i row0 = _mm_loadu_si128((__m128i*) & matrix_src[0]);  // Row 0
  __m128i row1 = _mm_loadu_si128((__m128i*) & matrix_src[width]);  // Row 1
  __m128i row2 = _mm_loadu_si128((__m128i*) & matrix_src[width * 2]); // Row 2
  __m128i row3 = _mm_loadu_si128((__m128i*) & matrix_src[width * 3]); // Row 3
  __m128i row4 = _mm_loadu_si128((__m128i*) & matrix_src[width * 4]); // Row 4
  __m128i row5 = _mm_loadu_si128((__m128i*) & matrix_src[width * 5]); // Row 5
  __m128i row6 = _mm_loadu_si128((__m128i*) & matrix_src[width * 6]); // Row 6
  __m128i row7 = _mm_loadu_si128((__m128i*) & matrix_src[width * 7]); // Row 7

  // Transpose step 1: Unpack 16-bit elements (interleave within pairs of rows)
  __m128i t0 = _mm_unpacklo_epi16(row0, row1);
  __m128i t1 = _mm_unpackhi_epi16(row0, row1);
  __m128i t2 = _mm_unpacklo_epi16(row2, row3);
  __m128i t3 = _mm_unpackhi_epi16(row2, row3);
  __m128i t4 = _mm_unpacklo_epi16(row4, row5);
  __m128i t5 = _mm_unpackhi_epi16(row4, row5);
  __m128i t6 = _mm_unpacklo_epi16(row6, row7);
  __m128i t7 = _mm_unpackhi_epi16(row6, row7);

  // Transpose step 2: Unpack 32-bit elements (interleave pairs of 16-bit results)
  __m128i tt0 = _mm_unpacklo_epi32(t0, t2);
  __m128i tt1 = _mm_unpackhi_epi32(t0, t2);
  __m128i tt2 = _mm_unpacklo_epi32(t1, t3);
  __m128i tt3 = _mm_unpackhi_epi32(t1, t3);
  __m128i tt4 = _mm_unpacklo_epi32(t4, t6);
  __m128i tt5 = _mm_unpackhi_epi32(t4, t6);
  __m128i tt6 = _mm_unpacklo_epi32(t5, t7);
  __m128i tt7 = _mm_unpackhi_epi32(t5, t7);

  // Transpose step 3: Unpack 64-bit elements (final interleave step)
  row0 = _mm_unpacklo_epi64(tt0, tt4);
  row1 = _mm_unpackhi_epi64(tt0, tt4);
  row2 = _mm_unpacklo_epi64(tt1, tt5);
  row3 = _mm_unpackhi_epi64(tt1, tt5);
  row4 = _mm_unpacklo_epi64(tt2, tt6);
  row5 = _mm_unpackhi_epi64(tt2, tt6);
  row6 = _mm_unpacklo_epi64(tt3, tt7);
  row7 = _mm_unpackhi_epi64(tt3, tt7);

  _mm_storeu_si128((__m128i*) & matrix_src[0], row0);
  _mm_storeu_si128((__m128i*) & matrix_src[width], row1);
  _mm_storeu_si128((__m128i*) & matrix_src[width * 2], row2);
  _mm_storeu_si128((__m128i*) & matrix_src[width * 3], row3);
  _mm_storeu_si128((__m128i*) & matrix_src[width * 4], row4);
  _mm_storeu_si128((__m128i*) & matrix_src[width * 5], row5);
  _mm_storeu_si128((__m128i*) & matrix_src[width * 6], row6);
  _mm_storeu_si128((__m128i*) & matrix_src[width * 7], row7);
}

static const void* const MemoryMapFile_Input(const int fd, const size_t fileSize) {
  const void* const mappedFile = mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mappedFile == MAP_FAILED) {
    if (PRINT_ERRORS)
      write(1, "mmap\n", 5);
    _exit(1);
  }
  return mappedFile;
}

static void* const MemoryMapFile_Output(const int fd, const size_t fileSize) {

  if (ftruncate(fd, fileSize) == -1) {
    if (PRINT_ERRORS)
      write(1, "ftruncate\n", 10);
    _exit(1);
  }

  void* const mappedFile = mmap(NULL, fileSize, PROT_WRITE, MAP_SHARED, fd, 0);

  if (mappedFile == MAP_FAILED) {
    if (PRINT_ERRORS)
      write(1, "mmap\n", 5);
    _exit(1);
  }

  return mappedFile;
}

static void transpose_image(int fd_in, int fd_out) {

  struct HEADER {
    uint32_t width32;
    uint32_t height32;
  } header;

  // Read header (image dimensions)
  const size_t read_bytes = read(fd_in, &header, sizeof(header));
  if (read_bytes != sizeof(header)) {
    if (PRINT_ERRORS)
      write(1, "readsize\n", 9);
    _exit(1);
  }
  
  const size_t width = header.width32;
  const size_t height = header.height32;

  const size_t total_pixels = width * height;
  const size_t data_to_write = 2 * sizeof(uint32_t) + total_pixels * sizeof(uint16_t);


  if (width == height) {
    // square matrix
    // transpose in-place

    void* const output_malloc = sbrk(data_to_write + 2 * 4096llu);

    // output_pixels have to be aligned
    uint16_t* output_pixels = (uint16_t*)(((uintptr_t)output_malloc + 4096llu + 4095llu) & ~((uintptr_t)4095llu));
    uint32_t* const output_buffer = (uint32_t*)(output_pixels)-2llu;
    output_buffer[0] = header.height32;
    output_buffer[1] = header.width32;

    read(fd_in, output_pixels, sizeof(uint16_t)* total_pixels);

    if ((width & 7) == 0) {
      // square matrix, width and height are multiple of 8 

      for (size_t source_row = 0; source_row < height; source_row+=8) {
        size_t source_col = source_row;
        size_t source_index = source_row * width + source_col;
        size_t target_index = source_index;
        transpose8x8SSE2_diagonal(&output_pixels[source_index], width);
        target_index += width * 8;
        source_index += 8;
        source_col += 8;

        for (; source_col < width; source_col+=8) {
          transpose8x8SSE2_inplace(&output_pixels[source_index], &output_pixels[target_index], width);
          target_index += width*8;
          source_index+=8;
        }
      }

    }
    else {
      // square matrix, width and height are not multiple of 8

      size_t source_row = 0;
      for (; source_row + 8 <= height; source_row += 8) {
        size_t source_col = source_row;
        size_t source_index = source_row * width + source_col;
        size_t target_index = source_index;

        transpose8x8SSE2_diagonal(&output_pixels[source_index], width);
        source_col += 8;
        source_index += 8;
        target_index += width * 8;

        while (source_col + 8 <= width) {
          transpose8x8SSE2_inplace(&output_pixels[source_index], &output_pixels[target_index], width);
          source_col += 8;
          source_index += 8;
          target_index += width * 8;
        }

        // finish the last rectangle in this source row
        for (size_t row_idx = 0; row_idx < 8; row_idx++) {
          size_t source_col0 = source_col;
          size_t source_row0 = source_row + row_idx;
          source_index = source_row0 * width + source_col0;
          size_t target_row0 = source_col0;
          size_t target_col0 = source_row0;
          target_index = target_row0 * width + target_col0;
          while (source_col0 < width) {
            swap(&output_pixels[source_index], &output_pixels[target_index]);
            source_col0++;
            source_index++;
            target_index += width;
          }
        }

      }

      // finish the last diagonal square (bottom right)
      for (; source_row < height; source_row++) {
        size_t source_col = source_row + 1; // skip diagonal
        size_t source_index = source_row * width + source_col;
        size_t target_row = source_col;
        size_t target_col = source_row;
        size_t target_index = target_row * width + target_col;
        while (source_col < width) {
          swap(&output_pixels[source_index], &output_pixels[target_index]);
          source_index++;
          target_index += width;
          source_col++;
        }
      }

    }
    ssize_t bytes_written = write(fd_out, output_buffer, data_to_write);
    if (bytes_written < 0) {
      if (PRINT_ERRORS)
        write(1, "write\n", 6);
      _exit(1);
    }
    close(fd_out);
  }
  else {
    // generic matrix
    // transpose by copying

    void* const mappedFile_out = MemoryMapFile_Output(fd_out, data_to_write);

    uint32_t* const output_buffer = (uint32_t*)mappedFile_out;
    output_buffer[0] = header.height32;
    output_buffer[1] = header.width32;
    uint16_t* const output_pixels = (uint16_t*)(output_buffer + 2);

    const void* const mappedFile_in = MemoryMapFile_Input(fd_in, data_to_write);

    uint16_t* const input_pixels = (uint16_t*)(mappedFile_in) + 4;

    size_t source_row = 0;
    for (; source_row + 8 <= height; source_row += 8) {
      size_t source_col = 0;
      size_t source_index = source_row * width + source_col;
      size_t target_row = source_col;
      size_t target_col = source_row;
      size_t target_index = target_row * height /*target width*/ + target_col;
      while (source_col + 8 <= width) {
        transpose8x8SSE2_outofplace(&input_pixels[source_index], &output_pixels[target_index], width, height);
        source_col += 8;
        source_index += 8;
        target_index += height /*target width*/ * 8;
      }

      // finish the last rectangle in this source row

      for (size_t row_idx = 0; row_idx < 8; row_idx++) {
        size_t source_col0 = source_col;
        size_t source_row0 = source_row + row_idx;
        source_index = source_row0 * width + source_col0;
        size_t target_row0 = source_col0;
        size_t target_col0 = source_row0;
        target_index = target_row0 * height /*target width*/ + target_col0;
        while (source_col0 < width) {
          output_pixels[target_index] = input_pixels[source_index];
          source_col0++;
          source_index++;
          target_index += height /*target width*/;
        }
      }

    }

    // finish the last source rows
    for (; source_row < height; source_row++) {
      size_t source_col = 0;
      size_t source_index = source_row * width + source_col;
      size_t target_row = source_col;
      size_t target_col = source_row;
      size_t target_index = target_row * height /*target width*/ + target_col;
      while (source_col < width) {
        output_pixels[target_index] = input_pixels[source_index];
        source_index++;
        target_index += height /*target width*/;
        source_col++;
      }
    }

    if (munmap(mappedFile_out, data_to_write) == -1) {
      if (PRINT_ERRORS)
        write(1, "munmap\n", 7);
      _exit(1);
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    if (PRINT_ERRORS)
      write(1, "argc\n", 5);
    _exit(1);
  }

  const char* const input_file = argv[1];
  const char* const output_file = argv[2];

  // Open the file
  const int fd_in = open(input_file, O_RDONLY, 0);
  if (fd_in == -1) {
    if (PRINT_ERRORS)
      write(1, "fd_in\n", 6);
    _exit(1);
  }

  const int fd_out = open(output_file, O_RDWR | O_CREAT, 0644);
  if (fd_out == -1) {
    if (PRINT_ERRORS)
      write(1, "fd_out\n", 7);
    _exit(1);
  }

  transpose_image(fd_in, fd_out);

  _exit(0);
}
