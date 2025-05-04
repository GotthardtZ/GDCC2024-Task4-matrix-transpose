#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <emmintrin.h>

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

#ifdef _WIN32

#include <windows.h>

static void* MemoryMapFile(const char* fileName, size_t* fileSize) {
  // Open the file
  HANDLE hFile = CreateFileA(
    fileName,
    GENERIC_READ,
    FILE_SHARE_READ,
    NULL,
    OPEN_EXISTING,
    FILE_ATTRIBUTE_NORMAL,
    NULL
  );

  if (hFile == INVALID_HANDLE_VALUE) {
    fprintf(stderr, "Error opening file: %lu\n", GetLastError());
    return NULL;
  }

  // Get the file size
  LARGE_INTEGER fileSizeStruct;
  if (!GetFileSizeEx(hFile, &fileSizeStruct)) {
    fprintf(stderr, "Error getting file size: %lu\n", GetLastError());
    CloseHandle(hFile);
    return NULL;
  }

  *fileSize = (size_t)fileSizeStruct.QuadPart;

  // Create a file mapping object
  HANDLE hMapping = CreateFileMappingA(
    hFile,
    NULL,
    PAGE_READONLY,
    0,
    0,
    NULL
  );

  if (hMapping == NULL) {
    fprintf(stderr, "Error creating file mapping: %lu\n", GetLastError());
    CloseHandle(hFile);
    return NULL;
  }

  // Map the file into memory
  void* mappedFile = MapViewOfFile(
    hMapping,
    FILE_MAP_READ,
    0,
    0,
    0
  );

  if (mappedFile == NULL) {
    fprintf(stderr, "Error mapping view of file: %lu\n", GetLastError());
    CloseHandle(hMapping);
    CloseHandle(hFile);
    return NULL;
  }

  // Close the file and mapping handles, leaving the mapped file active
  CloseHandle(hMapping);
  CloseHandle(hFile);

  return mappedFile;
}

static void UnmapFile(void* mappedFile, size_t fileSize) {
  if (mappedFile) {
    UnmapViewOfFile(mappedFile);
  }
}

#elif __linux__

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>

static void* MemoryMapFile(const char* fileName, size_t* fileSize) {
  // Open the file
  int fd = open(fileName, O_RDONLY);
  if (fd == -1) {
    perror("Error opening file");
    return NULL;
  }

  // Get the file size
  struct stat fileStat;
  if (fstat(fd, &fileStat) == -1) {
    perror("Error getting file size");
    close(fd);
    return NULL;
  }

  *fileSize = (size_t)fileStat.st_size;

  // Map the file into memory
  void* mappedFile = mmap(NULL, *fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mappedFile == MAP_FAILED) {
    perror("Error mapping file");
    close(fd);
    return NULL;
  }

  // Close the file descriptor, leaving the mapping active
  close(fd);

  return mappedFile;
}

static void UnmapFile(void* mappedFile, size_t fileSize) {
  if (mappedFile) {
    if (munmap(mappedFile, fileSize) == -1) {
      perror("Error unmapping file");
    }
  }
}

#endif

static void transpose_image(FILE* in_fp, FILE* out_fp, char* input_file) {

  // Read header (image dimensions)
  uint32_t width32, height32;
  if (fread(&width32, sizeof(uint32_t), 1, in_fp) != 1 ||
    fread(&height32, sizeof(uint32_t), 1, in_fp) != 1) {
    perror("Error reading image dimensions");
    fclose(in_fp);
    exit(EXIT_FAILURE);
  }

  // Write header (image dimensions)
  fwrite(&height32, sizeof(uint32_t), 1, out_fp);
  fwrite(&width32, sizeof(uint32_t), 1, out_fp);

  const size_t width = width32;
  const size_t height = height32;

  const size_t total_pixels = width * height;
  const void* output_malloc = malloc(total_pixels * sizeof(uint16_t) + 31);
  uint16_t* output_pixels = (uint16_t*)(((uintptr_t)output_malloc + 31) & ~((uintptr_t)31));

  if (width == height) {
    // square matrix
    // transpose in-place

    fread(output_pixels, sizeof(uint16_t), total_pixels, in_fp);

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

      fwrite(output_pixels, sizeof(uint16_t), total_pixels, out_fp);
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

      fwrite(output_pixels, sizeof(uint16_t), total_pixels, out_fp);
    }
  }
  else {
    // generic matrix
    // transpose by copying

    size_t fileSize = 0;
    void* mappedFile_in = MemoryMapFile(input_file, &fileSize);

    if (!mappedFile_in) {
      perror("Error opening input file");
      exit(EXIT_FAILURE);
    }

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

    UnmapFile(mappedFile_in, fileSize);
    fwrite(output_pixels, sizeof(uint16_t), total_pixels, out_fp);
  }
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
    return EXIT_FAILURE;
  }

  char* input_file = argv[1];
  char* output_file = argv[2];

  FILE* in_fp = fopen(input_file, "rb");
  if (!in_fp) {
    perror("Error opening input file");
    exit(EXIT_FAILURE);
  }

  FILE* out_fp = fopen(output_file, "wb");
  if (!out_fp) {
    perror("Error opening output file");
    fclose(in_fp);
    exit(EXIT_FAILURE);
  }
  
  transpose_image(in_fp, out_fp, input_file);

  fclose(in_fp);
  fclose(out_fp);

  return EXIT_SUCCESS;
}
