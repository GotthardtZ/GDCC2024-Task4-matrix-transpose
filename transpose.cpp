#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <emmintrin.h>

static void transpose8x8SSE2_inplace(uint16_t* matrix_src, uint16_t* matrix_dst, size_t width) {
  // Load rows of the matrix into SSE registers
  __m128i row0 = _mm_load_si128((__m128i*) & matrix_src[0]);  // Row 0
  __m128i row1 = _mm_load_si128((__m128i*) & matrix_src[width]);  // Row 1
  __m128i row2 = _mm_load_si128((__m128i*) & matrix_src[width*2]); // Row 2
  __m128i row3 = _mm_load_si128((__m128i*) & matrix_src[width*3]); // Row 3
  __m128i row4 = _mm_load_si128((__m128i*) & matrix_src[width*4]); // Row 4
  __m128i row5 = _mm_load_si128((__m128i*) & matrix_src[width*5]); // Row 5
  __m128i row6 = _mm_load_si128((__m128i*) & matrix_src[width*6]); // Row 6
  __m128i row7 = _mm_load_si128((__m128i*) & matrix_src[width*7]); // Row 7

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

  __m128i row0x = _mm_load_si128((__m128i*) & matrix_dst[0]);  // Row 0
  __m128i row1x = _mm_load_si128((__m128i*) & matrix_dst[width]);  // Row 1
  __m128i row2x = _mm_load_si128((__m128i*) & matrix_dst[width * 2]); // Row 2
  __m128i row3x = _mm_load_si128((__m128i*) & matrix_dst[width * 3]); // Row 3
  __m128i row4x = _mm_load_si128((__m128i*) & matrix_dst[width * 4]); // Row 4
  __m128i row5x = _mm_load_si128((__m128i*) & matrix_dst[width * 5]); // Row 5
  __m128i row6x = _mm_load_si128((__m128i*) & matrix_dst[width * 6]); // Row 6
  __m128i row7x = _mm_load_si128((__m128i*) & matrix_dst[width * 7]); // Row 7

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
  _mm_store_si128((__m128i*) & matrix_dst[0], row0);
  _mm_store_si128((__m128i*) & matrix_dst[width], row1);
  _mm_store_si128((__m128i*) & matrix_dst[width*2], row2);
  _mm_store_si128((__m128i*) & matrix_dst[width*3], row3);
  _mm_store_si128((__m128i*) & matrix_dst[width*4], row4);
  _mm_store_si128((__m128i*) & matrix_dst[width*5], row5);
  _mm_store_si128((__m128i*) & matrix_dst[width*6], row6);
  _mm_store_si128((__m128i*) & matrix_dst[width*7], row7);

  _mm_store_si128((__m128i*) & matrix_src[0], row0x);
  _mm_store_si128((__m128i*) & matrix_src[width], row1x);
  _mm_store_si128((__m128i*) & matrix_src[width * 2], row2x);
  _mm_store_si128((__m128i*) & matrix_src[width * 3], row3x);
  _mm_store_si128((__m128i*) & matrix_src[width * 4], row4x);
  _mm_store_si128((__m128i*) & matrix_src[width * 5], row5x);
  _mm_store_si128((__m128i*) & matrix_src[width * 6], row6x);
  _mm_store_si128((__m128i*) & matrix_src[width * 7], row7x);
}

static void transpose8x8SSE2_diagonal(uint16_t* matrix_src, size_t width) {
  // Load rows of the matrix into SSE registers
  __m128i row0 = _mm_load_si128((__m128i*) & matrix_src[0]);  // Row 0
  __m128i row1 = _mm_load_si128((__m128i*) & matrix_src[width]);  // Row 1
  __m128i row2 = _mm_load_si128((__m128i*) & matrix_src[width * 2]); // Row 2
  __m128i row3 = _mm_load_si128((__m128i*) & matrix_src[width * 3]); // Row 3
  __m128i row4 = _mm_load_si128((__m128i*) & matrix_src[width * 4]); // Row 4
  __m128i row5 = _mm_load_si128((__m128i*) & matrix_src[width * 5]); // Row 5
  __m128i row6 = _mm_load_si128((__m128i*) & matrix_src[width * 6]); // Row 6
  __m128i row7 = _mm_load_si128((__m128i*) & matrix_src[width * 7]); // Row 7

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

  _mm_store_si128((__m128i*) & matrix_src[0], row0);
  _mm_store_si128((__m128i*) & matrix_src[width], row1);
  _mm_store_si128((__m128i*) & matrix_src[width * 2], row2);
  _mm_store_si128((__m128i*) & matrix_src[width * 3], row3);
  _mm_store_si128((__m128i*) & matrix_src[width * 4], row4);
  _mm_store_si128((__m128i*) & matrix_src[width * 5], row5);
  _mm_store_si128((__m128i*) & matrix_src[width * 6], row6);
  _mm_store_si128((__m128i*) & matrix_src[width * 7], row7);
}

void transpose_image(const char* input_file, const char* output_file) {
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

  // Read header (image dimensions)
  uint32_t width, height;
  if (fread(&width, sizeof(uint32_t), 1, in_fp) != 1 ||
    fread(&height, sizeof(uint32_t), 1, in_fp) != 1) {
    perror("Error reading image dimensions");
    fclose(in_fp);
    exit(EXIT_FAILURE);
  }

  const size_t total_pixels = width * height;

  constexpr size_t header_size = sizeof(uint32_t) + sizeof(uint32_t);
  size_t output_file_size = header_size + total_pixels * sizeof(uint16_t);

  // Allocate memory so that the pixel area (offet 8) is aligned to 16 bytes for SSE
  void* output_malloc = malloc(output_file_size + 15);
  uint8_t* output_buffer = (uint8_t*)output_malloc;
  while (((uintptr_t)output_buffer & (uintptr_t)15) != 8)
    output_buffer++;

  uint32_t* output_header = (uint32_t*)output_buffer;

  output_header[0] = height;
  output_header[1] = width;
  uint16_t* output_pixels = (uint16_t*)(output_header + 2);

  if (width == height) {
    // square matrix
    // transpose in-place

    if (fread(output_pixels, sizeof(uint16_t), total_pixels, in_fp) != total_pixels) {
      perror("Error reading image");
      fclose(in_fp);
      exit(EXIT_FAILURE);
    }

    if ((width & 7) == 0) {
      // square matrix, width and height are multiple of 8 

      for (size_t source_row = 0; source_row < height; source_row+=8) {
        size_t source_col = source_row;
        size_t source_index = source_row * width + source_col;
        size_t target_index = source_col * width + source_row;
        transpose8x8SSE2_diagonal(&output_pixels[source_index], width);
        target_index += height * 8;
        source_index += 8;
        source_col += 8;
        for (; source_col < width; source_col+=8) {
          transpose8x8SSE2_inplace(&output_pixels[source_index], &output_pixels[target_index], width);
          target_index += height*8;
          source_index+=8;
        }
      }
    }
    else {
      // square matrix, width and height are not multiple of 8

      for (size_t source_row = 0; source_row < height; source_row++) {
        size_t source_col = source_row + 1;
        size_t source_index = source_row * width + source_col;
        size_t target_index = source_col * width + source_row;
        for (; source_col < width; source_col++) {
          short tmp1 = output_pixels[source_index];
          short tmp2 = output_pixels[target_index];
          output_pixels[source_index] = tmp2;
          output_pixels[target_index] = tmp1;
          target_index += height;
          source_index++;
        }
      }
    }
  }
  else {
    // generic matrix
    // transpose by copying

    constexpr size_t BUFFER_SIZE = 64 * 1024;
    uint16_t* input_pixels = (uint16_t*)malloc(BUFFER_SIZE);

    size_t input_column = 0;
    size_t target_index = 0;
    size_t elements_count = BUFFER_SIZE / sizeof(uint16_t);
    size_t elements_read;
    while ((elements_read = fread(input_pixels, sizeof(uint16_t), elements_count, in_fp)) > 0) {
      for (size_t i = 0; i < elements_read; ++i) {
        output_pixels[target_index] = input_pixels[i];
        input_column++;
        target_index += height;
        if (input_column == width) {
          input_column = 0;
          target_index -= total_pixels - 1;
        }
      }
    }
  }

  if (fwrite(output_buffer, 1, output_file_size, out_fp) != output_file_size) {
    perror("Error writing output data");
    free(output_malloc);
    fclose(out_fp);
    exit(EXIT_FAILURE);
  }

  free(output_malloc);
  fclose(out_fp);
  fclose(in_fp);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
    return EXIT_FAILURE;
  }

  transpose_image(argv[1], argv[2]);
  return EXIT_SUCCESS;
}
