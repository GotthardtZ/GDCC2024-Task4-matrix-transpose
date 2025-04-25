#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

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

  void* output_buffer = malloc(output_file_size);
  uint32_t* output_header = (uint32_t*)output_buffer;

  output_header[0] = height;
  output_header[1] = width;
  uint16_t* output_pixels = (uint16_t*)(output_header + 2);

  if (width == height) {
    //transpose in-place

    if (fread(output_pixels, sizeof(uint16_t), total_pixels, in_fp) != total_pixels) {
      perror("Error reading image");
      fclose(in_fp);
      exit(EXIT_FAILURE);
    }

    size_t input_column = 0;
    size_t target_index = 0;
    for (size_t input_index = 0; input_index < total_pixels; input_index++) {
      if (target_index < input_index) {
         uint16_t tmp1 = output_pixels[input_index];
         uint16_t tmp2 = output_pixels[target_index];
         output_pixels[input_index] = tmp2;
         output_pixels[target_index] =tmp1;
      }

      input_column++;
      target_index += height;
      if (input_column == width) {
        input_column = 0;
        target_index -= total_pixels - 1;
      }
    }
  }
  else {
    // transpose by copying

    uint16_t* input_pixels = (uint16_t*)malloc(total_pixels * sizeof(uint16_t));

    if (fread(input_pixels, sizeof(uint16_t), total_pixels, in_fp) != total_pixels) {
      perror("Error reading image");
      fclose(in_fp);
      exit(EXIT_FAILURE);
    }

    size_t input_column = 0;
    size_t target_index = 0;
    for (size_t input_index = 0; input_index < total_pixels; input_index++) {
      output_pixels[target_index] = input_pixels[input_index];
      input_column++;
      target_index += height;
      if (input_column == width) {
        input_column = 0;
        target_index -= total_pixels - 1;
      }
    }
  }

  if (fwrite(output_buffer, 1, output_file_size, out_fp) != output_file_size) {
    perror("Error writing output data");
    free(output_buffer);
    fclose(out_fp);
    exit(EXIT_FAILURE);
  }

  free(output_buffer);
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
