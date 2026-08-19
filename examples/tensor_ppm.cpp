#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <microllm/core/tensor.h>

namespace {

std::uint64_t fnv1a(const std::vector<std::uint8_t>& bytes) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const auto byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path output = argc > 1 ? argv[1] : "tensor.ppm";
        std::vector<float> pixels(64);
        for (std::size_t row = 0; row < 8; ++row) {
            for (std::size_t column = 0; column < 8; ++column) {
                pixels[row * 8 + column] = static_cast<float>(row * 32 + column * 4);
            }
        }

        const auto image = microllm::Tensor::from_vector(pixels, {8, 8});
        const auto transposed = image.transpose(0, 1);
        const auto values = transposed.to_vector();
        std::vector<std::uint8_t> bytes;
        bytes.reserve(values.size());
        for (const auto value : values) bytes.push_back(static_cast<std::uint8_t>(value));

        std::ofstream stream(output, std::ios::binary);
        if (!stream) throw std::runtime_error("cannot open output PPM");
        stream << "P5\n8 8\n255\n";
        stream.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
        stream.close();

        std::cout << "output=" << output << '\n';
        std::cout << "shape=8x8\n";
        std::cout << "checksum_fnv1a=" << fnv1a(bytes) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "microllm_tensor_ppm: " << error.what() << '\n';
        return 1;
    }
}
