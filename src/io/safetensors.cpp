#include <microllm/io/safetensors.h>

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>

namespace microllm::io {
namespace {

struct Json;
using JsonObject = std::map<std::string, Json>;
using JsonArray = std::vector<Json>;

struct Json {
    std::variant<std::nullptr_t, bool, double, std::string, JsonArray, JsonObject> value;

    [[nodiscard]] const JsonObject& object(const char* context) const {
        const auto* result = std::get_if<JsonObject>(&value);
        if (result == nullptr) throw std::runtime_error(std::string(context) + " must be an object");
        return *result;
    }
    [[nodiscard]] const JsonArray& array(const char* context) const {
        const auto* result = std::get_if<JsonArray>(&value);
        if (result == nullptr) throw std::runtime_error(std::string(context) + " must be an array");
        return *result;
    }
    [[nodiscard]] const std::string& string(const char* context) const {
        const auto* result = std::get_if<std::string>(&value);
        if (result == nullptr) throw std::runtime_error(std::string(context) + " must be a string");
        return *result;
    }
    [[nodiscard]] std::int64_t integer(const char* context) const {
        const auto* number = std::get_if<double>(&value);
        if (number == nullptr || !std::isfinite(*number) || std::floor(*number) != *number ||
            *number < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
            *number > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
            throw std::runtime_error(std::string(context) + " must be an integer");
        }
        return static_cast<std::int64_t>(*number);
    }
};

class JsonParser {
public:
    explicit JsonParser(std::string_view text) : text_(text) {}

    [[nodiscard]] Json parse() {
        auto result = value();
        whitespace();
        if (position_ != text_.size()) error("trailing JSON data");
        return result;
    }

private:
    [[noreturn]] void error(const char* message) const {
        throw std::runtime_error(std::string("invalid JSON at byte ") +
                                 std::to_string(position_) + ": " + message);
    }

    void whitespace() {
        while (position_ < text_.size() &&
               (text_[position_] == ' ' || text_[position_] == '\n' ||
                text_[position_] == '\r' || text_[position_] == '\t')) {
            ++position_;
        }
    }

    bool consume(char expected) {
        whitespace();
        if (position_ < text_.size() && text_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    void require(char expected) {
        if (!consume(expected)) error("unexpected character");
    }

    [[nodiscard]] Json value() {
        whitespace();
        if (position_ >= text_.size()) error("missing value");
        switch (text_[position_]) {
            case '{': return Json{object()};
            case '[': return Json{array()};
            case '"': return Json{string()};
            case 't': literal("true"); return Json{true};
            case 'f': literal("false"); return Json{false};
            case 'n': literal("null"); return Json{nullptr};
            default: return Json{number()};
        }
    }

    [[nodiscard]] JsonObject object() {
        require('{');
        JsonObject result;
        if (consume('}')) return result;
        while (true) {
            whitespace();
            if (position_ >= text_.size() || text_[position_] != '"') {
                error("object key must be a string");
            }
            auto key = string();
            require(':');
            if (!result.emplace(std::move(key), value()).second) error("duplicate object key");
            if (consume('}')) return result;
            require(',');
        }
    }

    [[nodiscard]] JsonArray array() {
        require('[');
        JsonArray result;
        if (consume(']')) return result;
        while (true) {
            result.push_back(value());
            if (consume(']')) return result;
            require(',');
        }
    }

    static void append_utf8(std::string& output, std::uint32_t codepoint) {
        if (codepoint <= 0x7fU) {
            output.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7ffU) {
            output.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else {
            output.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
            output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        }
    }

    [[nodiscard]] std::uint32_t hex4() {
        if (position_ + 4 > text_.size()) error("short unicode escape");
        std::uint32_t result = 0;
        for (int index = 0; index < 4; ++index) {
            const auto character = text_[position_++];
            result <<= 4U;
            if (character >= '0' && character <= '9') result |= character - '0';
            else if (character >= 'a' && character <= 'f') result |= character - 'a' + 10U;
            else if (character >= 'A' && character <= 'F') result |= character - 'A' + 10U;
            else error("invalid unicode escape");
        }
        return result;
    }

    [[nodiscard]] std::string string() {
        require('"');
        std::string result;
        while (position_ < text_.size()) {
            const auto character = text_[position_++];
            if (character == '"') return result;
            if (static_cast<unsigned char>(character) < 0x20U) error("control character in string");
            if (character != '\\') {
                result.push_back(character);
                continue;
            }
            if (position_ >= text_.size()) error("unfinished escape");
            switch (text_[position_++]) {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                case 'u': {
                    const auto codepoint = hex4();
                    if (codepoint >= 0xd800U && codepoint <= 0xdfffU) {
                        error("unicode surrogate escapes are not supported in weight names");
                    }
                    append_utf8(result, codepoint);
                    break;
                }
                default: error("unknown string escape");
            }
        }
        error("unterminated string");
    }

    [[nodiscard]] double number() {
        whitespace();
        const auto start = position_;
        if (position_ < text_.size() && text_[position_] == '-') ++position_;
        if (position_ >= text_.size()) error("unfinished number");
        if (text_[position_] == '0') {
            ++position_;
        } else {
            if (text_[position_] < '1' || text_[position_] > '9') error("invalid number");
            while (position_ < text_.size() && text_[position_] >= '0' &&
                   text_[position_] <= '9') ++position_;
        }
        if (position_ < text_.size() && text_[position_] == '.') {
            ++position_;
            const auto fractional_start = position_;
            while (position_ < text_.size() && text_[position_] >= '0' &&
                   text_[position_] <= '9') ++position_;
            if (position_ == fractional_start) error("fraction requires digits");
        }
        if (position_ < text_.size() && (text_[position_] == 'e' || text_[position_] == 'E')) {
            ++position_;
            if (position_ < text_.size() && (text_[position_] == '+' || text_[position_] == '-')) {
                ++position_;
            }
            const auto exponent_start = position_;
            while (position_ < text_.size() && text_[position_] >= '0' &&
                   text_[position_] <= '9') ++position_;
            if (position_ == exponent_start) error("exponent requires digits");
        }
        const std::string token(text_.substr(start, position_ - start));
        std::size_t consumed = 0;
        const auto result = std::stod(token, &consumed);
        if (consumed != token.size()) error("invalid number token");
        return result;
    }

    void literal(std::string_view expected) {
        if (text_.substr(position_, expected.size()) != expected) error("invalid literal");
        position_ += expected.size();
    }

    std::string_view text_;
    std::size_t position_ = 0;
};

const Json& required(const JsonObject& object, const char* key, const char* context) {
    const auto found = object.find(key);
    if (found == object.end()) {
        throw std::runtime_error(std::string(context) + " is missing " + key);
    }
    return found->second;
}

std::uint64_t read_u64_le(const std::byte* bytes) {
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index) {
        value |= static_cast<std::uint64_t>(std::to_integer<unsigned char>(bytes[index]))
                 << (index * 8U);
    }
    return value;
}

void write_u64_le(std::ostream& stream, std::uint64_t value) {
    for (unsigned index = 0; index < 8; ++index) {
        stream.put(static_cast<char>((value >> (index * 8U)) & 0xffU));
    }
}

float half_to_float(std::uint16_t half) {
    const auto sign = static_cast<std::uint32_t>(half & 0x8000U) << 16U;
    const auto exponent = static_cast<std::uint32_t>((half >> 10U) & 0x1fU);
    auto mantissa = static_cast<std::uint32_t>(half & 0x03ffU);
    std::uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int normalized_exponent = -14;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1U;
                --normalized_exponent;
            }
            mantissa &= 0x03ffU;
            bits = sign |
                   (static_cast<std::uint32_t>(normalized_exponent + 127) << 23U) |
                   (mantissa << 13U);
        }
    } else if (exponent == 31U) {
        bits = sign | 0x7f800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    }
    return std::bit_cast<float>(bits);
}

std::uint16_t float_to_half(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto sign = static_cast<std::uint16_t>((bits >> 16U) & 0x8000U);
    const auto exponent = static_cast<int>((bits >> 23U) & 0xffU) - 127 + 15;
    auto mantissa = bits & 0x007fffffU;
    if (exponent <= 0) {
        if (exponent < -10) return sign;
        mantissa |= 0x00800000U;
        const auto shift = static_cast<unsigned>(14 - exponent);
        auto rounded = mantissa >> shift;
        const auto remainder = mantissa & ((1U << shift) - 1U);
        const auto halfway = 1U << (shift - 1U);
        if (remainder > halfway || (remainder == halfway && (rounded & 1U) != 0)) ++rounded;
        return static_cast<std::uint16_t>(sign | rounded);
    }
    if (exponent >= 31) {
        if ((bits & 0x7fffffffU) > 0x7f800000U) return static_cast<std::uint16_t>(sign | 0x7e00U);
        return static_cast<std::uint16_t>(sign | 0x7c00U);
    }
    auto rounded = mantissa + 0x00000fffU + ((mantissa >> 13U) & 1U);
    auto half_exponent = exponent;
    if ((rounded & 0x00800000U) != 0) {
        rounded = 0;
        ++half_exponent;
        if (half_exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
    }
    return static_cast<std::uint16_t>(sign | (static_cast<unsigned>(half_exponent) << 10U) |
                                      (rounded >> 13U));
}

std::uint16_t float_to_bfloat16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U) return 0x7fc0U;
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

float bfloat16_to_float(std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

std::string escape_json(std::string_view value) {
    std::string output;
    output.reserve(value.size() + 2);
    output.push_back('"');
    for (const auto character : value) {
        switch (character) {
            case '"': output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\b': output += "\\b"; break;
            case '\f': output += "\\f"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (static_cast<unsigned char>(character) < 0x20U) {
                    std::ostringstream escaped;
                    escaped << "\\u" << std::hex << std::setfill('0') << std::setw(4)
                            << static_cast<unsigned>(static_cast<unsigned char>(character));
                    output += escaped.str();
                } else {
                    output.push_back(character);
                }
        }
    }
    output.push_back('"');
    return output;
}

std::size_t file_element_bytes(const std::string& dtype) {
    if (dtype == "F32") return 4;
    if (dtype == "BF16" || dtype == "F16") return 2;
    throw std::runtime_error("unsupported safetensors dtype: " + dtype);
}

std::string file_dtype_name(WeightFileDType dtype) {
    switch (dtype) {
        case WeightFileDType::Float32: return "F32";
        case WeightFileDType::BFloat16: return "BF16";
        case WeightFileDType::Float16: return "F16";
    }
    throw std::logic_error("unknown weight file dtype");
}

std::size_t file_dtype_bytes(WeightFileDType dtype) {
    return dtype == WeightFileDType::Float32 ? 4U : 2U;
}

std::vector<std::byte> encode_values(const Tensor& tensor, WeightFileDType dtype) {
    if (!tensor.defined() || tensor.dtype() != DType::Float32) {
        throw std::invalid_argument("safetensors save requires defined float32 tensors");
    }
    const auto values = tensor.to_vector();
    std::vector<std::byte> output(values.size() * file_dtype_bytes(dtype));
    if (dtype == WeightFileDType::Float32) {
        std::memcpy(output.data(), values.data(), output.size());
        return output;
    }
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto converted = dtype == WeightFileDType::BFloat16
                                   ? float_to_bfloat16(values[index])
                                   : float_to_half(values[index]);
        output[index * 2] = static_cast<std::byte>(converted & 0xffU);
        output[index * 2 + 1] = static_cast<std::byte>((converted >> 8U) & 0xffU);
    }
    return output;
}

std::vector<float> decode_values(const std::vector<std::byte>& bytes,
                                 const std::string& dtype) {
    const auto element_bytes = file_element_bytes(dtype);
    if (bytes.size() % element_bytes != 0) throw std::runtime_error("misaligned tensor bytes");
    std::vector<float> output(bytes.size() / element_bytes);
    if (dtype == "F32") {
        std::memcpy(output.data(), bytes.data(), bytes.size());
        return output;
    }
    for (std::size_t index = 0; index < output.size(); ++index) {
        const auto low = std::to_integer<std::uint16_t>(bytes[index * 2]);
        const auto high = std::to_integer<std::uint16_t>(bytes[index * 2 + 1]);
        const auto value = static_cast<std::uint16_t>(low | (high << 8U));
        output[index] = dtype == "BF16" ? bfloat16_to_float(value) : half_to_float(value);
    }
    return output;
}

std::uint64_t checked_tensor_bytes(const Shape& shape, std::size_t element_bytes) {
    const auto elements = checked_numel(shape);
    if (static_cast<std::uint64_t>(elements) >
        std::numeric_limits<std::uint64_t>::max() / element_bytes) {
        throw std::overflow_error("weight tensor byte size overflows");
    }
    return static_cast<std::uint64_t>(elements) * element_bytes;
}

struct TensorDescriptor {
    std::string name;
    std::string dtype;
    Shape shape;
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
};

std::vector<TensorDescriptor> descriptors(const JsonObject& root) {
    std::vector<TensorDescriptor> output;
    for (const auto& [name, json] : root) {
        if (name == "__metadata__") continue;
        if (name.empty()) throw std::runtime_error("safetensors tensor name cannot be empty");
        const auto& object = json.object("tensor descriptor");
        TensorDescriptor descriptor;
        descriptor.name = name;
        descriptor.dtype = required(object, "dtype", "tensor descriptor").string("dtype");
        for (const auto& dimension : required(object, "shape", "tensor descriptor").array("shape")) {
            descriptor.shape.push_back(dimension.integer("shape dimension"));
        }
        const auto& offsets = required(object, "data_offsets", "tensor descriptor").array("data_offsets");
        if (offsets.size() != 2) throw std::runtime_error("data_offsets must have two entries");
        const auto begin = offsets[0].integer("data offset");
        const auto end = offsets[1].integer("data offset");
        if (begin < 0 || end < begin) throw std::runtime_error("invalid tensor data offsets");
        descriptor.begin = static_cast<std::uint64_t>(begin);
        descriptor.end = static_cast<std::uint64_t>(end);
        const auto expected = checked_tensor_bytes(descriptor.shape,
                                                   file_element_bytes(descriptor.dtype));
        if (descriptor.end - descriptor.begin != expected) {
            throw std::runtime_error("tensor byte range does not match dtype and shape: " + name);
        }
        output.push_back(std::move(descriptor));
    }
    auto ordered = output;
    std::sort(ordered.begin(), ordered.end(), [](const auto& left, const auto& right) {
        return std::tie(left.begin, left.end, left.name) <
               std::tie(right.begin, right.end, right.name);
    });
    std::uint64_t expected_begin = 0;
    for (const auto& descriptor : ordered) {
        if (descriptor.begin != expected_begin) {
            throw std::runtime_error("safetensors data ranges overlap or leave a hole");
        }
        expected_begin = descriptor.end;
    }
    return output;
}

DType tensor_dtype(const std::string& dtype) {
    if (dtype == "F32") return DType::Float32;
    if (dtype == "BF16") return DType::BFloat16;
    if (dtype == "F16") return DType::Float16;
    throw std::runtime_error("unsupported safetensors dtype: " + dtype);
}

struct ParsedSafetensorsFile {
    std::uint64_t file_size = 0;
    std::uint64_t data_start = 0;
    std::vector<TensorDescriptor> tensors;
};

ParsedSafetensorsFile parse_file_header(std::ifstream& input,
                                        const std::filesystem::path& path) {
    input.seekg(0, std::ios::end);
    const auto end_position = input.tellg();
    if (end_position < std::streamoff{8}) {
        throw std::runtime_error("safetensors file is too short");
    }
    ParsedSafetensorsFile result;
    result.file_size = static_cast<std::uint64_t>(end_position);
    input.seekg(0);
    std::byte prefix[8]{};
    input.read(reinterpret_cast<char*>(prefix), 8);
    if (!input) throw std::runtime_error("cannot read safetensors header length");
    const auto header_bytes = read_u64_le(prefix);
    constexpr std::uint64_t kMaximumHeaderBytes = 256U * 1024U * 1024U;
    if (header_bytes == 0 || header_bytes > kMaximumHeaderBytes ||
        header_bytes > result.file_size - 8U) {
        throw std::runtime_error("invalid safetensors header length");
    }
    std::string header(static_cast<std::size_t>(header_bytes), '\0');
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (!input) throw std::runtime_error("cannot read safetensors header");
    result.tensors = descriptors(JsonParser(header).parse().object("safetensors header"));
    if (result.tensors.empty()) throw std::runtime_error("safetensors file contains no tensors");
    result.data_start = 8U + header_bytes;
    const auto payload_bytes = result.file_size - result.data_start;
    const auto last_end = std::max_element(
        result.tensors.begin(), result.tensors.end(),
        [](const auto& left, const auto& right) { return left.end < right.end; })->end;
    if (last_end != payload_bytes) {
        throw std::runtime_error("safetensors payload size does not match tensor ranges: " +
                                 path.string());
    }
    return result;
}

std::string build_header(const StateDict& state, WeightFileDType dtype) {
    std::ostringstream header;
    header << '{';
    std::uint64_t offset = 0;
    bool first = true;
    const auto dtype_name = file_dtype_name(dtype);
    for (const auto& [name, tensor] : state) {
        if (name.empty()) throw std::invalid_argument("weight name cannot be empty");
        if (!first) header << ',';
        first = false;
        const auto bytes = checked_tensor_bytes(tensor.shape(), file_dtype_bytes(dtype));
        header << escape_json(name) << ":{\"dtype\":\"" << dtype_name << "\",\"shape\":[";
        for (std::size_t index = 0; index < tensor.shape().size(); ++index) {
            if (index != 0) header << ',';
            header << tensor.shape()[index];
        }
        header << "],\"data_offsets\":[" << offset << ',' << offset + bytes << "]}";
        offset += bytes;
    }
    if (!first) header << ',';
    header << "\"__metadata__\":{\"format\":\"pt\",\"producer\":\"microLLM-rocm\"}}";
    return header.str();
}

}  // namespace

std::vector<SafetensorsTensorInfo> inspect_safetensors(
    const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open safetensors file: " + path.string());
    const auto parsed = parse_file_header(input, path);
    std::vector<SafetensorsTensorInfo> output;
    output.reserve(parsed.tensors.size());
    for (const auto& tensor : parsed.tensors) {
        output.push_back({tensor.name, tensor_dtype(tensor.dtype), tensor.shape,
                          tensor.end - tensor.begin});
    }
    return output;
}

void visit_safetensors(const std::filesystem::path& path,
                       const SafetensorsTensorVisitor& visitor) {
    if (!visitor) throw std::invalid_argument("safetensors visitor must be callable");
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open safetensors file: " + path.string());
    auto parsed = parse_file_header(input, path);
    std::sort(parsed.tensors.begin(), parsed.tensors.end(),
              [](const auto& left, const auto& right) { return left.begin < right.begin; });
    const auto maximum = std::max_element(
        parsed.tensors.begin(), parsed.tensors.end(), [](const auto& left, const auto& right) {
            return left.end - left.begin < right.end - right.begin;
        });
    std::vector<std::byte> buffer(static_cast<std::size_t>(maximum->end - maximum->begin));
    for (const auto& tensor : parsed.tensors) {
        const auto byte_count = static_cast<std::size_t>(tensor.end - tensor.begin);
        input.seekg(static_cast<std::streamoff>(parsed.data_start + tensor.begin));
        if (byte_count != 0) {
            input.read(reinterpret_cast<char*>(buffer.data()),
                       static_cast<std::streamsize>(byte_count));
        }
        if (!input) throw std::runtime_error("cannot read tensor data: " + tensor.name);
        visitor({tensor.name, tensor_dtype(tensor.dtype), tensor.shape, byte_count},
                std::span<const std::byte>(buffer.data(), byte_count));
    }
}

StateDict load_safetensors(const std::filesystem::path& path, Device target) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open safetensors file: " + path.string());
    const auto file = parse_file_header(input, path);

    StateDict state;
    for (const auto& descriptor : file.tensors) {
        if (descriptor.end > file.file_size - file.data_start) {
            throw std::runtime_error("tensor data range exceeds safetensors file: " +
                                     descriptor.name);
        }
        const auto byte_count = descriptor.end - descriptor.begin;
        std::vector<std::byte> bytes(static_cast<std::size_t>(byte_count));
        input.seekg(static_cast<std::streamoff>(file.data_start + descriptor.begin));
        if (!bytes.empty()) {
            input.read(reinterpret_cast<char*>(bytes.data()),
                       static_cast<std::streamsize>(bytes.size()));
        }
        if (!input) throw std::runtime_error("cannot read tensor data: " + descriptor.name);
        auto tensor = Tensor::from_vector(decode_values(bytes, descriptor.dtype), descriptor.shape);
        if (target != Device::cpu()) tensor = tensor.to(target);
        state.emplace(descriptor.name, std::move(tensor));
    }
    return state;
}

StateDict load_safetensors_files(const std::vector<std::filesystem::path>& paths,
                                 Device target) {
    if (paths.empty()) throw std::invalid_argument("at least one safetensors file is required");
    StateDict combined;
    for (const auto& path : paths) {
        auto shard = load_safetensors(path, target);
        for (auto& [name, tensor] : shard) {
            if (!combined.emplace(name, std::move(tensor)).second) {
                throw std::runtime_error("duplicate weight across safetensors files: " + name);
            }
        }
    }
    return combined;
}

SafetensorsIndex inspect_safetensors_index(
    const std::filesystem::path& index_path) {
    std::ifstream input(index_path);
    if (!input) throw std::runtime_error("cannot open safetensors index: " + index_path.string());
    std::ostringstream contents;
    contents << input.rdbuf();
    const auto root = JsonParser(contents.str()).parse().object("safetensors index");
    const auto& weight_map = required(root, "weight_map", "safetensors index")
                                 .object("weight_map");
    if (weight_map.empty()) throw std::runtime_error("safetensors weight_map is empty");

    SafetensorsIndex output;
    for (const auto& [name, file_json] : weight_map) {
        const auto& filename = file_json.string("weight shard filename");
        const std::filesystem::path relative(filename);
        if (relative.empty() || relative.is_absolute()) {
            throw std::runtime_error("weight shard filename must be relative");
        }
        for (const auto& component : relative) {
            if (component == "..") throw std::runtime_error("weight shard escapes index directory");
        }
        output.emplace(name, (index_path.parent_path() / relative).lexically_normal());
    }
    return output;
}

StateDict load_safetensors_index(const std::filesystem::path& index_path, Device target) {
    const auto weight_map = inspect_safetensors_index(index_path);
    std::map<std::filesystem::path, std::set<std::string>> names_by_file;
    for (const auto& [name, path] : weight_map) names_by_file[path].insert(name);

    StateDict output;
    for (const auto& [path, expected_names] : names_by_file) {
        const auto shard = load_safetensors(path, target);
        for (const auto& name : expected_names) {
            const auto found = shard.find(name);
            if (found == shard.end()) {
                throw std::runtime_error("indexed weight is absent from shard: " + name);
            }
            output.emplace(name, found->second);
        }
    }
    return output;
}

void save_safetensors(const std::filesystem::path& path, const StateDict& state,
                      const SafetensorsSaveOptions& options) {
    if (state.empty()) throw std::invalid_argument("cannot save an empty state dict");
    auto header = build_header(state, options.dtype);
    header.append((8U - header.size() % 8U) % 8U, ' ');
    const auto output_path = options.atomic_replace
                                 ? std::filesystem::path(path.string() + ".tmp")
                                 : path;
    try {
        std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot open safetensors output: " + output_path.string());
        write_u64_le(output, static_cast<std::uint64_t>(header.size()));
        output.write(header.data(), static_cast<std::streamsize>(header.size()));
        for (const auto& [name, tensor] : state) {
            (void)name;
            const auto bytes = encode_values(tensor, options.dtype);
            if (!bytes.empty()) {
                output.write(reinterpret_cast<const char*>(bytes.data()),
                             static_cast<std::streamsize>(bytes.size()));
            }
        }
        output.flush();
        if (!output) throw std::runtime_error("failed while writing safetensors output");
        output.close();
        if (options.atomic_replace) std::filesystem::rename(output_path, path);
    } catch (...) {
        if (options.atomic_replace) {
            std::error_code ignored;
            std::filesystem::remove(output_path, ignored);
        }
        throw;
    }
}

}  // namespace microllm::io
