#include <microllm/training/checkpoint.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace microllm::training {
namespace {

constexpr std::string_view kMagic = "MLLMCKPT";
constexpr std::uint32_t kEndianMarker = 0x01020304U;
constexpr std::uint64_t kMaxCollectionItems = 1'000'000;

class Writer {
public:
    template <typename Integer>
    void integer(Integer value) {
        static_assert(std::is_integral_v<Integer>);
        using Unsigned = std::make_unsigned_t<Integer>;
        auto bits = static_cast<Unsigned>(value);
        for (std::size_t byte = 0; byte < sizeof(Integer); ++byte) {
            bytes_.push_back(static_cast<std::byte>((bits >> (byte * 8U)) & 0xffU));
        }
    }

    void floating(float value) { integer(std::bit_cast<std::uint32_t>(value)); }

    void string(std::string_view value) {
        integer<std::uint64_t>(value.size());
        for (const auto character : value) {
            bytes_.push_back(static_cast<std::byte>(static_cast<unsigned char>(character)));
        }
    }

    void tensor(const Tensor& value) {
        if (!value.device().is_cpu() || value.dtype() != DType::Float32 ||
            !value.is_contiguous()) {
            throw std::invalid_argument("checkpoint tensors must be contiguous CPU float32");
        }
        integer<std::uint64_t>(value.shape().size());
        for (const auto dimension : value.shape()) integer<std::int64_t>(dimension);
        const auto elements = value.to_vector();
        integer<std::uint64_t>(elements.size());
        for (const auto element : elements) floating(element);
    }

    [[nodiscard]] const std::vector<std::byte>& bytes() const noexcept { return bytes_; }

private:
    std::vector<std::byte> bytes_;
};

class Reader {
public:
    explicit Reader(const std::vector<std::byte>& bytes) : bytes_(bytes) {}

    template <typename Integer>
    Integer integer() {
        static_assert(std::is_integral_v<Integer>);
        require(sizeof(Integer));
        using Unsigned = std::make_unsigned_t<Integer>;
        Unsigned bits = 0;
        for (std::size_t byte = 0; byte < sizeof(Integer); ++byte) {
            bits |= static_cast<Unsigned>(std::to_integer<unsigned char>(bytes_[position_++]))
                    << (byte * 8U);
        }
        return static_cast<Integer>(bits);
    }

    float floating() { return std::bit_cast<float>(integer<std::uint32_t>()); }

    std::string string() {
        const auto size = integer<std::uint64_t>();
        if (size > remaining()) throw std::runtime_error("checkpoint string exceeds payload");
        std::string result;
        result.reserve(static_cast<std::size_t>(size));
        for (std::uint64_t index = 0; index < size; ++index) {
            result.push_back(static_cast<char>(std::to_integer<unsigned char>(bytes_[position_++])));
        }
        return result;
    }

    Tensor tensor() {
        const auto rank = integer<std::uint64_t>();
        if (rank > 64) throw std::runtime_error("checkpoint tensor rank is unreasonable");
        Shape shape;
        shape.reserve(static_cast<std::size_t>(rank));
        for (std::uint64_t index = 0; index < rank; ++index) {
            shape.push_back(integer<std::int64_t>());
        }
        const auto elements = integer<std::uint64_t>();
        if (elements != static_cast<std::uint64_t>(checked_numel(shape))) {
            throw std::runtime_error("checkpoint tensor element count mismatch");
        }
        if (elements > remaining() / sizeof(float)) {
            throw std::runtime_error("checkpoint tensor exceeds payload");
        }
        std::vector<float> values(static_cast<std::size_t>(elements));
        for (auto& value : values) value = floating();
        return Tensor::from_vector(values, std::move(shape));
    }

    [[nodiscard]] std::size_t remaining() const noexcept { return bytes_.size() - position_; }
    [[nodiscard]] bool finished() const noexcept { return position_ == bytes_.size(); }

private:
    void require(std::size_t count) const {
        if (count > remaining()) throw std::runtime_error("checkpoint payload is truncated");
    }

    const std::vector<std::byte>& bytes_;
    std::size_t position_ = 0;
};

std::uint64_t payload_checksum(const std::vector<std::byte>& payload) {
    std::uint64_t value = 14695981039346656037ULL;
    for (const auto byte : payload) {
        value ^= std::to_integer<unsigned char>(byte);
        value *= 1099511628211ULL;
    }
    return value;
}

void write_config(Writer& writer, const AdamWConfig& config) {
    writer.floating(config.learning_rate);
    writer.floating(config.beta1);
    writer.floating(config.beta2);
    writer.floating(config.epsilon);
    writer.floating(config.weight_decay);
}

AdamWConfig read_config(Reader& reader) {
    return {.learning_rate = reader.floating(),
            .beta1 = reader.floating(),
            .beta2 = reader.floating(),
            .epsilon = reader.floating(),
            .weight_decay = reader.floating()};
}

void write_state(Writer& writer, const AdamWState& state) {
    writer.integer<std::uint64_t>(state.step);
    writer.integer<std::uint64_t>(state.first_moments.size());
    for (const auto& moment : state.first_moments) writer.tensor(moment);
    writer.integer<std::uint64_t>(state.second_moments.size());
    for (const auto& moment : state.second_moments) writer.tensor(moment);
}

AdamWState read_state(Reader& reader) {
    AdamWState state;
    state.step = reader.integer<std::uint64_t>();
    const auto first_count = reader.integer<std::uint64_t>();
    if (first_count > kMaxCollectionItems) throw std::runtime_error("too many first moments");
    state.first_moments.reserve(static_cast<std::size_t>(first_count));
    for (std::uint64_t index = 0; index < first_count; ++index) {
        state.first_moments.push_back(reader.tensor());
    }
    const auto second_count = reader.integer<std::uint64_t>();
    if (second_count > kMaxCollectionItems) throw std::runtime_error("too many second moments");
    state.second_moments.reserve(static_cast<std::size_t>(second_count));
    for (std::uint64_t index = 0; index < second_count; ++index) {
        state.second_moments.push_back(reader.tensor());
    }
    return state;
}

void validate_named_parameters(const NamedParameters& parameters) {
    std::unordered_set<std::string> names;
    for (const auto& [name, parameter] : parameters) {
        if (name.empty() || !names.insert(name).second) {
            throw std::invalid_argument("checkpoint parameter names must be unique and non-empty");
        }
        if (parameter == nullptr || !parameter->defined()) {
            throw std::invalid_argument("checkpoint parameter must be defined");
        }
    }
}

bool same_config(const AdamWConfig& left, const AdamWConfig& right) {
    return left.learning_rate == right.learning_rate && left.beta1 == right.beta1 &&
           left.beta2 == right.beta2 && left.epsilon == right.epsilon &&
           left.weight_decay == right.weight_decay;
}

}  // namespace

void save_checkpoint(const std::filesystem::path& path, const NamedParameters& parameters,
                     const AdamW& optimizer, const ExperimentState& experiment) {
    validate_named_parameters(parameters);
    Writer payload;
    payload.integer<std::uint64_t>(experiment.global_step);
    payload.integer<std::uint64_t>(experiment.data_cursor);
    payload.string(experiment.rng_state);
    payload.string(experiment.model_config);
    payload.string(experiment.data_config);
    payload.integer<std::uint64_t>(parameters.size());
    for (const auto& [name, parameter] : parameters) {
        payload.string(name);
        payload.tensor(parameter->data());
    }
    write_config(payload, optimizer.config());
    write_state(payload, optimizer.state());

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open checkpoint for writing");
    output.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    Writer header;
    header.integer<std::uint32_t>(kCheckpointFormatVersion);
    header.integer<std::uint32_t>(kEndianMarker);
    header.integer<std::uint64_t>(payload.bytes().size());
    header.integer<std::uint64_t>(payload_checksum(payload.bytes()));
    output.write(reinterpret_cast<const char*>(header.bytes().data()),
                 static_cast<std::streamsize>(header.bytes().size()));
    output.write(reinterpret_cast<const char*>(payload.bytes().data()),
                 static_cast<std::streamsize>(payload.bytes().size()));
    output.flush();
    if (!output) throw std::runtime_error("checkpoint write failed");
}

LoadedCheckpoint load_checkpoint(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open checkpoint for reading");
    const auto file_size = input.tellg();
    if (file_size < 0) throw std::runtime_error("cannot determine checkpoint size");
    input.seekg(0);
    std::vector<std::byte> file(static_cast<std::size_t>(file_size));
    input.read(reinterpret_cast<char*>(file.data()), static_cast<std::streamsize>(file.size()));
    if (!input) throw std::runtime_error("checkpoint read failed");
    if (file.size() < kMagic.size() + 24) throw std::runtime_error("checkpoint is too small");
    for (std::size_t index = 0; index < kMagic.size(); ++index) {
        if (std::to_integer<char>(file[index]) != kMagic[index]) {
            throw std::runtime_error("checkpoint magic mismatch");
        }
    }
    std::vector<std::byte> header_bytes(file.begin() + static_cast<std::ptrdiff_t>(kMagic.size()),
                                        file.begin() + static_cast<std::ptrdiff_t>(kMagic.size() + 24));
    Reader header(header_bytes);
    const auto version = header.integer<std::uint32_t>();
    if (version != kCheckpointFormatVersion) throw std::runtime_error("unsupported checkpoint version");
    if (header.integer<std::uint32_t>() != kEndianMarker) {
        throw std::runtime_error("checkpoint endian marker mismatch");
    }
    const auto payload_size = header.integer<std::uint64_t>();
    const auto expected_checksum = header.integer<std::uint64_t>();
    if (payload_size != file.size() - kMagic.size() - 24) {
        throw std::runtime_error("checkpoint payload size mismatch");
    }
    std::vector<std::byte> payload(file.begin() + static_cast<std::ptrdiff_t>(kMagic.size() + 24),
                                   file.end());
    if (payload_checksum(payload) != expected_checksum) {
        throw std::runtime_error("checkpoint payload integrity check failed");
    }

    Reader reader(payload);
    LoadedCheckpoint result;
    result.format_version = version;
    result.experiment.global_step = reader.integer<std::uint64_t>();
    result.experiment.data_cursor = reader.integer<std::uint64_t>();
    result.experiment.rng_state = reader.string();
    result.experiment.model_config = reader.string();
    result.experiment.data_config = reader.string();
    const auto parameter_count = reader.integer<std::uint64_t>();
    if (parameter_count > kMaxCollectionItems) throw std::runtime_error("too many parameters");
    result.parameters.reserve(static_cast<std::size_t>(parameter_count));
    for (std::uint64_t index = 0; index < parameter_count; ++index) {
        result.parameters.push_back({reader.string(), reader.tensor()});
    }
    result.optimizer_config = read_config(reader);
    result.optimizer_state = read_state(reader);
    if (!reader.finished()) throw std::runtime_error("checkpoint has trailing payload data");
    return result;
}

void restore_checkpoint(const LoadedCheckpoint& checkpoint, const NamedParameters& parameters,
                        AdamW& optimizer, ExperimentState& experiment) {
    validate_named_parameters(parameters);
    if (checkpoint.format_version != kCheckpointFormatVersion) {
        throw std::invalid_argument("checkpoint version cannot be restored");
    }
    if (checkpoint.parameters.size() != parameters.size()) {
        throw std::invalid_argument("checkpoint parameter count mismatch");
    }
    if (!same_config(checkpoint.optimizer_config, optimizer.config())) {
        throw std::invalid_argument("checkpoint optimizer config mismatch");
    }
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        const auto& [expected_name, parameter] = parameters[index];
        const auto& saved = checkpoint.parameters[index];
        if (saved.name != expected_name) throw std::invalid_argument("checkpoint parameter name mismatch");
        if (saved.tensor.shape() != parameter->data().shape()) {
            throw std::invalid_argument("checkpoint parameter shape mismatch");
        }
        parameter->mutable_data() =
            Tensor::from_vector(saved.tensor.to_vector(), saved.tensor.shape());
    }
    optimizer.load_state(checkpoint.optimizer_state);
    experiment = checkpoint.experiment;
}

}  // namespace microllm::training
