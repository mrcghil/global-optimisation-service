#pragma once
#include <stdexcept>
#include <string>
namespace goss::transcription {
class TranscriptionError : public std::runtime_error {
 public:
    explicit TranscriptionError(const std::string& message) : std::runtime_error(message) {}
};
}  // namespace goss::transcription
