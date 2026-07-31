#include <gtest/gtest.h>
#include "goss/transcription/errors.hpp"

TEST(TranscriptionError, IsThrowable) {
    EXPECT_THROW(throw goss::transcription::TranscriptionError("boom"),
                 goss::transcription::TranscriptionError);
}
