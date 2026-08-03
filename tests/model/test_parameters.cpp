// tests/model/test_parameters.cpp
#include <gtest/gtest.h>
#include <string>
#include "goss/model/model.hpp"
#include "goss/model/parameter.hpp"
#include "goss/model/errors.hpp"

TEST(ModelParameters, ValidatorAcceptsInRangeAndReportsDefaults) {
    goss::model::Model model;
    model.add_parameter("arrival_rate", /*default=*/2.0, /*lower=*/0.0, /*upper=*/10.0);
    model.add_parameter("cost_weight",  /*default=*/1.0, /*lower=*/0.0, /*upper=*/100.0);

    ASSERT_EQ(model.num_parameters(), 2u);
    auto validator = model.parameter_validator();
    ASSERT_EQ(validator.size(), 2u);
    EXPECT_EQ(validator.defaults(), (std::vector<double>{2.0, 1.0}));
    EXPECT_NO_THROW(validator.validate({3.0, 50.0}));
}

TEST(ModelParameters, ValidatorRejectsWrongSizeWithExplicitMessage) {
    goss::model::Model model;
    model.add_parameter("arrival_rate", 2.0, 0.0, 10.0);
    auto validator = model.parameter_validator();
    try {
        validator.validate({1.0, 2.0});
        FAIL() << "expected ModelError";
    } catch (const goss::model::ModelError& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("expected 1"), std::string::npos);
        EXPECT_NE(message.find("got 2"), std::string::npos);
    }
}

TEST(ModelParameters, ValidatorRejectsOutOfRangeNamingTheParameter) {
    goss::model::Model model;
    model.add_parameter("arrival_rate", 2.0, 0.0, 10.0);
    auto validator = model.parameter_validator();
    try {
        validator.validate({99.0});
        FAIL() << "expected ModelError";
    } catch (const goss::model::ModelError& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("arrival_rate"), std::string::npos);  // names the offender
        EXPECT_NE(message.find("99"), std::string::npos);
        EXPECT_NE(message.find("10"), std::string::npos);            // reports the bound
    }
}

TEST(ModelParameters, ParameterNameCollidesWithStateThrows) {
    goss::model::Model model;
    model.add_state("q");
    EXPECT_THROW(model.add_parameter("q", 1.0), goss::model::ModelError);
}
