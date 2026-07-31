// tests/model/test_model_build.cpp
#include <gtest/gtest.h>
#include "goss/model/model.hpp"
#include "goss/transcription/transcription.hpp"

TEST(ModelSetters, RecordsStateAndControlBounds) {
    goss::model::Model model;
    auto q = model.add_state("q");
    auto r = model.add_control("r");
    model.set_state_bounds(q, 0.0, goss::transcription::kInf);
    model.set_control_bounds(r, -2.0, 2.0);
    EXPECT_DOUBLE_EQ(model.state_lower(0), 0.0);
    EXPECT_DOUBLE_EQ(model.state_upper(0), goss::transcription::kInf);
    EXPECT_DOUBLE_EQ(model.control_lower(0), -2.0);
    EXPECT_DOUBLE_EQ(model.control_upper(0), 2.0);
}

TEST(ModelSetters, RecordsBoundaryConditions) {
    goss::model::Model model;
    auto q = model.add_state("q");
    model.set_initial_state(q, 10.0);
    EXPECT_TRUE(model.initial_fixed(0));
    EXPECT_DOUBLE_EQ(model.initial_value(0), 10.0);
    EXPECT_FALSE(model.final_fixed(0));   // not set → free
}

TEST(ModelSetters, RejectsInvertedBounds) {
    goss::model::Model model;
    auto q = model.add_state("q");
    EXPECT_THROW(model.set_state_bounds(q, 5.0, -5.0), goss::model::ModelError);
}

TEST(ModelSetters, RejectsOutOfRangeHandle) {
    goss::model::Model model;
    model.add_state("q");
    goss::model::StateHandle bogus{7};
    EXPECT_THROW(model.set_state_bounds(bogus, 0.0, 1.0), goss::model::ModelError);
}

// ---- ModelBuild tests ----

namespace {
struct DummyDyn {
    template <typename T>
    std::vector<T> operator()(const std::vector<T>& x, const std::vector<T>& /*u*/, T /*t*/) const {
        return { -x[0] };
    }
};
struct DummyCost {
    template <typename T>
    T operator()(const std::vector<T>&, const std::vector<T>&, T) const { return T(0); }
};
}  // namespace

TEST(ModelBuild, AssemblesOcpProblemFields) {
    goss::model::Model model;
    auto q = model.add_state("q");
    auto r = model.add_control("r");
    model.set_state_bounds(q, 0.0, goss::transcription::kInf);
    model.set_control_bounds(r, -1.0, 1.0);
    model.set_initial_state(q, 10.0);
    model.set_mesh(0.0, 2.0, 5);

    auto ocp = model.build(DummyDyn{}, DummyCost{});
    EXPECT_EQ(ocp.num_states, 1u);
    EXPECT_EQ(ocp.num_controls, 1u);
    ASSERT_EQ(ocp.state_lower.size(), 1u);
    EXPECT_DOUBLE_EQ(ocp.state_lower[0], 0.0);
    ASSERT_EQ(ocp.control_upper.size(), 1u);
    EXPECT_DOUBLE_EQ(ocp.control_upper[0], 1.0);
    ASSERT_EQ(ocp.initial_state_fixed.size(), 1u);
    EXPECT_DOUBLE_EQ(ocp.initial_state_fixed[0], 1.0);   // pinned
    EXPECT_DOUBLE_EQ(ocp.initial_state[0], 10.0);
    ASSERT_EQ(ocp.final_state_fixed.size(), 1u);
    EXPECT_DOUBLE_EQ(ocp.final_state_fixed[0], 0.0);      // free
    EXPECT_EQ(ocp.mesh.num_intervals, 5u);
}

TEST(ModelBuild, RejectsBuildWithoutMesh) {
    goss::model::Model model;
    model.add_state("q");
    EXPECT_THROW(model.build(DummyDyn{}, DummyCost{}), goss::model::ModelError);
}

TEST(ModelBuild, RejectsBuildWithNoStates) {
    goss::model::Model model;
    model.set_mesh(0.0, 1.0, 4);
    EXPECT_THROW(model.build(DummyDyn{}, DummyCost{}), goss::model::ModelError);
}
