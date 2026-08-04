// tests/spec/test_registry.cpp
#include <gtest/gtest.h>
#include "goss/spec/errors.hpp"
#include "goss/spec/registry.hpp"
#include "goss/spec/specs.hpp"
#include "queue_fixture.hpp"

using goss::spec::test::build_queue;

TEST(ProblemRegistry, RegisterBuildAtTwoMeshes) {
    goss::spec::ProblemRegistry registry;
    registry.register_problem({"queue", "v1"}, build_queue);

    EXPECT_TRUE(registry.contains({"queue", "v1"}));
    EXPECT_FALSE(registry.contains({"queue", "v2"}));

    goss::spec::DiscretizationSpec coarse;
    coarse.t_initial = 0.0; coarse.t_final = 5.0; coarse.num_intervals = 10;
    goss::spec::DiscretizationSpec fine = coarse;
    fine.num_intervals = 40;

    auto built_coarse = registry.build({"queue", "v1"}, coarse);
    auto built_fine   = registry.build({"queue", "v1"}, fine);

    EXPECT_EQ(built_coarse.mesh.num_intervals, 10u);
    EXPECT_EQ(built_fine.mesh.num_intervals, 40u);
    EXPECT_EQ(built_coarse.model.num_parameters(), 2u);
    EXPECT_EQ(built_coarse.compiled.layout.num_nodes(), 11u);
    EXPECT_EQ(built_fine.compiled.layout.num_nodes(), 41u);
    EXPECT_NE(built_coarse.compiled.problem.get(), nullptr);
}

TEST(ProblemRegistry, DispatchSelectsScheme) {
    goss::spec::ProblemRegistry registry;
    registry.register_problem({"queue", "v1"}, build_queue);

    goss::spec::DiscretizationSpec trap;
    trap.scheme = "trapezoidal";
    trap.t_initial = 0.0; trap.t_final = 5.0; trap.num_intervals = 20;
    auto built = registry.build({"queue", "v1"}, trap);
    EXPECT_EQ(built.scheme, "trapezoidal");
    EXPECT_EQ(built.compiled.layout.num_nodes(), 21u);
}

TEST(ProblemRegistry, UnknownKeyThrows) {
    goss::spec::ProblemRegistry registry;
    goss::spec::DiscretizationSpec disc;
    EXPECT_THROW(registry.build({"nope", "v1"}, disc), goss::spec::SpecError);
}

TEST(ProblemRegistry, DuplicateRegistrationThrows) {
    goss::spec::ProblemRegistry registry;
    registry.register_problem({"queue", "v1"}, build_queue);
    EXPECT_THROW(registry.register_problem({"queue", "v1"}, build_queue),
                 goss::spec::SpecError);
    // Same name, different version is allowed.
    EXPECT_NO_THROW(registry.register_problem({"queue", "v2"}, build_queue));
}

TEST(CompileDispatch, UnknownSchemeThrows) {
    goss::spec::ProblemRegistry registry;
    registry.register_problem({"queue", "v1"}, build_queue);
    goss::spec::DiscretizationSpec bad;
    bad.scheme = "colocation_typo";
    EXPECT_THROW(registry.build({"queue", "v1"}, bad), goss::spec::SpecError);
}
