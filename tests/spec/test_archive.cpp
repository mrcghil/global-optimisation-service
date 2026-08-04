// tests/spec/test_archive.cpp
#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include "goss/sim/archive.hpp"
#include "goss/spec/executor.hpp"
#include "goss/spec/json.hpp"  // from_json for RunSpec
#include "goss/spec/registry.hpp"
#include "goss/spec/specs.hpp"
#include "queue_fixture.hpp"

using goss::spec::test::build_queue;

namespace {

goss::spec::RunSpec base_run(const std::string& root) {
    goss::spec::RunSpec run;
    run.problem = {"queue", "v1"};
    run.parameters = {{"arrival_rate", 2.0}, {"cost_weight", 0.1}};
    run.discretization.scheme = "hermite_simpson";
    run.discretization.t_initial = 0.0;
    run.discretization.t_final = 5.0;
    run.discretization.num_intervals = 25;
    run.solver.kind = "ipopt";
    run.storage.root = root;
    run.image_pipeline = "trajectory_overlay";
    run.label = "archive_test";
    return run;
}

std::string temp_root() {
    const auto dir = std::filesystem::temp_directory_path() /
                     ("goss_archive_test_" + std::to_string(::getpid()));
    return dir.string();
}

goss::spec::RunArchive solved(const std::string& root) {
    goss::spec::ProblemRegistry registry;
    registry.register_problem({"queue", "v1"}, build_queue);
    return goss::spec::execute_run(base_run(root), registry);
}

}  // namespace

TEST(Archive, PathHelpersFollowStoragePrecedence) {
    const std::string root = temp_root();
    goss::spec::RunArchive a = solved(root);

    const std::string dir = goss::sim::resolve_run_dir(a);
    EXPECT_NE(dir.find("queue"), std::string::npos);
    EXPECT_NE(dir.find("v1"), std::string::npos);
    EXPECT_EQ(goss::sim::archive_path(a), dir + "/" + a.run_id + ".h5");
    EXPECT_EQ(goss::sim::sidecar_path(a), dir + "/" + a.run_id + ".json");
}

TEST(Archive, SidecarRoundTripsSpecAndCarriesImagePipeline) {
    const std::string root = temp_root();
    goss::spec::RunArchive a = solved(root);
    const std::string path = goss::sim::sidecar_path(a);
    goss::sim::write_sidecar(path, a);

    ASSERT_TRUE(std::filesystem::exists(path));
    std::ifstream in(path);
    const nlohmann::json j = nlohmann::json::parse(in);
    EXPECT_EQ(j.at("run_id").get<std::string>(), a.run_id);
    EXPECT_EQ(j.at("result").at("status").get<std::string>(), "Success");
    // The image_pipeline name is carried through as the downstream hook.
    const auto restored = j.at("spec").get<goss::spec::RunSpec>();
    EXPECT_EQ(restored.image_pipeline, "trajectory_overlay");

    std::filesystem::remove_all(root);
}

#ifdef GOSS_HAVE_HDF5
TEST(Archive, Hdf5RoundTripPreservesTrajectory) {
    const std::string root = temp_root();
    goss::spec::RunArchive a = solved(root);
    const std::string path = goss::sim::archive_path(a);
    goss::sim::write_run_archive(path, a);
    ASSERT_TRUE(std::filesystem::exists(path));

    const goss::spec::RunArchive reloaded = goss::sim::read_run_archive(path);
    EXPECT_EQ(reloaded.run_id, a.run_id);
    EXPECT_EQ(reloaded.result.status, goss::solver::SolverStatus::Success);
    EXPECT_NEAR(reloaded.result.objective_value, a.result.objective_value, 1e-12);
    EXPECT_EQ(reloaded.spec.problem.name, "queue");
    EXPECT_EQ(reloaded.spec.image_pipeline, "trajectory_overlay");

    // The trajectory survives the round-trip byte-for-byte.
    ASSERT_EQ(reloaded.trajectory.times.size(), a.trajectory.times.size());
    const auto& q_in = a.trajectory.state("queue_length");
    const auto& q_out = reloaded.trajectory.state("queue_length");
    ASSERT_EQ(q_out.size(), q_in.size());
    for (std::size_t k = 0; k < q_in.size(); ++k)
        EXPECT_DOUBLE_EQ(q_out[k], q_in[k]) << "node " << k;

    std::filesystem::remove_all(root);
}
#endif  // GOSS_HAVE_HDF5
