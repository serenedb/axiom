/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "axiom/logical_plan/PlanBuilder.h"
#include "axiom/optimizer/tests/HiveQueriesTestBase.h"
#include "axiom/optimizer/tests/PlanMatcher.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::axiom::optimizer::test {
namespace {

using namespace facebook::velox;
namespace lp = facebook::axiom::logical_plan;

class HiveCrossJoinQueriesTest : public test::HiveQueriesTestBase {
 protected:
  static core::PlanMatcherBuilder matchScan(const std::string& tableName) {
    return core::PlanMatcherBuilder().tableScan(tableName);
  }
};

TEST_F(HiveCrossJoinQueriesTest, basic) {
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto scan = [&](const std::string& tableName) {
    return exec::test::PlanBuilder(planNodeIdGenerator)
        .tableScan(tableName, getSchema(tableName));
  };

  checkResults(
      "SELECT * FROM nation, region",
      scan("nation")
          .nestedLoopJoin(
              scan("region").planNode(),
              "",
              {"n_nationkey",
               "n_name",
               "n_regionkey",
               "n_comment",
               "r_regionkey",
               "r_name",
               "r_comment"})
          .planNode());

  {
    auto plan =
        scan("nation")
            .nestedLoopJoin(
                scan("region").planNode(),
                "",
                {
                    "n_name",
                    "r_name",
                })
            .nestedLoopJoin(
                scan("region").project({"r_name as r2_name"}).planNode(),
                "",
                {"n_name", "r_name", "r2_name"})
            .planNode();
    checkResults(
        "SELECT n.n_name, r1.r_name AS r_name, r2.r_name AS r2_name FROM nation n, region r1, region r2",
        plan);
  }

  checkResults(
      "SELECT c.c_custkey, n.n_name, r.r_name FROM customer c INNER JOIN nation n ON c.c_nationkey = n.n_regionkey CROSS JOIN region r",
      scan("customer")
          .project({"c_custkey", "c_nationkey"})
          .hashJoin(
              {"c_nationkey"},
              {"n_regionkey"},
              scan("nation").project({"n_name", "n_regionkey"}).planNode(),
              "",
              {"c_custkey", "c_nationkey", "n_name"})
          .nestedLoopJoin(
              scan("region").project({"r_name"}).planNode(),
              "",
              {"c_custkey", "n_name", "r_name"})
          .planNode());
}

TEST_F(HiveCrossJoinQueriesTest, filterPushdown) {
  lp::PlanBuilder::Context context(exec::test::kHiveConnectorId);
  auto logicalPlan =
      lp::PlanBuilder(context)
          .tableScan("nation")
          .as("n")
          .crossJoin(lp::PlanBuilder(context).tableScan("region").as("r"))
          .crossJoin(lp::PlanBuilder(context).tableScan("customer"))
          .filter("n.n_regionkey != r.r_regionkey")
          .build();

  {
    auto plan = toSingleNodePlan(logicalPlan);

    auto matcher = matchScan("nation")
                       .nestedLoopJoin(matchScan("region").build())
                       .filter("n_regionkey != r_regionkey")
                       .nestedLoopJoin(matchScan("customer").build())
                       .build();
    ASSERT_TRUE(matcher->match(plan));
  }

  {
    auto plan = planVelox(logicalPlan);
    const auto& fragments = plan.plan->fragments();
    ASSERT_EQ(2, fragments.size());

    auto matcher = matchScan("nation")
                       .nestedLoopJoin(matchScan("region").build())
                       .filter("n_regionkey != r_regionkey")
                       .nestedLoopJoin(matchScan("customer").build())
                       .partitionedOutput()
                       .build();

    ASSERT_TRUE(matcher->match(fragments.at(0).fragment.planNode));

    matcher = core::PlanMatcherBuilder().exchange().build();

    ASSERT_TRUE(matcher->match(fragments.at(1).fragment.planNode));
  }
}

TEST_F(HiveCrossJoinQueriesTest, manyTables) {
  lp::PlanBuilder::Context context(exec::test::kHiveConnectorId);
  auto logicalPlan =
      lp::PlanBuilder(context)
          .tableScan("nation")

          .crossJoin(lp::PlanBuilder(context).tableScan("region"))
          .crossJoin(lp::PlanBuilder(context).tableScan("customer"))
          .crossJoin(lp::PlanBuilder(context).tableScan("lineitem"))
          .build();

  {
    auto plan = toSingleNodePlan(logicalPlan);

    auto matcher = matchScan("lineitem")
                       .nestedLoopJoin(matchScan("nation").build())
                       .nestedLoopJoin(matchScan("customer").build())
                       .nestedLoopJoin(matchScan("region").build())
                       .project()
                       .build();
    ASSERT_TRUE(matcher->match(plan));
  }

  {
    auto plan = planVelox(logicalPlan);
    const auto& fragments = plan.plan->fragments();
    ASSERT_EQ(2, fragments.size());

    auto matcher = matchScan("lineitem")
                       .nestedLoopJoin(matchScan("nation").build())
                       .nestedLoopJoin(matchScan("customer").build())
                       .nestedLoopJoin(matchScan("region").build())
                       .project()
                       .partitionedOutput()
                       .build();

    ASSERT_TRUE(matcher->match(fragments.at(0).fragment.planNode));

    matcher = core::PlanMatcherBuilder().exchange().build();

    ASSERT_TRUE(matcher->match(fragments.at(1).fragment.planNode));
  }
}

TEST_F(HiveCrossJoinQueriesTest, innerJoin) {
  lp::PlanBuilder::Context context(exec::test::kHiveConnectorId);
  auto logicalPlan =
      lp::PlanBuilder(context)
          .tableScan("nation")
          .as("n")
          .join(
              lp::PlanBuilder(context)
                  .tableScan("region", getSchema("region")->names())
                  .as("r"),
              "n.n_regionkey = r.r_regionkey",
              lp::JoinType::kInner)
          .crossJoin(lp::PlanBuilder(context).tableScan(
              "customer", getSchema("customer")->names()))
          .build();

  {
    auto plan = toSingleNodePlan(logicalPlan);
    auto matcher = matchScan("customer")
                       .nestedLoopJoin(matchScan("region").build())
                       .hashJoin(matchScan("nation").build())
                       .build();

    ASSERT_TRUE(matcher->match(plan));
  }

  {
    auto plan = planVelox(logicalPlan);
    const auto& fragments = plan.plan->fragments();
    ASSERT_EQ(3, fragments.size());

    auto matcher = matchScan("nation").partitionedOutput().build();
    ASSERT_TRUE(matcher->match(fragments.at(0).fragment.planNode));

    matcher = matchScan("customer")
                  .nestedLoopJoin(matchScan("region").build())
                  .hashJoin(core::PlanMatcherBuilder().exchange().build())
                  .partitionedOutput()
                  .build();
    ASSERT_TRUE(matcher->match(fragments.at(1).fragment.planNode));

    matcher = core::PlanMatcherBuilder().exchange().build();
    ASSERT_TRUE(matcher->match(fragments.at(2).fragment.planNode));
  }
}

} // namespace
} // namespace facebook::axiom::optimizer::test
