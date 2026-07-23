/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "xla/backends/profiler/cpu/python_tracer.h"

#include <any>
#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace {

TEST(PythonTracerTest, SerializeFailsWithNullSpace) {
  PythonTracerOptions options;
  options.enable_trace_python_function = true;
  auto tracer = CreatePythonTracer(options);
  ASSERT_NE(tracer, nullptr);

  std::any data = std::make_any<PythonTracerChunk>();

  absl::Status status = tracer->Serialize(std::move(data), nullptr);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(PythonTracerTest, SerializeFailsWithInvalidDataType) {
  PythonTracerOptions options;
  options.enable_trace_python_function = true;
  auto tracer = CreatePythonTracer(options);
  ASSERT_NE(tracer, nullptr);

  tensorflow::profiler::XSpace space;
  std::any invalid_data =
      std::make_any<int>(42);  // Int instead of PythonTracerChunk

  absl::Status status = tracer->Serialize(std::move(invalid_data), &space);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(PythonTracerTest, SerializeOkWithEmptyPlane) {
  PythonTracerOptions options;
  options.enable_trace_python_function = true;
  auto tracer = CreatePythonTracer(options);
  ASSERT_NE(tracer, nullptr);

  std::any data = std::make_any<PythonTracerChunk>();  // Empty plane

  tensorflow::profiler::XSpace space;
  EXPECT_OK(tracer->Serialize(std::move(data), &space));
  EXPECT_EQ(space.planes_size(), 0);
}

TEST(PythonTracerTest, ConsumeAndSerializeWithPlaneData) {
  PythonTracerOptions options;
  options.enable_trace_python_function = true;
  auto tracer = CreatePythonTracer(options);
  ASSERT_NE(tracer, nullptr);

  // Consume should return an empty chunk when no active session
  ASSERT_OK_AND_ASSIGN(tsl::profiler::ConsumeResult consume_result,
                       tracer->Consume());
  PythonTracerChunk* chunk =
      std::any_cast<PythonTracerChunk>(&consume_result.data);
  ASSERT_NE(chunk, nullptr);
  EXPECT_EQ(chunk->plane.lines_size(), 0);

  // Add dummy data to chunk plane to verify Serialize logic
  chunk->plane.set_name(tsl::profiler::kPythonTracerPlaneName);
  tensorflow::profiler::XLine* line = chunk->plane.add_lines();
  line->set_id(123);
  line->set_name("MainThread");

  tensorflow::profiler::XSpace space;
  EXPECT_OK(tracer->Serialize(std::move(consume_result.data), &space));
  ASSERT_EQ(space.planes_size(), 1);
  EXPECT_EQ(space.planes(0).name(), ::tsl::profiler::kPythonTracerPlaneName);
  EXPECT_EQ(space.planes(0).lines_size(), 1);
  EXPECT_EQ(space.planes(0).lines(0).id(), 123);
}

TEST(PythonTracerTest, StopWithoutStartFails) {
  PythonTracerOptions options;
  options.enable_trace_python_function = true;
  auto tracer = CreatePythonTracer(options);
  ASSERT_NE(tracer, nullptr);

  absl::Status status = tracer->Stop();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
}

TEST(PythonTracerTest, DoubleStartFails) {
  PythonTracerOptions options;
  options.enable_trace_python_function = true;
  auto tracer = CreatePythonTracer(options);
  ASSERT_NE(tracer, nullptr);

  EXPECT_OK(tracer->Start());
  absl::Status status = tracer->Start();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);

  EXPECT_OK(tracer->Stop());
}

TEST(PythonTracerTest, RegularProfilingStartStopCollectData) {
  PythonTracerOptions options;
  options.enable_trace_python_function = true;
  auto tracer = CreatePythonTracer(options);
  ASSERT_NE(tracer, nullptr);

  EXPECT_OK(tracer->Start());
  EXPECT_OK(tracer->Stop());

  tensorflow::profiler::XSpace space;
  EXPECT_OK(tracer->CollectData(&space));
}

}  // namespace
}  // namespace profiler
}  // namespace xla
