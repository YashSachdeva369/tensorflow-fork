/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#endif

#if GOOGLE_CUDA && CUDA_VERSION >= 12060
#define EIGEN_USE_GPU

#include <vector>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg/linalg_ops_common.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_solvers.h"

namespace tensorflow {
namespace {

using GPUDevice = Eigen::GpuDevice;

template <typename InputScalar, typename OutputScalar>
__global__ void ConvertEigVectorsKernel(
    GpuLaunchConfig config, int64_t n,
    const InputScalar* __restrict__ input_vectors,
    const OutputScalar* __restrict__ eigenvalues,
    OutputScalar* __restrict__ output_vectors) {
  using RealScalar = typename Eigen::NumTraits<InputScalar>::Real;
  using CudaComplex = typename CUDAComplexT<OutputScalar>::type;
  const CudaComplex* values = reinterpret_cast<const CudaComplex*>(eigenvalues);
  CudaComplex* output = reinterpret_cast<CudaComplex*>(output_vectors);

  GPU_1D_KERNEL_LOOP(index, config.virtual_thread_count) {
    const int64_t matrix_size = n * n;
    const int64_t batch = index / matrix_size;
    const int64_t matrix_index = index % matrix_size;
    const int64_t row = matrix_index / n;
    const int64_t column = matrix_index % n;
    const int64_t column_major_index = batch * matrix_size + row + column * n;

    if constexpr (Eigen::NumTraits<InputScalar>::IsComplex) {
      const CudaComplex* vectors =
          reinterpret_cast<const CudaComplex*>(input_vectors);
      output[index] = vectors[column_major_index];
    } else {
      const InputScalar* vectors = input_vectors;
      const RealScalar eigenvalue_imag = values[batch * n + column].y;
      RealScalar real = vectors[column_major_index];
      RealScalar imag = RealScalar(0);
      if (eigenvalue_imag > RealScalar(0)) {
        imag = vectors[column_major_index + n];
      } else if (eigenvalue_imag < RealScalar(0)) {
        real = vectors[column_major_index - n];
        imag = -vectors[column_major_index];
      }
      output[index].x = real;
      output[index].y = imag;
    }
  }
}

template <class InputScalar, class OutputScalar>
class EigOpGpu : public AsyncOpKernel {
 public:
  explicit EigOpGpu(OpKernelConstruction* context) : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("compute_v", &compute_v_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) final {
    const Tensor& input = context->input(0);
    OP_REQUIRES_ASYNC(context, input.dims() >= 2,
                      errors::InvalidArgument("Input must have rank >= 2."),
                      done);

    const int64_t n = input.dim_size(input.dims() - 1);
    OP_REQUIRES_ASYNC(
        context, input.dim_size(input.dims() - 2) == n,
        errors::InvalidArgument("Input matrices must be square."), done);
    TensorShape batch_shape = input.shape();
    batch_shape.RemoveLastDims(2);
    const int64_t batch_size = batch_shape.num_elements();

    TensorShape eigenvalue_shape = batch_shape;
    eigenvalue_shape.AddDim(n);
    Tensor* eigenvalues = nullptr;
    OP_REQUIRES_OK_ASYNC(
        context, context->allocate_output(0, eigenvalue_shape, &eigenvalues),
        done);

    Tensor* eigenvectors = nullptr;
    if (compute_v_) {
      OP_REQUIRES_OK_ASYNC(
          context, context->allocate_output(1, input.shape(), &eigenvectors),
          done);
    }

    if (n == 0 || batch_size == 0) {
      done();
      return;
    }

    std::unique_ptr<GpuSolver> solver(new GpuSolver(context));
    Tensor input_copy;
    OP_REQUIRES_OK_ASYNC(
        context,
        solver->allocate_scoped_tensor(input.dtype(), input.shape(),
                                       &input_copy),
        done);
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    OP_REQUIRES_OK_ASYNC(context, DoMatrixTranspose(device, input, &input_copy),
                         done);

    Tensor vectors_copy;
    InputScalar* vectors_ptr = nullptr;
    if (compute_v_) {
      OP_REQUIRES_OK_ASYNC(
          context,
          solver->allocate_scoped_tensor(input.dtype(), input.shape(),
                                         &vectors_copy),
          done);
      vectors_ptr = vectors_copy.flat<InputScalar>().data();
    }

    std::vector<DeviceLapackInfo> dev_info;
    dev_info.push_back(solver->GetDeviceLapackInfo(batch_size, "geev"));
    OP_REQUIRES_OK_ASYNC(
        context,
        solver->template Geev<InputScalar, OutputScalar>(
            n, input_copy.flat<InputScalar>().data(),
            eigenvalues->flat<OutputScalar>().data(), vectors_ptr,
            dev_info.back().mutable_data(), batch_size),
        done);

    if (compute_v_) {
      GpuLaunchConfig config =
          GetGpuLaunchConfig(eigenvectors->NumElements(), device);
      OP_REQUIRES_OK_ASYNC(
          context,
          GpuLaunchKernel(
              ConvertEigVectorsKernel<InputScalar, OutputScalar>,
              config.block_count, config.thread_per_block, 0, device.stream(),
              config, n, vectors_copy.flat<InputScalar>().data(),
              eigenvalues->flat<OutputScalar>().data(),
              eigenvectors->flat<OutputScalar>().data()),
          done);
    }

    GpuSolver::CheckLapackInfoAndDeleteSolverAsync(std::move(solver), dev_info,
                                                   std::move(done));
  }

 private:
  bool compute_v_;
};

}  // namespace

REGISTER_LINALG_OP_GPU("Eig", (EigOpGpu<float, complex64>), float);
REGISTER_LINALG_OP_GPU("Eig", (EigOpGpu<double, complex128>), double);
REGISTER_LINALG_OP_GPU("Eig", (EigOpGpu<complex64, complex64>), complex64);
REGISTER_LINALG_OP_GPU("Eig", (EigOpGpu<complex128, complex128>), complex128);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA && CUDA_VERSION >= 12060
