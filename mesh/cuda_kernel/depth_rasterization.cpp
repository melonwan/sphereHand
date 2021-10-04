#include <torch/torch.h>
#include <vector>

// C++ interface
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

at::Tensor depth_rasterization_forward(
    int32_t width,
    int32_t height,
    at::Tensor vertices) {
  CHECK_INPUT(vertices);

  const auto batch_size = vertices.size(0);
  auto depth_map = at::zeros({batch_size, height, width});
  return depth_map;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &depth_rasterization_forward, "Depth rasterization forward");
}