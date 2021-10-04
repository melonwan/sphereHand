#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__device__ static float atomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void kernel(
  const int num_batches,
  const int num_faces,
  const int width,
  const int height,
  const float *face_vertices,
  float *depth_map){
    int tid = blockIdx.x;
    int fn = tid % num_faces;
    int bn = tid / num_faces;

    // get the corresponding face
    const float* face = &face_vertices[tid*9];

    // return if backside
    if ((face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0])) return;

    // reorder the points
    // pi[0], pi[1], pi[2] = leftmost, middle, rightmost points
    int pi[3];
    if (face[0] < face[3]) {
        if (face[6] < face[0]) pi[0] = 2; else pi[0] = 0;
        if (face[3] < face[6]) pi[2] = 2; else pi[2] = 1;
    } else {
        if (face[6] < face[3]) pi[0] = 2; else pi[0] = 1;
        if (face[0] < face[6]) pi[2] = 2; else pi[2] = 0;
    }
    for (int k = 0; k < 3; k++) if (pi[0] != k && pi[2] != k) pi[1] = k;

    // create new memory for the face point
    float p[3][3];
    for (int num = 0; num < 3; num++) {
        for (int dim = 0; dim < 3; dim++) {
            p[num][dim] = face[3 * pi[num] + dim];
        }
    }
    if (p[0][0] == p[2][0]) return; // line, not triangle

    // compute face_inv
    float face_inv[9] = {
      p[1][1] - p[2][1], p[2][0] - p[1][0], p[1][0] * p[2][1] - p[2][0] * p[1][1],
      p[2][1] - p[0][1], p[0][0] - p[2][0], p[2][0] * p[0][1] - p[0][0] * p[2][1],
      p[0][1] - p[1][1], p[1][0] - p[0][0], p[0][0] * p[1][1] - p[1][0] * p[0][1]};
  float face_inv_denominator = (
      p[2][0] * (p[0][1] - p[1][1]) +
      p[0][0] * (p[1][1] - p[2][1]) +
      p[1][0] * (p[2][1] - p[0][1]));
  for (int k = 0; k < 9; k++) face_inv[k] /= face_inv_denominator;

  // from left to right
  const int32_t xi_min = max(ceil(p[0][0]), 0.);
  const int32_t xi_max = min(p[2][0], width - 1.);
  for (int32_t xi = xi_min; xi <= xi_max; xi++) {
      // compute yi_min and yi_max
      float yi1, yi2;
      if (xi <= p[1][0]) {
          if (p[1][0] - p[0][0] != 0) {
              yi1 = (p[1][1] - p[0][1]) / (p[1][0] - p[0][0]) * (xi - p[0][0]) + p[0][1];
          } else {
              yi1 = p[1][1];
          }
      } else {
          if (p[2][0] - p[1][0] != 0) {
              yi1 = (p[2][1] - p[1][1]) / (p[2][0] - p[1][0]) * (xi - p[1][0]) + p[1][1];
          } else {
              yi1 = p[1][1];
          }
      }
      yi2 = (p[2][1] - p[0][1]) / (p[2][0] - p[0][0]) * (xi - p[0][0]) + p[0][1];

      // loop over all pixels within the triangle
      int32_t yi_min = max(0., ceil(min(yi1, yi2)));
      int32_t yi_max = min(max(yi1, yi2), height - 1.);
      for (int32_t yi = yi_min; yi <= yi_max; yi++) {
        // index in output buffers
        int index = bn * width * height + yi * width + xi;

        // compute barycentric coordinate
        // w = face_inv * p
        float w[3];
        for (int k = 0; k < 3; k++)
            w[k] = face_inv[3 * k + 0] * xi + face_inv[3 * k + 1] * yi + face_inv[3 * k + 2];

        float w_sum = 0;
        for (int k = 0; k < 3; k++) {
            w[k] = min(max(w[k], 0.), 1.);
            w_sum += w[k];
        }
        for (int k = 0; k < 3; k++) w[k] /= w_sum;

        // compute 1 / zp = sum(w / z)
        const float zp = 1. / (w[0] / p[0][2] + w[1] / p[1][2] + w[2] / p[2][2]);
        atomicMin(&depth_map[index], zp);
      }
  }
}

at::Tensor depth_rasterization_cuda_forward(
    int32_t width,
    int32_t height,
    at::Tensor vertices) {
  
  const auto batch_size = vertices.size(0);
  const auto face_size = vertices.size(1);
  at::Tensor depth_map = at::ones({batch_size, height, width}, vertices.options()) * 1000.0;

  AT_DISPATCH_FLOATING_TYPES(vertices.type(), "depth_rasterization_forward_cuda", ([&] {
      kernel<<<batch_size*face_size, 1>>>(
          batch_size, 
          face_size, 
          width, 
          height, 
          vertices.data<float>(), 
          depth_map.data<float>());
  }));
  return depth_map;
}