#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Compute_count_data_gpu(int nthreads, const int M, const Dtype* label, Dtype* count) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    count[index] = 0;
    for (int m = 0; m < M; m++) {
      const int label_value = static_cast<int>(label[m]);
      if (label_value == index) {
        count[index]++;
      }
    }
  }
}

template <typename Dtype>
__global__ void Compute_distance_data_gpu(int nthreads, const int K, const Dtype* bottom,
	      const Dtype* label, const Dtype* center, const Dtype* gamma, Dtype* distance) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int m = index / K;
    int k = index % K;
    const int label_value = static_cast<int>(label[m]);
    // distance(i) = x(i) - g_{y(i)} * c_{y(i)}
    distance[index] = bottom[index] - gamma[label_value] * center[label_value * K + k];
  }
}

template <typename Dtype>
__global__ void Compute_center_diff_gpu(int nthreads, const int M, const int K, 
        const Dtype* label, const Dtype* distance, const Dtype* count, 
        Dtype* center_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / K;
    int k = index % K;
    for (int m = 0; m < M; m++) {
      const int label_value = static_cast<int>(label[m]);
      if (label_value == n) {
        center_diff[index] = center_diff[index] - distance[m * K + k] / count[label_value];
      }
    }
  }
}

template <typename Dtype>
__global__ void Compute_gamma_diff_gpu_(int nthreads, const int M, const int K,
        const Dtype* bottom, const Dtype* label, const Dtype* center, 
        const Dtype* gamma, const Dtype* count, Dtype* gamma_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // gamma_diff = gamma - x'w/w'w;
    for (int m = 0; m < M; m++) {
      const int label_value = static_cast<int>(label[m]);
      if (index == label_value) {
        Dtype dot_wx = (Dtype)0.;
        Dtype dot_ww = (Dtype)0.;
        for (int k = 0; k < K; k++) {
          dot_wx += bottom[m * K + k] * center[index * K + k];
          dot_ww += center[index * K + k] * center[index * K + k];
        }
        gamma_diff[index] += (gamma[index] - dot_wx / dot_ww) / count[index];
      }
    }
  }
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // simplified
  if (!this->layer_param_.center_loss_param().simplified()) {
    caffe_gpu_set(N_, Dtype(1), this->blobs_[1]->mutable_gpu_data());
  }

  int nthreads = N_;
  Compute_count_data_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, bottom[1]->gpu_data(), count_.mutable_gpu_data());

  nthreads = M_ * K_;
  Compute_distance_data_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
                                this->blobs_[0]->gpu_data(), this->blobs_[1]->gpu_data(), 
                                distance_.mutable_gpu_data());

  Dtype dot;
  caffe_gpu_dot(M_ * K_, distance_.gpu_data(), distance_.gpu_data(), &dot);
  Dtype loss = dot / M_ / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->iter_idx_ % 2 == 1) {
	//LOG(INFO)<<this->iter_idx_;
	  if (this->param_propagate_down_[0] && !this->layer_param_.center_loss_param().simplified()) {
		int nthreads = N_ * K_;
		Compute_center_diff_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
			CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, K_, bottom[1]->gpu_data(), distance_.gpu_data(), 
									  count_.gpu_data(), this->blobs_[0]->mutable_gpu_diff());
	  }
	  if (this->param_propagate_down_[1] && this->layer_param_.center_loss_param().simplified()) {
		int nthreads = N_;
		Compute_gamma_diff_gpu_<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
		  CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, K_, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
									this->blobs_[0]->gpu_data(), this->blobs_[1]->gpu_data(), count_.gpu_data(),
									this->blobs_[1]->mutable_gpu_diff());
	  }

	  if (propagate_down[0]) {
		caffe_gpu_scale(M_ * K_, top[0]->cpu_diff()[0] / M_, 
								 distance_.gpu_data(), bottom[0]->mutable_gpu_diff());
	  }
	  if (propagate_down[1]) {
		LOG(FATAL) << this->type()
				   << " Layer cannot backpropagate to label inputs.";
	  }
  }
  else{
	caffe_gpu_set(
			  bottom[0]->count(),             
			  Dtype(0),                             
			  bottom[0]->mutable_gpu_diff()); 
	caffe_gpu_set(
			  bottom[1]->count(),             
			  Dtype(0),                             
			  bottom[1]->mutable_gpu_diff()); 
  }
  this->iter_idx_++;
  
}

INSTANTIATE_LAYER_GPU_FUNCS(CenterLossLayer);

}  // namespace caffe
