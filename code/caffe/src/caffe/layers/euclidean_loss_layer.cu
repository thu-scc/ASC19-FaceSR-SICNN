#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //iter_num = this->layer_param_.iter_num();
  //if (this->iter_idx_ % 2 == iter_num) {
	  int count = bottom[0]->count();
	  caffe_gpu_sub(
		  count,
		  bottom[0]->gpu_data(),
		  bottom[1]->gpu_data(),
		  diff_.mutable_gpu_data());
	  Dtype dot;
	  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
	  Dtype loss = dot / bottom[0]->num() / Dtype(2);
	  top[0]->mutable_cpu_data()[0] = loss;
  //}
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int iter_num = this->layer_param_.iter_num();
  //LOG(INFO)<<this->iter_idx_;
  //if (this->iter_idx_ % 4 > iter_num || this->iter_idx_ % 4 == 0) {
  if (this->iter_idx_ % 2 == iter_num) {
	  //LOG(INFO)<<this->iter_idx_;
	  for (int i = 0; i < 2; ++i) {
		if (propagate_down[i]) {
		  const Dtype sign = (i == 0) ? 1 : -1;
		  const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
		  caffe_gpu_axpby(
			  bottom[i]->count(),              // count
			  alpha,                              // alpha
			  diff_.gpu_data(),                   // a
			  Dtype(0),                           // beta
			  bottom[i]->mutable_gpu_diff());  // b
		}
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

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossLayer);

}  // namespace caffe
