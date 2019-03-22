#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CenterLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  iter_idx_ = 1;
  const int num_output = this->layer_param_.center_loss_param().num_output();  
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.center_loss_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(2);
    // Intialize the weight
    vector<int> center_shape(2);
    center_shape[0] = N_;
    center_shape[1] = K_;
    this->blobs_[0].reset(new Blob<Dtype>(center_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > center_filler(GetFiller<Dtype>(
        this->layer_param_.center_loss_param().center_filler()));
    center_filler->Fill(this->blobs_[0].get());

    // Intiialize and fill the amplitute term
    vector<int> gamma_shape(1, N_);
    this->blobs_[1].reset(new Blob<Dtype>(gamma_shape));
//    shared_ptr<Filler<Dtype> > gamma_filler(GetFiller<Dtype>(
//        this->layer_param_.center_loss_param().gamma_filler()));
//    gamma_filler->Fill(this->blobs_[1].get());
    // simplified
    if (!this->layer_param_.center_loss_param().simplified()) {
      caffe_set(N_, (Dtype)1., this->blobs_[1]->mutable_cpu_data());
    }

  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  M_ = bottom[0]->num();
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  LossLayer<Dtype>::Reshape(bottom, top);
  distance_.ReshapeLike(*bottom[0]);
  vector<int> shape_1_X_N(1, N_);
  count_.Reshape(shape_1_X_N);
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* center = this->blobs_[0]->cpu_data();
  const Dtype* gamma = this->blobs_[1]->cpu_data();
  Dtype* distance_data = distance_.mutable_cpu_data();
  Dtype* count_data = count_.mutable_cpu_data();
 
  caffe_set(N_, (Dtype)0., count_data);
  caffe_copy(M_ * K_, bottom_data, distance_data); 
  // the i-th distance_data
  for (int i = 0; i < M_; i++) {
    const int label_value = static_cast<int>(label[i]);
    // D(i,:) = X(i,:) - C(y(i),:)
    caffe_axpy(K_, - gamma[label_value], center + label_value * K_, distance_data + i * K_);
    count_data[label_value]++;
  }
  Dtype dot = caffe_cpu_dot(M_ * K_, distance_.cpu_data(), distance_.cpu_data());
  Dtype loss = dot / M_ / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // Gradient with respect to centers
  
  if (this->iter_idx_ % 2 == 1) {
	  
	  if (this->param_propagate_down_[0] && !this->layer_param_.center_loss_param().simplified()) {
		const Dtype* label = bottom[1]->cpu_data();
		const Dtype* distance_data = distance_.cpu_data();
		const Dtype* count_data = count_.cpu_data();
		Dtype* center_diff = this->blobs_[0]->mutable_cpu_diff();

		for (int m = 0; m < M_; m++) {
		  const int label_value = static_cast<int>(label[m]);
		  caffe_axpy(K_, - (Dtype)1. / count_data[label_value], 
						   distance_data + m * K_, center_diff + label_value * K_);
		}
	  }

	  if (this->param_propagate_down_[1] && this->layer_param_.center_loss_param().simplified()) {
		const Dtype* label = bottom[1]->cpu_data();
		const Dtype* center = this->blobs_[0]->cpu_data();
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* gamma_data = this->blobs_[1]->cpu_data();
		const Dtype* count_data = count_.cpu_data();
		Dtype* gamma_diff = this->blobs_[1]->mutable_cpu_diff();
		for (int i = 0; i < M_; i++) {
		  const int label_value = static_cast<int>(label[i]);
		  Dtype dot_xw = caffe_cpu_dot(K_, bottom_data + i * K_, center + label_value * K_);
		  Dtype dot_ww = caffe_cpu_dot(K_, center + label_value * K_, center + label_value * K_);
		  gamma_diff[label_value] += (gamma_data[label_value] - dot_xw / dot_ww) / count_data[label_value];
		}
	  }

	  // Gradient with respect to bottom data 
	  if (propagate_down[0]) {
		caffe_copy(M_ * K_, distance_.cpu_data(), bottom[0]->mutable_cpu_diff());
		caffe_scal(M_ * K_, top[0]->cpu_diff()[0] / M_, bottom[0]->mutable_cpu_diff());
	  }
	  if (propagate_down[1]) {
		LOG(FATAL) << this->type()
				   << "Layer cannot backpropagate to label inputs.";
	  }
  }
  this->iter_idx_++;
}

#ifdef CPU_ONLY
STUB_GPU(CenterLossLayer);
#endif

INSTANTIATE_CLASS(CenterLossLayer);
REGISTER_LAYER_CLASS(CenterLoss);

}  // namespace caffe
