#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/box_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
BoxDataLayer<Dtype>::BoxDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
BoxDataLayer<Dtype>::~BoxDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void BoxDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  this->box_label_ = false;
  const DataParameter param = this->layer_param_.data_param();
  const int batch_size = param.batch_size();
  darknet_version_ = param.darknet_version();
  // Read a data point, and use it to initialize the top blob.
  Datum& datum = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  sides_ = param.side();
  CHECK_EQ(darknet_version_ == 1, sides_ > 0) <<
      "side setting needs darknet version to be v1";
  CHECK_EQ(darknet_version_ == 2, sides_ == 0) <<
      "side setting needs darknet version to be v1";

  // label
  if (this->output_labels_) {
    if (darknet_version_ == 1) { //v1 
      vector<int> label_shape(1, batch_size);
      int label_size = sides_ * sides_ * (1 + 1 + 4);
      label_shape.push_back(label_size);
      top[1]->Reshape(label_shape);

      for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
//        this->prefetch_[i].multi_label_.clear();
//        shared_ptr<Blob<Dtype> > tmp_blob;
//        tmp_blob.reset(new Blob<Dtype>(label_shape));
//        this->prefetch_[i].multi_label_.push_back(tmp_blob);
        this->prefetch_[i].label_.Reshape(label_shape);
      }
      
    } else { //v2
      vector<int> label_shape(1, batch_size);
      label_shape.push_back(30*5);
      top[1]->Reshape(label_shape);
      for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
//        this->prefetch_[i].multi_label_.clear();
//        shared_ptr<Blob<Dtype> > tmp_blob;
 //       tmp_blob.reset(new Blob<Dtype>(label_shape));
 //       this->prefetch_[i].multi_label_.push_back(tmp_blob);
        this->prefetch_[i].label_.Reshape(label_shape);
      }
    }
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void BoxDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  Datum& datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label;

  if (this->output_labels_) {
//    top_label = batch->multi_label_[0]->mutable_cpu_data();   
    top_label = batch->label_.mutable_cpu_data();   
  }

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    Datum& datum = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    vector<BoxLabel> box_labels;
    this->transformed_data_.set_cpu_data(top_data + offset);
    if (this->output_labels_) {
      // rand sample a patch, adjust box labels
      this->data_transformer_->Transform(datum, &(this->transformed_data_), &box_labels);

      if(darknet_version_ == 1) {
        // transform label v1
//          int label_offset = batch->multi_label_[0]->offset(item_id);  
          int label_offset = batch->label_.offset(item_id);  
          transform_label_v1(top_label + label_offset, box_labels, sides_);
        }
      else { // darknet v2
//          int label_offset = batch->multi_label_[0]->offset(item_id);  
          int label_offset = batch->label_.offset(item_id);  
//          LOG(INFO)<<"label_offset: "<<label_offset;
          transform_label_v2(top_label + label_offset, box_labels);
      }
    } else {
      this->data_transformer_->Transform(datum, &(this->transformed_data_));
    }
    trans_time += timer.MicroSeconds();
    reader_.free().push(const_cast<Datum*>(&datum));
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template<typename Dtype>
void BoxDataLayer<Dtype>::transform_label_v1(Dtype* top_label,
    const vector<BoxLabel>& box_labels, int side) {
  int locations = pow(side, 2);
  // isobj
  caffe_set(locations, Dtype(0), top_label);
  // class label
  caffe_set(locations, Dtype(-1), top_label + locations * 1);
  // box
  caffe_set(locations*4, Dtype(0), top_label + locations * 2);
  for (int i = 0; i < box_labels.size(); ++i) {
    float class_label = box_labels[i].class_label_;
    CHECK_GE(class_label, 0) << "class_label must >= 0";
    float x = box_labels[i].box_[0];
    float y = box_labels[i].box_[1];
    // LOG(INFO) << "x: " << x << " y: " << y;
    int x_index = floor(x * side);
    int y_index = floor(y * side);
    x_index = std::min(x_index, side - 1);
    y_index = std::min(y_index, side - 1);
    int obj_index = side * y_index + x_index;
    int class_index = locations + obj_index;
    int cor_index = locations * 2 + obj_index * 4;
    top_label[obj_index] = 1;
    // LOG(INFO) << "dif_index: " << dif_index << " class_label: " << class_label;
    top_label[class_index] = class_label;
    for (int j = 0; j < 4; ++j) {
      top_label[cor_index + j] = box_labels[i].box_[j];
    }
  }
}

template<typename Dtype>
void BoxDataLayer<Dtype>::transform_label_v2(Dtype* top_label, const vector<BoxLabel>& box_labels) {

  caffe_set(30 * 5, Dtype(0), top_label);
//  LOG(INFO)<<"box_labels size: "<<box_labels.size();
  
  for (int i = 0; i < box_labels.size(); ++i) {
    float x = box_labels[i].box_[0];
    float y = box_labels[i].box_[1];
    float w = box_labels[i].box_[2];
    float h = box_labels[i].box_[3];
    float id = box_labels[i].class_label_; 
//    LOG(INFO)<<"X Y W H: "<<x<<" "<<y<<" "<<w<<" "<<h<<" "<<id;

    if ((w < .005 || h < .005)) continue;

    top_label[i*5 + 0] = x;
    top_label[i*5 + 1] = y;
    top_label[i*5 + 2] = w;
    top_label[i*5 + 3] = h;
    top_label[i*5 + 4] = id;
  }
}

INSTANTIATE_CLASS(BoxDataLayer);
REGISTER_LAYER_CLASS(BoxData);

}  // namespace caffe
