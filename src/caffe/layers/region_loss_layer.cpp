#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>


#include "caffe/layers/detection_loss_layer.hpp"
#include "caffe/layers/region_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/box.hpp"

namespace caffe {

static int data_seen = 0;

template <typename Dtype>
static inline Dtype logistic_activate(Dtype x){return 1./(1. + exp(-x));}

template <typename Dtype>
static inline Dtype logistic_gradient(Dtype x){return (1-x)*x;}


template <typename Dtype>
void flatten(Dtype *x, int size, int layers, int batch, int forward)
{
    Dtype *swap = (Dtype*) calloc(size*layers*batch, sizeof(Dtype));
    int i,c,b;
    for(b = 0; b < batch; ++b){
        for(c = 0; c < layers; ++c){
            for(i = 0; i < size; ++i){
                int i1 = b*layers*size + c*size + i;
                int i2 = b*layers*size + i*layers + c;
                if (forward) swap[i2] = x[i1];
                else swap[i1] = x[i2];
            }
        }
    }
    memcpy(x, swap, size*layers*batch*sizeof(Dtype));
    free(swap);
}

template <typename Dtype>
void softmax(Dtype *input, int n, Dtype temp, Dtype *output)
{
    int i;
    Dtype sum = 0;
    Dtype largest = -FLT_MAX;
    for(i = 0; i < n; ++i){
        if(input[i] > largest) largest = input[i];
    }
    for(i = 0; i < n; ++i){
        Dtype e = exp(input[i]/temp - largest/temp);
        sum += e;
        output[i] = e;
    }
    for(i = 0; i < n; ++i){
        output[i] /= sum;
    }
}

template <typename Dtype>
box get_region_box(Dtype *x, Dtype *biases, int n, int index, int i, int j, int w, int h)
{
    box b;
    b.x = (i + logistic_activate(x[index + 0])) / w;
    b.y = (j + logistic_activate(x[index + 1])) / h;
    b.w = exp(x[index + 2]) * biases[2*n]   / w;
    b.h = exp(x[index + 3]) * biases[2*n+1] / h;
    return b;
}

template <typename Dtype>
float delta_region_box(box truth, Dtype *x, Dtype *biases, int n, int index, int i, int j, int w, int h, Dtype *delta, float scale, Dtype &coord_loss, Dtype &area_loss)
{
    box pred = get_region_box(x, biases, n, index, i, j, w, h);
    float iou = box_iou(pred, truth);

    Dtype tx = (truth.x*w - i);
    Dtype ty = (truth.y*h - j);
    Dtype tw = log(truth.w*w / biases[2*n]);
    Dtype th = log(truth.h*h / biases[2*n + 1]);

    delta[index + 0] = (-1) * scale * (tx - logistic_activate(x[index + 0])) * logistic_gradient(logistic_activate(x[index + 0]));
    delta[index + 1] = (-1) * scale * (ty - logistic_activate(x[index + 1])) * logistic_gradient(logistic_activate(x[index + 1]));
    delta[index + 2] = (-1) * scale * (tw - x[index + 2]);
    delta[index + 3] = (-1) * scale * (th - x[index + 3]);


//    std::cout<<"delta coord: "<<delta[index + 0]<<" "<<delta[index + 1]<<" "<<delta[index + 2]<<" "<<delta[index + 3]<<std::endl;

    coord_loss += scale * (pow((tx-logistic_activate(x[index + 0])), 2) + pow((ty - logistic_activate(x[index + 1])), 2));
    area_loss += scale * (pow((tw - x[index + 2]), 2) + pow((th - x[index + 3]), 2));
    return iou;
}

template <typename Dtype>
void delta_region_class(Dtype *output, Dtype *delta, int index, int class_ind, int classes, float scale, Dtype &avg_cat, Dtype &class_loss)
{
    int n;
    
    for(n = 0; n < classes; ++n){
        delta[index + n] = (-1) * scale * (((n == class_ind)?1 : 0) - output[index + n]);
        class_loss += scale * pow((((n == class_ind)?1 : 0) - output[index + n]), 2);
        if(n == class_ind) avg_cat += output[index + n];

    }
    
}

template <typename Dtype>
void RegionLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  RegionLossParameter param = this->layer_param_.region_loss_param();

  w_ = bottom[0]->width();
  h_ = bottom[0]->height();
  n_ = param.num_object();
  coords_ = param.num_coord();
  classes_ = param.num_class();

  object_scale_ = param.object_scale();
  noobject_scale_ = param.noobject_scale();
  class_scale_ = param.class_scale();
  coord_scale_ = param.coord_scale();

  softmax_ = param.softmax();
  rescore_ = param.rescore();

  thresh_ = param.thresh();
  bias_match_ = param.bias_match();

  int anchor_x_size = param.anchor_x_size();
  int anchor_y_size = param.anchor_y_size();

  CHECK_EQ(anchor_x_size, anchor_y_size);
  CHECK_EQ(anchor_x_size, n_);
  
  vector<int> bias_shape;
  bias_shape.push_back(2*n_);
  biases_.Reshape(bias_shape);

  Dtype* l_biases = biases_.mutable_cpu_data();

  caffe_set(n_ * 2, Dtype(0.5), l_biases);

  for(int i=0; i<n_; i++)
  {
      l_biases[2*i + 0] = param.anchor_x(i);
      l_biases[2*i + 1] = param.anchor_y(i);
  }

  batch_ = bottom[0]->num();
  outputs_ = h_*w_*n_*(classes_ + coords_ + 1);
  inputs_ = outputs_;
  truths_ = 30*(5);
  output_.ReshapeLike(*bottom[0]);

  CHECK_EQ(outputs_, bottom[0]->count(1));
  CHECK_EQ(truths_, bottom[1]->count(1));

}

template <typename Dtype>
void RegionLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  output_.ReshapeLike(*bottom[0]);
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void RegionLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  data_seen += batch_;
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  int size = coords_ + classes_ + 1;

  Dtype* diff = diff_.mutable_cpu_data();
  Dtype* l_biases = biases_.mutable_cpu_data();

  caffe_set(diff_.count(), Dtype(0.), diff);

  Dtype* l_output = output_.mutable_cpu_data();
  caffe_copy(diff_.count(), input_data, l_output); 
  flatten(l_output, w_ * h_, n_*size, batch_, 1);

  Dtype loss(0.0), class_loss(0.0), noobj_loss(0.0), obj_loss(0.0), coord_loss(0.0), area_loss(0.0);
  Dtype avg_iou(0.0), recall(0.0), avg_cat(0.0), avg_obj(0.0), avg_anyobj(0.0);
  Dtype obj_count(0), class_count(0);

  int i,j,b,t,n;

  for (b = 0; b < batch_; ++b){
    for(i = 0; i < h_*w_*n_; ++i){
        int index = size *i + b*outputs_;
        l_output[index + 4] = logistic_activate(l_output[index + 4]);
    }
  }

  if (softmax_){
    for (b = 0; b < batch_; ++b){
      for(i = 0; i < h_*w_*n_; ++i){
        int index = size*i + b*outputs_;
        softmax(l_output + index + 5, classes_, (Dtype)1.0, l_output + index + 5);
      }
    }
  }

  for (b = 0; b < batch_; ++b) {
    for (j = 0; j < h_; ++j) {
      for (i = 0; i < w_; ++i) {
        for (n = 0; n < n_; ++n) {
          int index = size*(j*w_*n_ + i*n_ + n) + b*outputs_;
          
          box pred = get_region_box(l_output, l_biases, n, index, i, j, w_, h_);
          float best_iou = 0;
          for(t = 0; t < 30; ++t){
            box truth = float_to_box(label_data+ t*5 + b*truths_);
            if(!truth.x) break;
            float iou = box_iou(pred, truth);
            if (iou > best_iou) {
                best_iou = iou;
            }
          }
          avg_anyobj += l_output[index + 4];
          
          if (best_iou > thresh_) {
            diff[index + 4] = 0;
          }
          else {
            noobj_loss += noobject_scale_ * pow(l_output[index + 4], 2);
            diff[index + 4] =  (-1) * noobject_scale_ * ((0 - l_output[index + 4]) * logistic_gradient(l_output[index + 4]));
          }

          if(data_seen < 12800){
            box truth = {0};
            truth.x = (i + .5)/w_;
            truth.y = (j + .5)/h_;
            truth.w = l_biases[2*n]/w_;
            truth.h = l_biases[2*n+1]/h_;
            delta_region_box(truth, l_output, l_biases, n, index, i, j, w_, h_, diff, .01, coord_loss, area_loss);
          }
          
        }
      }
    }
    for(t = 0; t < 30; ++t) {
      box truth = float_to_box(label_data+ t*5 + b*truths_);
      if(!truth.x) break;
      float best_iou = 0;
      int best_index = 0;
      int best_n = 0;
      i = (truth.x * w_);
      j = (truth.y * h_);
      box truth_shift = truth;
      truth_shift.x = 0;
      truth_shift.y = 0;
      for(n = 0; n < n_; ++n) {
          int index = size*(j*w_*n_ + i*n_ + n) + b*outputs_;
          box pred = get_region_box(l_output, l_biases, n, index, i, j, w_, h_);
          if(bias_match_){
              pred.w = l_biases[2*n]/w_;
              pred.h = l_biases[2*n+1]/h_;
          }
          pred.x = 0;
          pred.y = 0;
          float iou = box_iou(pred, truth_shift);
          if (iou > best_iou){
              best_index = index;
              best_iou = iou;
              best_n = n;
          }
      }

      float iou = delta_region_box(truth, l_output, l_biases, best_n, best_index, i, j, w_, h_, diff, coord_scale_, coord_loss, area_loss);
      if(iou > .5) recall += 1;
      avg_iou += iou;

      avg_obj += l_output[best_index + 4];

      if (rescore_) {
        obj_loss += object_scale_ * pow(iou - l_output[best_index + 4], 2);
        diff[best_index + 4] = (-1) * object_scale_ * (iou - l_output[best_index + 4]) * logistic_gradient(l_output[best_index + 4]);
      }
      else {
        obj_loss += object_scale_ * pow(1 - l_output[best_index + 4], 2);
        diff[best_index + 4] = (-1) * object_scale_ * (1 - l_output[best_index + 4]) * logistic_gradient(l_output[best_index + 4]);
      }

      int class_ind = label_data[t*5 + b*truths_ + 4];
      delta_region_class(l_output, diff, best_index + 5, class_ind, classes_, class_scale_, avg_cat, class_loss);

      obj_count += 1;
      class_count += 1;
    }


   
  }
  
  flatten(diff, w_ * h_, n_*size, batch_, 0);

  obj_count += 0.01;
  class_count += 0.01;

  class_loss /= class_count;
  coord_loss /= obj_count;
  area_loss /= obj_count;
  obj_loss /= obj_count;
  noobj_loss /= (w_ * h_ * n_ * batch_ - obj_count);

  loss = class_loss + coord_loss + area_loss + obj_loss + noobj_loss;
  top[0]->mutable_cpu_data()[0] = loss;

  avg_iou /= obj_count;
  avg_cat /= class_count;
  avg_obj /= obj_count;
  avg_anyobj /= (w_*h_*n_*batch_ - obj_count);
  recall /= obj_count;

  LOG(INFO) << "loss: " << loss << " class_loss: " << class_loss << " obj_loss: " 
        << obj_loss << " noobj_loss: " << noobj_loss << " coord_loss: " << coord_loss
        << " area_loss: " << area_loss;
  LOG(INFO) << "avg_iou: " << avg_iou << " Class: " << avg_cat << " Obj: "
        << avg_obj << " No Obj: " << avg_anyobj << " Avg Recall: " << recall << " count: "<<(int)(obj_count);

}

template <typename Dtype>
void RegionLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype sign(1.);
    const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[0]->num();

    caffe_cpu_axpby(
        bottom[0]->count(),
        alpha,
        diff_.cpu_data(),
        Dtype(0),
        bottom[0]->mutable_cpu_diff());
  }
}

INSTANTIATE_CLASS(RegionLossLayer);
REGISTER_LAYER_CLASS(RegionLoss);

}  // namespace caffe
