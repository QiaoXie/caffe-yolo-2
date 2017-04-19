#ifndef BOX_H
#define BOX_H
#endif


#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

typedef struct{
    float x, y, w, h;
} box;

typedef struct{
    float dx, dy, dw, dh;
} dbox;

//template <typename Dtype>
box float_to_box(const double *f);

box float_to_box(const float *f);

float box_iou(box a, box b);


float box_rmse(box a, box b);

dbox diou(box a, box b);


void do_nms(box *boxes, float **probs, int total, int classes, float thresh);


void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh);


void do_nms_obj(box *boxes, float **probs, int total, int classes, float thresh);


box decode_box(box b, box anchor);


box encode_box(box b, box anchor);

}