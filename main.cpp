//
// Created by yanhao on 17-6-4.
//

#include <stdio.h>
#include <vector>
#include <string>
#include <fstream>
#include <common.h>
#include "common.h"

using namespace std;


const int kWndSize = 40;
int min_face_size_ = 20;
int max_face_size_=-1;
int slide_wnd_step_x_ = 4;
int slide_wnd_step_y_ = 4;
float cls_thresh_ = 3.85f;

int  wnd_size_=40;
//int slide_wnd_step_x_=4;
//int slide_wnd_step_y_=4;

int num_hierarchy_ = 0;


//0

typedef struct LABFeature {
    int x;
    int y;
} LABFeature;




typedef struct FeatureMap {
    FeatureMap() {
        width_ = 0;
        height_ =0;
        roi_.x = 0;
        roi_.y = 0;
        roi_.width = 0;
        roi_.height = 0;
    }

    int width_;
    int height_;
    Rect roi_;
};

typedef struct LABFeatureMap {
    LABFeatureMap(){
        width_ = 0;
        height_ =0;
        roi_.x = 0;
        roi_.y = 0;

        roi_.width = 0;
        roi_.height = 0;
        rect_width_ = 3;
        rect_height_ = 3;
        num_rect_ =3;
    }

    int width_ = 0;
    int height_;
    Rect roi_;

    int rect_width_ ;
    int rect_height_;
    int num_rect_;

    std::vector<unsigned char> feat_map_;
    std::vector<int> rect_sum_;
    std::vector<int> int_img_;
    std::vector<unsigned int> square_int_img_;

};

typedef struct MLPLayer {
    
    MLPLayer()
            : input_dim_(0), output_dim_(0), act_func_type_(1) {}
    int act_func_type_;
    int input_dim_;
    int output_dim_;
    std::vector<float> weights_;
    std::vector<float> bias_;
}MLPLayer;


typedef struct MLP {
    void Compute(const float* input, float* output);
    std::vector<MLPLayer*> layers_;
    std::vector<float> layer_buf_[2];
};


typedef struct LABBaseClassifier {

    LABBaseClassifier()
            : num_bin_(255), thresh_(0.0f) {
        weights_.resize(num_bin_ + 1);
    }
    int num_bin_;
    std::vector<float> weights_;
    float thresh_;
};

typedef struct LABBoostedClassifier{
    const int kFeatGroupSize = 10;
    const float kStdDevThresh = 10.0f;
    std::vector<LABFeature> feat_;
    std::vector<LABBaseClassifier *> base_classifiers_;
    LABFeatureMap* feat_map_;
    bool use_std_dev_;
}LABBoostedClassifier;






typedef struct Impl {
public:
    Impl(){
        kWndSize = 40;
        slide_wnd_step_x_=4;
        slide_wnd_step_y_=4;
        min_face_size_=20;
        max_face_size_=-1;
        cls_thresh_=3.85f;
    }
    int kWndSize;
    int min_face_size_;
    int max_face_size_;
    int slide_wnd_step_x_;
    int slide_wnd_step_y_;
    float cls_thresh_;

    //seeta::fd::Detector *detector_;
}Impl;




typedef struct LABBoostModelReader {
    int num_bin_;
    int num_base_classifer_;
}LABBoostModelReader;

typedef struct SURFMLPModelReader {
    std::vector<int> feat_id_buf_;
    std::vector<float> weights_buf_;
    std::vector<float> bias_buf_;
}SURFMLPModelReader;





//FaceDetection::FaceDetection(const char *model_path)
//        : impl_(new seeta::FaceDetection::Impl()) {
//    impl_->detector_->LoadModel(model_path);
//    impl_->detector_->SetWindowSize(impl_->kWndSize);
//    impl_->detector_->SetSlideWindowStep(impl_->slide_wnd_step_x_,
//                                         impl_->slide_wnd_step_y_);
//    //printf("hahaha!\n");
//}
//
//FaceDetection::~FaceDetection() {
//    if (impl_ != nullptr)
//        delete impl_;
//}
//
//std::vector<seeta::FaceInfo> FaceDetection::Detect(
//        const seeta::ImageData &img) {
//    if (!impl_->IsLegalImage(img))
//        return std::vector<seeta::FaceInfo>();
//
//
//    //printf("mmmmm");
//
//    std::vector<seeta::FaceInfo> pos_wnds_;
//
//
//
//    seeta::fd::ImagePyramid img_pyramid_;
//    int min_img_size = img.height <= img.width ? img.height : img.width;
//    min_img_size = (impl_->max_face_size_ > 0 ?
//                    (min_img_size >= impl_->max_face_size_ ? impl_->max_face_size_ : min_img_size) :
//                    min_img_size);
//
//    img_pyramid_.SetImage1x(img.data, img.width, img.height);
//    img_pyramid_.SetMinScale(static_cast<float>(impl_->kWndSize) / min_img_size);
//
//    impl_->detector_->SetWindowSize(impl_->kWndSize);
//    impl_->detector_->SetSlideWindowStep(impl_->slide_wnd_step_x_,
//                                         impl_->slide_wnd_step_y_);
//
//    pos_wnds_ = impl_->detector_->Detect(&img_pyramid_);
//
//    for (int i = 0; i < pos_wnds_.size(); i++) {
//        if (pos_wnds_[i].score < impl_->cls_thresh_) {
//            pos_wnds_.resize(i);
//            break;
//        }
//    }
//
//
//    return pos_wnds_;
//
//
//
//
//
//
//}





//int num_bin_;
//int num_base_classifer_;
//const int kFeatGroupSize = 10;
//const float kStdDevThresh = 10.0f;
//
//int num_bin_classifer = 255;
//
//std::vector<float> weights_;
//float thresh_=0.0f;





//std::vector<LABFeature> feat_;
//std::vector< LABBaseClassifier*> base_classifiers_;
//seeta::fd::LABFeatureMap* feat_map_;
//bool use_std_dev_;
//
//std::vector<int> hierarchy_size_;
//std::vector<int> num_stage_;
//std::vector<std::vector<int> > wnd_src_id_;






//inline void SetThreshold(float thresh) { thresh_ = thresh; }
//inline int num_bin() const { return num_bin_; }
//inline float weights(int val) const { return weights_[val]; }
//inline float threshold() const { return thresh_; }

//void SetWeights(const float* weights, int num_bin) {
//    weights_.resize(num_bin + 1);
//    num_bin_classifer = num_bin;
//    std::copy(weights, weights + num_bin_ + 1, weights_.begin());
//}
//
//bool ReadFeatureParam0(std::istream* input) {
//    int x;
//    int y;
//    for (int i = 0; i < num_base_classifer_; i++) {
//        input->read((char*)(&x), sizeof(int));
//        input->read((char*)(&y), sizeof(int));
//        //model->AddFeature(x, y);
//        LABFeature feat;
//        feat.x = x;
//        feat.y = y;
//        feat_.push_back(feat);
//
//    }
//
//    return !input->fail();
//}
//
//bool ReadBaseClassifierParam0(std::istream* input) {
//    std::vector<float> thresh;
//    thresh.resize(num_base_classifer_);
//    input->read((char*)(thresh.data()),
//                sizeof(float)* num_base_classifer_);
//
//    int weight_len = sizeof(float)* (num_bin_ + 1);
//    std::vector<float> weights;
//    weights.resize(num_bin_ + 1);
//    for (int i = 0; i < num_base_classifer_; i++) {
//        input->read((char*)(weights.data()), weight_len);
//
//        SetWeights(weights.data(), num_bin_);
//        SetThreshold(thresh[i]);
//
//        model->AddBaseClassifier(weights.data(), num_bin_, thresh[i]);
//    }
//
//    return !input->fail();
//}
//
//
//bool Read_0(std::istream* input,
//                               seeta::fd::Classifier* model) {
//    bool is_read;
////    seeta::fd::LABBoostedClassifier* lab_boosted_classifier =
////            dynamic_cast<seeta::fd::LABBoostedClassifier*>(model);
//
//    int num_bin_;
//    int num_base_classifer_;
//
//    input->read(reinterpret_cast<char*>(&num_base_classifer_), sizeof(int));
//    input->read(reinterpret_cast<char*>(&num_bin_), sizeof(int));
//
//    is_read = (!input->fail()) && num_base_classifer_ > 0 && num_bin_ > 0 &&
//              ReadFeatureParam0(input, lab_boosted_classifier) &&
//              ReadBaseClassifierParam0(input, lab_boosted_classifier);
//
//    return is_read;
//}
//
//



int main(){


    const int kWndSize = 40;
    int min_face_size_ = 20;
    int max_face_size_=-1;
    int slide_wnd_step_x_ = 4;
    int slide_wnd_step_y_ = 4;
    float cls_thresh_ = 3.85f;

    int  wnd_size_=40;
    //int slide_wnd_step_x_=4;
    //int slide_wnd_step_y_=4;

    int num_hierarchy_ = 0;

    std::vector<int> hierarchy_size_;
    std::vector<int> num_stage_;
    std::vector<std::vector<int> > wnd_src_id_;


//    slide_wnd_step_x_(4), slide_wnd_step_y_(4),
//            min_face_size_(20), max_face_size_(-1),
//            //cls_thresh_(0.5f) {
//            cls_thresh_(3.85f) {



    printf("hello,world!\n");
}