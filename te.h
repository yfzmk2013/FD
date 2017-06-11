//
// Created by yanhao on 17-6-10.
//

#ifndef FD_TE_H
#define FD_TE_H


#include <elf.h>

//typedef struct ImageData {
//    ImageData() {
//        data = nullptr;
//        width = 0;
//        height = 0;
//        num_channels = 0;
//    }
//
////    ImageData(int32_t img_width, int32_t img_height,
////              int32_t img_num_channels = 1) {
////        data = nullptr;
////        width = img_width;
////        height = img_height;
////        num_channels = img_num_channels;
////    }
//
//    uint8_t* data;
//    int32_t width;
//    int32_t height;
//    int32_t num_channels;
//} ImageData;


typedef struct ImagePyramid {
    ImagePyramid()
            : max_scale_(1.0f), min_scale_(1.0f),
              scale_factor_(1.0f), scale_step_(0.8f),
              width1x_(0), height1x_(0),
              width_scaled_(0), height_scaled_(0),
              buf_img_width_(2), buf_img_height_(2),
              buf_scaled_width_(2), buf_scaled_height_(2) {
        buf_img_ = new uint8_t[buf_img_width_ * buf_img_height_];
        buf_img_scaled_ = new uint8_t[buf_scaled_width_ * buf_scaled_height_];
    }

//    ~ImagePyramid() {
//        delete[] buf_img_;
//        buf_img_ = nullptr;
//
//        buf_img_width_ = 0;
//        buf_img_height_ = 0;
//
//        delete[] buf_img_scaled_;
//        buf_img_scaled_ = nullptr;
//
//        buf_scaled_width_ = 0;
//        buf_scaled_height_ = 0;
//
//        img_scaled_.data = nullptr;
//        img_scaled_.width = 0;
//        img_scaled_.height = 0;
//    }
//
//    inline void SetScaleStep(float step) {
//        if (step > 0.0f && step <= 1.0f)
//            scale_step_ = step;
//    }
//    inline void SetMinScale(float min_scale) {
//        min_scale_ = min_scale;
//    }
//    inline void SetMaxScale(float max_scale) {
//        max_scale_ = max_scale;
//        scale_factor_ = max_scale;
//        UpdateBufScaled();
//    }
//    void SetImage1x(const uint8_t* img_data, int32_t width, int32_t height);
//    inline float min_scale() const { return min_scale_; }
//    inline float max_scale() const { return max_scale_; }
//    inline ImageData image1x() {
//        ImageData img(width1x_, height1x_, 1);
//        img.data = buf_img_;
//        return img;
//    }
//    ImageData* GetNextScaleImage(float* scale_factor = nullptr);
//    void UpdateBufScaled();

    float max_scale_;
    float min_scale_;

    float scale_factor_;
    float scale_step_;

    int32_t width1x_;
    int32_t height1x_;

    int32_t width_scaled_;
    int32_t height_scaled_;

    uint8_t* buf_img_;
    int32_t buf_img_width_;
    int32_t buf_img_height_;

    uint8_t* buf_img_scaled_;
    int32_t buf_scaled_width_;
    int32_t buf_scaled_height_;

    ImageData img_scaled_;
}ImagePyramid;



#endif //FD_TE_H
