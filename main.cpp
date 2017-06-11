//
// Created by yanhao on 17-6-4.
//

#include <stdio.h>
#include <vector>
#include <string>
#include <fstream>
#include <string.h>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cstring>
#include "common.h"
#include "te.h"
#include "math_func.h"

using namespace std;


const int kWndSize = 40;
const int min_face_size_ = 20;
const int max_face_size_ = -1;
const int slide_wnd_step_x_ = 4;
const int slide_wnd_step_y_ = 4;
const float cls_thresh_ = 3.85f;
const int wnd_size_ = 40;









//0

typedef struct LABFeature {
    int x;
    int y;
} LABFeature;


typedef struct FeatureMap {
    FeatureMap() {
        width_ = 0;
        height_ = 0;
        roi_.x = 0;
        roi_.y = 0;
        roi_.width = 0;
        roi_.height = 0;
    }

    int width_;
    int height_;
    Rect roi_;
} FeatureMap;

typedef struct LABFeatureMap {
    LABFeatureMap() {
        width_ = 0;
        height_ = 0;
        roi_.x = 0;
        roi_.y = 0;

        roi_.width = 0;
        roi_.height = 0;
        rect_width_ = 3;
        rect_height_ = 3;
        num_rect_ = 3;
    }

    int width_ = 0;
    int height_;
    Rect roi_;

    int rect_width_;
    int rect_height_;
    int num_rect_;

    std::vector<unsigned char> feat_map_;
    std::vector<int> rect_sum_;
    std::vector<int> int_img_;
    std::vector<unsigned int> square_int_img_;

} LABFeatureMap;


typedef struct LABBaseClassifier {

    LABBaseClassifier()
            : num_bin_(255), thresh_(0.0f) {
        weights_.resize(num_bin_ + 1);
    }

    int num_bin_;
    std::vector<float> weights_;
    float thresh_;
} LABBaseClassifier;

typedef struct _LABBoostedClassifier {
    const int kFeatGroupSize = 10;
    const float kStdDevThresh = 10.0f;
    std::vector<LABFeature> feat_;
    std::vector<LABBaseClassifier *> base_classifiers_;
    LABFeatureMap *feat_map_;
    bool use_std_dev_;
} LABBoostedClassifier;


typedef struct SURFFeature {
    Rect patch;
    int32_t num_cell_per_row;
    int32_t num_cell_per_col;
} SURFFeature;
typedef struct SURFFeaturePool {


    SURFFeaturePool()
            : sample_width_(40), sample_height_(40),
              patch_move_step_x_(16), patch_move_step_y_(16), patch_size_inc_step_(1),
              patch_min_width_(16), patch_min_height_(16){}

    typedef struct SURFPatchFormat {
        /**< aspect ratio, s.t. GCD(width, height) = 1 */
        int32_t width;
        int32_t height;

        /**< cell partition */
        int32_t num_cell_per_row;
        int32_t num_cell_per_col;
    } SURFPatchFormat;

    int32_t sample_width_;
    int32_t sample_height_;
    int32_t patch_move_step_x_;
    int32_t patch_move_step_y_;
    int32_t patch_size_inc_step_; /**< incremental step of patch width and */
    /**< height when build feature pool      */
    int32_t patch_min_width_;
    int32_t patch_min_height_;

    std::vector<SURFFeature> pool_;
    std::vector<SURFPatchFormat> format_;
} SURFFeaturePool;


typedef struct SURFFeatureMap {

    SURFFeatureMap()
            : width_(0), height_(0) {
        roi_.x = 0;
        roi_.y = 0;
        roi_.width = 0;
        roi_.height = 0;
    }


    int32_t width_;
    int32_t height_;

    Rect roi_;


    const int32_t kNumIntChannel = 8;
    bool buf_valid_reset_;
    std::vector<int32_t> grad_x_;
    std::vector<int32_t> grad_y_;
    std::vector<int32_t> int_img_;
    std::vector<int32_t> img_buf_;
    std::vector<std::vector<int32_t> > feat_vec_buf_;
    std::vector<std::vector<float> > feat_vec_normed_buf_;
    std::vector<int32_t> buf_valid_;


    SURFFeaturePool feat_pool_;
} SURFFeatureMap;


typedef struct MLPLayer {

    MLPLayer(int32_t act_func_type = 1)
            : input_dim_(0), output_dim_(0), act_func_type_(1) {}

    int act_func_type_;
    int input_dim_;
    int output_dim_;
    std::vector<float> weights_;
    std::vector<float> bias_;
} MLPLayer;


typedef struct MLP {
    std::vector<MLPLayer *> layers_;
    std::vector<float> layer_buf_[2];
} MLP;

typedef struct SURFMLP {

    int32_t width_;
    int32_t height_;

    Rect roi_;

    std::vector<int32_t> feat_id_;
    std::vector<float> input_buf_;
    std::vector<float> output_buf_;

    MLP model_;
    float thresh_;
    SURFFeatureMap *feat_map_;
} SURFMLP;


typedef struct Impl {
public:
    Impl() {
        kWndSize = 40;
        slide_wnd_step_x_ = 4;
        slide_wnd_step_y_ = 4;
        min_face_size_ = 20;
        max_face_size_ = -1;
        cls_thresh_ = 3.85f;
    }

    int kWndSize;
    int min_face_size_;
    int max_face_size_;
    int slide_wnd_step_x_;
    int slide_wnd_step_y_;
    float cls_thresh_;

    //seeta::fd::Detector *detector_;
} Impl;


typedef struct LABBoostModelReader {
    int num_bin_;
    int num_base_classifer_;
} LABBoostModelReader;

typedef struct SURFMLPModelReader {
    std::vector<int> feat_id_buf_;
    std::vector<float> weights_buf_;
    std::vector<float> bias_buf_;
} SURFMLPModelReader;


typedef struct FuStDetector {
    FuStDetector()
            : wnd_size_(40), slide_wnd_step_x_(4), slide_wnd_step_y_(4),
              num_hierarchy_(0) {
        wnd_data_buf_.resize(wnd_size_ * wnd_size_);
        wnd_data_.resize(wnd_size_ * wnd_size_);
    }


    int32_t wnd_size_;
    int32_t slide_wnd_step_x_;
    int32_t slide_wnd_step_y_;

    int32_t num_hierarchy_;
    std::vector<int32_t> hierarchy_size_;
    std::vector<int32_t> num_stage_;
    std::vector<std::vector<int32_t> > wnd_src_id_;

    std::vector<uint8_t> wnd_data_buf_;
    std::vector<uint8_t> wnd_data_;

    //std::vector<std::shared_ptr<seeta::fd::Classifier> > model_;
    //std::vector<std::shared_ptr<seeta::fd::FeatureMap> > feat_map_;
    //std::map<seeta::fd::ClassifierType, int32_t> cls2feat_idx_;


} FuStDetector;

static void ResizeImage(ImageData &src, ImageData *dest) {
    int32_t src_width = src.width;
    int32_t src_height = src.height;
    int32_t dest_width = dest->width;
    int32_t dest_height = dest->height;
    if (src_width == dest_width && src_height == dest_height) {
        memcpy(dest->data, src.data, src_width * src_height * sizeof(uint8_t));
        return;
    }

    double lf_x_scl = static_cast<double>(src_width) / dest_width;
    double lf_y_Scl = static_cast<double>(src_height) / dest_height;
    const uint8_t *src_data = src.data;
    uint8_t *dest_data = dest->data;
    {

        for (int32_t y = 0; y < dest_height; y++) {
            for (int32_t x = 0; x < dest_width; x++) {
                double lf_x_s = lf_x_scl * x;
                double lf_y_s = lf_y_Scl * y;

                int32_t n_x_s = static_cast<int>(lf_x_s);
                n_x_s = (n_x_s <= (src_width - 2) ? n_x_s : (src_width - 2));
                int32_t n_y_s = static_cast<int>(lf_y_s);
                n_y_s = (n_y_s <= (src_height - 2) ? n_y_s : (src_height - 2));

                double lf_weight_x = lf_x_s - n_x_s;
                double lf_weight_y = lf_y_s - n_y_s;

                double dest_val = (1 - lf_weight_y) * ((1 - lf_weight_x) *
                                                       src_data[n_y_s * src_width + n_x_s] +
                                                       lf_weight_x * src_data[n_y_s * src_width + n_x_s + 1]) +
                                  lf_weight_y * ((1 - lf_weight_x) * src_data[(n_y_s + 1) * src_width + n_x_s] +
                                                 lf_weight_x * src_data[(n_y_s + 1) * src_width + n_x_s + 1]);

                dest_data[y * dest_width + x] = static_cast<uint8_t>(dest_val);
            }
        }
    }
}


void initImageData(ImageData &src, int32_t img_width, int32_t img_height,
                   int32_t img_num_channels = 1) {
    src.data = 0;
    src.width = img_width;
    src.height = img_height;
    src.num_channels = img_num_channels;
}

//int32_t width_ = 0;
//int32_t height_ = 0;
//Rect roi_;
//
//
//const int32_t rect_width_ = 3;
//const int32_t rect_height_ = 3;
//const int32_t num_rect_ = 3;
//
//std::vector<uint8_t> feat_map_;
//std::vector<int32_t> rect_sum_;
//std::vector<int32_t> int_img_;
//std::vector<uint32_t> square_int_img_;

std::vector<LABBoostedClassifier> lABBoostedClassifierS;
std::vector<SURFMLP> sURFMLPS;


inline void Integral(LABFeatureMap *feat_map_, int32_t *data) {
    const int32_t *src = data;
    int32_t *dest = data;
    const int32_t *dest_above = dest;
    *dest = *(src++);
    for (int32_t c = 1; c < feat_map_->width_; c++, src++, dest++)
        *(dest + 1) = (*dest) + (*src);
    dest++;
    for (int32_t r = 1; r < feat_map_->height_; r++) {
        for (int32_t c = 0, s = 0; c < feat_map_->width_; c++, src++, dest++, dest_above++) {
            s += (*src);
            *dest = *dest_above + s;
        }
    }
}

inline void Integral(LABFeatureMap *feat_map_, uint32_t *data) {
    const uint32_t *src = data;
    uint32_t *dest = data;
    const uint32_t *dest_above = dest;

    *dest = *(src++);
    for (int32_t c = 1; c < feat_map_->width_; c++, src++, dest++)
        *(dest + 1) = (*dest) + (*src);
    dest++;
    for (int32_t r = 1; r < feat_map_->height_; r++) {
        for (int32_t c = 0, s = 0; c < feat_map_->width_; c++, src++, dest++, dest_above++) {
            s += (*src);
            *dest = *dest_above + s;
        }
    }
}


void Reshape(LABFeatureMap *feat_map_, int32_t width, int32_t height) {
    feat_map_->width_ = width;
    feat_map_->height_ = height;

    int32_t len = feat_map_->width_ * feat_map_->height_;
    feat_map_->feat_map_.resize(len);
    feat_map_->rect_sum_.resize(len);
    feat_map_->int_img_.resize(len);
    feat_map_->square_int_img_.resize(len);
}

void ComputeIntegralImages(LABFeatureMap *feat_map_, const uint8_t *input) {
    int32_t len = feat_map_->width_ * feat_map_->height_;

    UInt8ToInt32(input, feat_map_->int_img_.data(), len);
    Square(feat_map_->int_img_.data(), feat_map_->square_int_img_.data(), len);
    Integral(feat_map_, feat_map_->int_img_.data());
    Integral(feat_map_, feat_map_->square_int_img_.data());
}

void ComputeRectSum(LABFeatureMap *feat_map_) {
    int32_t width = feat_map_->width_ - feat_map_->rect_width_;
    int32_t height = feat_map_->height_ - feat_map_->rect_height_;
    const int32_t *int_img = feat_map_->int_img_.data();
    int32_t *rect_sum = feat_map_->rect_sum_.data();

    *rect_sum = *(int_img + (feat_map_->rect_height_ - 1) * feat_map_->width_ + feat_map_->rect_width_ - 1);
    VectorSub(int_img + (feat_map_->rect_height_ - 1) * feat_map_->width_ +
              feat_map_->rect_width_, int_img + (feat_map_->rect_height_ - 1) * feat_map_->width_, rect_sum + 1, width);

    {

        for (int32_t i = 1; i <= height; i++) {
            const int32_t *top_left = int_img + (i - 1) * feat_map_->width_;
            const int32_t *top_right = top_left + feat_map_->rect_width_ - 1;
            const int32_t *bottom_left = top_left + feat_map_->rect_height_ * feat_map_->width_;
            const int32_t *bottom_right = bottom_left + feat_map_->rect_width_ - 1;
            int32_t *dest = rect_sum + i * feat_map_->width_;

            *(dest++) = (*bottom_right) - (*top_right);
            VectorSub(bottom_right + 1, top_right + 1, dest, width);
            VectorSub(dest, bottom_left, dest, width);
            VectorAdd(dest, top_left, dest, width);
        }
    }
}

void ComputeFeatureMap(LABFeatureMap *feat_map_) {
    int32_t width = feat_map_->width_ - feat_map_->rect_width_ * feat_map_->num_rect_;
    int32_t height = feat_map_->height_ - feat_map_->rect_height_ * feat_map_->num_rect_;
    int32_t offset = feat_map_->width_ * feat_map_->rect_height_;
    uint8_t *feat_map = feat_map_->feat_map_.data();
    {

        for (int32_t r = 0; r <= height; r++) {
            for (int32_t c = 0; c <= width; c++) {
                uint8_t *dest = feat_map + r * feat_map_->width_ + c;
                *dest = 0;

                int32_t white_rect_sum = feat_map_->rect_sum_[(r + feat_map_->rect_height_) * feat_map_->width_ + c +
                                                              feat_map_->rect_width_];
                int32_t black_rect_idx = r * feat_map_->width_ + c;
                *dest |= (white_rect_sum >= feat_map_->rect_sum_[black_rect_idx] ? 0x80 : 0x0);
                black_rect_idx += feat_map_->rect_width_;
                *dest |= (white_rect_sum >= feat_map_->rect_sum_[black_rect_idx] ? 0x40 : 0x0);
                black_rect_idx += feat_map_->rect_width_;
                *dest |= (white_rect_sum >= feat_map_->rect_sum_[black_rect_idx] ? 0x20 : 0x0);
                black_rect_idx += offset;
                *dest |= (white_rect_sum >= feat_map_->rect_sum_[black_rect_idx] ? 0x08 : 0x0);
                black_rect_idx += offset;
                *dest |= (white_rect_sum >= feat_map_->rect_sum_[black_rect_idx] ? 0x01 : 0x0);
                black_rect_idx -= feat_map_->rect_width_;
                *dest |= (white_rect_sum >= feat_map_->rect_sum_[black_rect_idx] ? 0x02 : 0x0);
                black_rect_idx -= feat_map_->rect_width_;
                *dest |= (white_rect_sum >= feat_map_->rect_sum_[black_rect_idx] ? 0x04 : 0x0);
                black_rect_idx -= offset;
                *dest |= (white_rect_sum >= feat_map_->rect_sum_[black_rect_idx] ? 0x10 : 0x0);
            }
        }
    }
}

void Compute(LABFeatureMap *feat_map_, const uint8_t *input, int32_t width,
             int32_t height) {
    if (input == 0 || width <= 0 || height <= 0) {
        return;  // @todo handle the errors!!!
    }

    Reshape(feat_map_, width, height);
    ComputeIntegralImages(feat_map_, input);
    ComputeRectSum(feat_map_);
    ComputeFeatureMap(feat_map_);
}


const ImageData *GetNextScaleImage(ImagePyramid *img_pyramid, float *scale_factor) {
    if (img_pyramid->scale_factor_ >= img_pyramid->min_scale_) {
        if (scale_factor != 0)
            *scale_factor = img_pyramid->scale_factor_;

        img_pyramid->width_scaled_ = (int32_t) (img_pyramid->width1x_ * img_pyramid->scale_factor_);
        img_pyramid->height_scaled_ = static_cast<int32_t>(img_pyramid->height1x_ * img_pyramid->scale_factor_);

        ImageData src_img;
        initImageData(src_img, img_pyramid->width1x_, img_pyramid->height1x_);
        ImageData dest_img;
        initImageData(dest_img, img_pyramid->width_scaled_, img_pyramid->height_scaled_);
        src_img.data = img_pyramid->buf_img_;
        dest_img.data = img_pyramid->buf_img_scaled_;
        ResizeImage(src_img, &dest_img);
        img_pyramid->scale_factor_ *= img_pyramid->scale_step_;

        img_pyramid->img_scaled_.data = img_pyramid->buf_img_scaled_;
        img_pyramid->img_scaled_.width = img_pyramid->width_scaled_;
        img_pyramid->img_scaled_.height = img_pyramid->height_scaled_;
        return &img_pyramid->img_scaled_;
    } else {
        return 0;
    }
}

//enum ClassifierType {
//    LAB_Boosted_Classifier,
//    SURF_MLP
//};


inline void SetROI(LABFeatureMap *feat_map_, Rect &roi) {
    feat_map_->roi_ = roi;
}


const ImageData *GetNextScaleImage(float *scale_factor, ImagePyramid *img_pyramid) {
    if (img_pyramid->scale_factor_ >= img_pyramid->min_scale_) {
        if (scale_factor != 0)
            *scale_factor = img_pyramid->scale_factor_;

        img_pyramid->width_scaled_ = static_cast<int32_t>(img_pyramid->width1x_ * img_pyramid->scale_factor_);
        img_pyramid->height_scaled_ = static_cast<int32_t>(img_pyramid->height1x_ * img_pyramid->scale_factor_);

        ImageData src_img(img_pyramid->width1x_, img_pyramid->height1x_);
        ImageData dest_img(img_pyramid->width_scaled_, img_pyramid->height_scaled_);
        src_img.data = img_pyramid->buf_img_;
        dest_img.data = img_pyramid->buf_img_scaled_;
        ResizeImage(src_img, &dest_img);
        img_pyramid->scale_factor_ *= img_pyramid->scale_step_;

        img_pyramid->img_scaled_.data = img_pyramid->buf_img_scaled_;
        img_pyramid->img_scaled_.width = img_pyramid->width_scaled_;
        img_pyramid->img_scaled_.height = img_pyramid->height_scaled_;
        return &img_pyramid->img_scaled_;
    } else {
        return 0;
    }
}


inline uint8_t GetFeatureVal(LABFeatureMap *feat_map_, int32_t offset_x, int32_t offset_y) {
    return feat_map_->feat_map_[(feat_map_->roi_.y + offset_y) * feat_map_->width_ + feat_map_->roi_.x + offset_x];
}

float GetStdDev(LABFeatureMap *labFeatureMap) {
    double mean;
    double m2;
    Rect roi_;
    roi_.x = labFeatureMap->roi_.x;
    roi_.y = labFeatureMap->roi_.y;
    roi_.width = labFeatureMap->roi_.width;
    roi_.height = labFeatureMap->roi_.height;
    double area = roi_.width * roi_.height;

    int32_t top_left;
    int32_t top_right;
    int32_t bottom_left;
    int32_t bottom_right;


    if (roi_.x != 0) {
        if (roi_.y != 0) {
            top_left = (roi_.y - 1) * labFeatureMap->width_ + roi_.x - 1;
            top_right = top_left + roi_.width;
            bottom_left = top_left + roi_.height * labFeatureMap->width_;
            bottom_right = bottom_left + roi_.width;

            mean = (labFeatureMap->int_img_[bottom_right] - labFeatureMap->int_img_[bottom_left] +
                    labFeatureMap->int_img_[top_left] - labFeatureMap->int_img_[top_right]) / area;
            m2 = (labFeatureMap->square_int_img_[bottom_right] - labFeatureMap->square_int_img_[bottom_left] +
                  labFeatureMap->square_int_img_[top_left] - labFeatureMap->square_int_img_[top_right]) / area;
        } else {
            bottom_left = (roi_.height - 1) * labFeatureMap->width_ + roi_.x - 1;
            bottom_right = bottom_left + roi_.width;

            mean = (labFeatureMap->int_img_[bottom_right] - labFeatureMap->int_img_[bottom_left]) / area;
            m2 = (labFeatureMap->square_int_img_[bottom_right] - labFeatureMap->square_int_img_[bottom_left]) / area;
        }
    } else {
        if (roi_.y != 0) {
            top_right = (roi_.y - 1) * labFeatureMap->width_ + roi_.width - 1;
            bottom_right = top_right + roi_.height * labFeatureMap->width_;

            mean = (labFeatureMap->int_img_[bottom_right] - labFeatureMap->int_img_[top_right]) / area;
            m2 = (labFeatureMap->square_int_img_[bottom_right] - labFeatureMap->square_int_img_[top_right]) / area;
        } else {
            bottom_right = (roi_.height - 1) * labFeatureMap->width_ + roi_.width - 1;
            mean = labFeatureMap->int_img_[bottom_right] / area;
            m2 = labFeatureMap->square_int_img_[bottom_right] / area;
        }
    }

    return static_cast<float>(std::sqrt(m2 - mean * mean));
}

bool Classify0(float *score, float *outputs, LABBoostedClassifier &labBoostedClassifier) {
    bool isPos = true;
    float s = 0.0f;

    for (size_t i = 0; isPos && i < labBoostedClassifier.base_classifiers_.size();) {
        for (int32_t j = 0; j < labBoostedClassifier.kFeatGroupSize; j++, i++) {
            uint8_t featVal = GetFeatureVal(labBoostedClassifier.feat_map_, labBoostedClassifier.feat_[i].x,
                                            labBoostedClassifier.feat_[i].y);
            s += labBoostedClassifier.base_classifiers_[i]->weights_[featVal];
        }
        if (s < labBoostedClassifier.base_classifiers_[i - 1]->thresh_)
            isPos = false;
    }
    isPos = isPos && ((!labBoostedClassifier.use_std_dev_) ||
                      GetStdDev(labBoostedClassifier.feat_map_) > labBoostedClassifier.kStdDevThresh);

    if (score != nullptr)
        *score = s;
    if (outputs != nullptr)
        *outputs = s;

    return isPos;
}

bool CompareBBox(const FaceInfo &a, const FaceInfo &b) {
    return a.score > b.score;
}

void NonMaximumSuppression(std::vector<FaceInfo> *bboxes,
                           std::vector<FaceInfo> *bboxes_nms, float iou_thresh) {
    bboxes_nms->clear();
    std::sort(bboxes->begin(), bboxes->end(), CompareBBox);

    int32_t select_idx = 0;
    int32_t num_bbox = static_cast<int32_t>(bboxes->size());
    std::vector<int32_t> mask_merged(num_bbox, 0);
    bool all_merged = false;

    while (!all_merged) {
        while (select_idx < num_bbox && mask_merged[select_idx] == 1)
            select_idx++;
        if (select_idx == num_bbox) {
            all_merged = true;
            continue;
        }

        bboxes_nms->push_back((*bboxes)[select_idx]);
        mask_merged[select_idx] = 1;

        Rect select_bbox = (*bboxes)[select_idx].bbox;
        float area1 = static_cast<float>(select_bbox.width * select_bbox.height);
        float x1 = static_cast<float>(select_bbox.x);
        float y1 = static_cast<float>(select_bbox.y);
        float x2 = static_cast<float>(select_bbox.x + select_bbox.width - 1);
        float y2 = static_cast<float>(select_bbox.y + select_bbox.height - 1);

        select_idx++;
        for (int32_t i = select_idx; i < num_bbox; i++) {
            if (mask_merged[i] == 1)
                continue;

            Rect &bbox_i = (*bboxes)[i].bbox;
            float x = std::max<float>(x1, static_cast<float>(bbox_i.x));
            float y = std::max<float>(y1, static_cast<float>(bbox_i.y));
            float w = std::min<float>(x2, static_cast<float>(bbox_i.x + bbox_i.width - 1)) - x + 1;
            float h = std::min<float>(y2, static_cast<float>(bbox_i.y + bbox_i.height - 1)) - y + 1;
            if (w <= 0 || h <= 0)
                continue;

            float area2 = static_cast<float>(bbox_i.width * bbox_i.height);
            float area_intersect = w * h;
            float area_union = area1 + area2 - area_intersect;
            if (static_cast<float>(area_intersect) / area_union > iou_thresh) {
                mask_merged[i] = 1;
                bboxes_nms->back().score += (*bboxes)[i].score;
            }
        }
    }
}


inline ImageData iimage1x(ImagePyramid *img_pyramid) {
    ImageData img(img_pyramid->width1x_, img_pyramid->height1x_, 1);
    img.data = img_pyramid->buf_img_;
    return img;
}


void GetWindowData(FuStDetector *fuStDetector, const ImageData &img,
                   const Rect &wnd) {
    int32_t pad_left;
    int32_t pad_right;
    int32_t pad_top;
    int32_t pad_bottom;
    Rect roi = wnd;

    pad_left = pad_right = pad_top = pad_bottom = 0;
    if (roi.x + roi.width > img.width)
        pad_right = roi.x + roi.width - img.width;
    if (roi.x < 0) {
        pad_left = -roi.x;
        roi.x = 0;
    }
    if (roi.y + roi.height > img.height)
        pad_bottom = roi.y + roi.height - img.height;
    if (roi.y < 0) {
        pad_top = -roi.y;
        roi.y = 0;
    }

    fuStDetector->wnd_data_buf_.resize(roi.width * roi.height);
    const uint8_t *src = img.data + roi.y * img.width + roi.x;
    uint8_t *dest = fuStDetector->wnd_data_buf_.data();
    int32_t len = sizeof(uint8_t) * roi.width;
    int32_t len2 = sizeof(uint8_t) * (roi.width - pad_left - pad_right);

    if (pad_top > 0) {
        std::memset(dest, 0, len * pad_top);
        dest += (roi.width * pad_top);
    }
    if (pad_left == 0) {
        if (pad_right == 0) {
            for (int32_t y = pad_top; y < roi.height - pad_bottom; y++) {
                std::memcpy(dest, src, len);
                src += img.width;
                dest += roi.width;
            }
        } else {
            for (int32_t y = pad_top; y < roi.height - pad_bottom; y++) {
                std::memcpy(dest, src, len2);
                src += img.width;
                dest += roi.width;
                std::memset(dest - pad_right, 0, sizeof(uint8_t) * pad_right);
            }
        }
    } else {
        if (pad_right == 0) {
            for (int32_t y = pad_top; y < roi.height - pad_bottom; y++) {
                std::memset(dest, 0, sizeof(uint8_t) * pad_left);
                std::memcpy(dest + pad_left, src, len2);
                src += img.width;
                dest += roi.width;
            }
        } else {
            for (int32_t y = pad_top; y < roi.height - pad_bottom; y++) {
                std::memset(dest, 0, sizeof(uint8_t) * pad_left);
                std::memcpy(dest + pad_left, src, len2);
                src += img.width;
                dest += roi.width;
                std::memset(dest - pad_right, 0, sizeof(uint8_t) * pad_right);
            }
        }
    }
    if (pad_bottom > 0)
        std::memset(dest, 0, len * pad_bottom);

    ImageData src_img(roi.width, roi.height);
    ImageData dest_img(wnd_size_, wnd_size_);
    src_img.data = fuStDetector->wnd_data_buf_.data();
    dest_img.data = fuStDetector->wnd_data_.data();
    ResizeImage(src_img, &dest_img);
}


void ComputeGradX(SURFFeatureMap *feat_map_, const int32_t *input) {
    int32_t *dx = feat_map_->grad_x_.data();
    int32_t len = feat_map_->width_ - 2;

    {

        for (int32_t r = 0; r < feat_map_->height_; r++) {
            const int32_t *src = input + r * feat_map_->width_;
            int32_t *dest = dx + r * feat_map_->width_;
            *dest = ((*(src + 1)) - (*src)) << 1;
            VectorSub(src + 2, src, dest + 1, len);
            dest += (feat_map_->width_ - 1);
            src += (feat_map_->width_ - 1);
            *dest = ((*src) - (*(src - 1))) << 1;
        }
    }
}

void ComputeGradY(SURFFeatureMap *feat_map_, const int32_t *input) {
    int32_t *dy = feat_map_->grad_y_.data();
    int32_t len = feat_map_->width_;
    VectorSub(input + feat_map_->width_, input, dy, len);
    VectorAdd(dy, dy, dy, len);

    {

        for (int32_t r = 1; r < feat_map_->height_ - 1; r++) {
            const int32_t *src = input + (r - 1) * feat_map_->width_;
            int32_t *dest = dy + r * feat_map_->width_;
            VectorSub(src + (feat_map_->width_ << 1), src, dest, len);
        }
    }
    int32_t offset = (feat_map_->height_ - 1) * feat_map_->width_;
    dy += offset;
    VectorSub(input + offset, input + offset - feat_map_->width_,
              dy, len);
    VectorAdd(dy, dy, dy, len);

}


void ComputeGradientImages(SURFFeatureMap *feat_map_, const uint8_t *input) {
    int32_t len = feat_map_->width_ * feat_map_->height_;
    UInt8ToInt32(input, feat_map_->img_buf_.data(), len);
    ComputeGradX(feat_map_, feat_map_->img_buf_.data());
    ComputeGradY(feat_map_, feat_map_->img_buf_.data());
}

void Reshape2(SURFFeatureMap *feat_map_, int32_t width, int32_t height) {
    feat_map_->width_ = width;
    feat_map_->height_ = height;

    int32_t len = feat_map_->width_ * feat_map_->height_;
    feat_map_->grad_x_.resize(len);
    feat_map_->grad_y_.resize(len);
    feat_map_->int_img_.resize(len * feat_map_->kNumIntChannel);
    feat_map_->img_buf_.resize(len);
}


void MaskIntegralChannel(SURFFeatureMap *feat_map_) {
    const int32_t *grad_x = feat_map_->grad_x_.data();
    const int32_t *grad_y = feat_map_->grad_y_.data();
    int32_t len = feat_map_->width_ * feat_map_->height_;
#ifdef USE_SSE
    __m128i dx;
__m128i dy;
__m128i dx_mask;
__m128i dy_mask;
__m128i zero = _mm_set1_epi32(0);
__m128i xor_bits = _mm_set_epi32(0x0, 0x0, 0xffffffff, 0xffffffff);
__m128i data;
__m128i result;
__m128i* src = reinterpret_cast<__m128i*>(int_img_.data());

for (int32_t i = 0; i < len; i++) {
dx = _mm_set1_epi32(*(grad_x++));
dy = _mm_set1_epi32(*(grad_y++));
dx_mask = _mm_xor_si128(_mm_cmplt_epi32(dx, zero), xor_bits);
dy_mask = _mm_xor_si128(_mm_cmplt_epi32(dy, zero), xor_bits);

data = _mm_loadu_si128(src);
result = _mm_and_si128(data, dy_mask);
_mm_storeu_si128(src++, result);
data = _mm_loadu_si128(src);
result = _mm_and_si128(data, dx_mask);
_mm_storeu_si128(src++, result);
}
#else
    int32_t dx, dy, dx_mask, dy_mask, cmp;
    int32_t xor_bits[] = {-1, -1, 0, 0};

    int32_t *src = feat_map_->int_img_.data();
    for (int32_t i = 0; i < len; i++) {
        dy = *(grad_y++);
        dx = *(grad_x++);

        cmp = dy < 0 ? 0xffffffff : 0x0;
        for (int32_t j = 0; j < 4; j++) {
            // cmp xor xor_bits
            dy_mask = cmp ^ xor_bits[j];
            *(src) = (*src) & dy_mask;
            src++;
        }

        cmp = dx < 0 ? 0xffffffff : 0x0;
        for (int32_t j = 0; j < 4; j++) {
            // cmp xor xor_bits
            dx_mask = cmp ^ xor_bits[j];
            *(src) = (*src) & dx_mask;
            src++;
        }
    }
#endif
}


void FillIntegralChannel(SURFFeatureMap *feat_map_, const int32_t *src, int32_t ch) {
    int32_t *dest = feat_map_->int_img_.data() + ch;
    int32_t len = feat_map_->width_ * feat_map_->height_;
    for (int32_t i = 0; i < len; i++) {
        *dest = *src;
        *(dest + 2) = *src;
        dest += feat_map_->kNumIntChannel;
        src++;
    }
}


void VectorCumAdd(int32_t *x, int32_t len,
                  int32_t num_channel) {
#ifdef USE_SSE
    __m128i x1;
__m128i y1;
__m128i z1;
__m128i* x2 = reinterpret_cast<__m128i*>(x);
__m128i* y2 = reinterpret_cast<__m128i*>(x + num_channel);
__m128i* z2 = y2;

len = len / num_channel - 1;
for (int32_t i = 0; i < len; i++) {
// first 4 channels
x1 = _mm_loadu_si128(x2++);
y1 = _mm_loadu_si128(y2++);
z1 = _mm_add_epi32(x1, y1);
_mm_storeu_si128(z2, z1);
z2 = y2;

// second 4 channels
x1 = _mm_loadu_si128(x2++);
y1 = _mm_loadu_si128(y2++);
z1 = _mm_add_epi32(x1, y1);
_mm_storeu_si128(z2, z1);
z2 = y2;
}
#else
    int32_t cols = len / num_channel - 1;
    for (int32_t i = 0; i < cols; i++) {
        int32_t *col1 = x + i * num_channel;
        int32_t *col2 = col1 + num_channel;
        VectorAdd(col1, col2, col2, num_channel);
    }
#endif
}

void Integral(SURFFeatureMap *feat_map_) {
    int32_t *data = feat_map_->int_img_.data();
    int32_t len = feat_map_->kNumIntChannel * feat_map_->width_;

    // Cummulative sum by row
    for (int32_t r = 0; r < feat_map_->height_ - 1; r++) {
        int32_t *row1 = data + r * len;
        int32_t *row2 = row1 + len;
        VectorAdd(row1, row2, row2, len);
    }
    // Cummulative sum by column
    for (int32_t r = 0; r < feat_map_->height_; r++)
        VectorCumAdd(data + r * len, len, feat_map_->kNumIntChannel);
}


void ComputeIntegralImages2(SURFFeatureMap *feat_map_) {
    FillIntegralChannel(feat_map_, feat_map_->grad_x_.data(), 0);
    FillIntegralChannel(feat_map_, feat_map_->grad_y_.data(), 4);

    int32_t len = feat_map_->width_ * feat_map_->height_;
    VectorAbs(feat_map_->grad_x_.data(), feat_map_->img_buf_.data(), len);
    FillIntegralChannel(feat_map_, feat_map_->img_buf_.data(), 1);
    VectorAbs(feat_map_->grad_y_.data(), feat_map_->img_buf_.data(), len);
    FillIntegralChannel(feat_map_, feat_map_->img_buf_.data(), 5);
    MaskIntegralChannel(feat_map_);
    Integral(feat_map_);
}


void Compute2(SURFFeatureMap *feat_map_, const uint8_t *input, int32_t width,
              int32_t height) {
    if (input == nullptr || width <= 0 || height <= 0) {
        return;  // @todo handle the error!
    }
    Reshape2(feat_map_, width, height);
    ComputeGradientImages(feat_map_, input);
    ComputeIntegralImages2(feat_map_);
}


inline void SetROI2(SURFFeatureMap *feat_map_, Rect &roi) {
    feat_map_->roi_ = roi;
    if (feat_map_->buf_valid_reset_) {
        std::memset(feat_map_->buf_valid_.data(), 0, feat_map_->buf_valid_.size() * sizeof(int32_t));
        feat_map_->buf_valid_reset_ = false;
    }
}


void ComputeFeatureVector(SURFFeatureMap *feat_map_, const SURFFeature &feat,
                          int32_t *feat_vec) {
    int32_t init_cell_x = feat_map_->roi_.x + feat.patch.x;
    int32_t init_cell_y = feat_map_->roi_.y + feat.patch.y;
    int32_t cell_width = feat.patch.width / feat.num_cell_per_row * feat_map_->kNumIntChannel;
    int32_t cell_height = feat.patch.height / feat.num_cell_per_col;
    int32_t row_width = feat_map_->width_ * feat_map_->kNumIntChannel;
    const int32_t *cell_top_left[feat_map_->kNumIntChannel];
    const int32_t *cell_top_right[feat_map_->kNumIntChannel];
    const int32_t *cell_bottom_left[feat_map_->kNumIntChannel];
    const int32_t *cell_bottom_right[feat_map_->kNumIntChannel];
    int *feat_val = feat_vec;
    const int32_t *int_img = feat_map_->int_img_.data();
    int32_t offset = 0;

    if (init_cell_y != 0) {
        if (init_cell_x != 0) {
            const int32_t *tmp_cell_top_right[feat_map_->kNumIntChannel];

            // cell #1
            offset = row_width * (init_cell_y - 1) +
                     (init_cell_x - 1) * feat_map_->kNumIntChannel;
            for (int32_t i = 0; i < feat_map_->kNumIntChannel; i++) {
                cell_top_left[i] = int_img + (offset++);
                cell_top_right[i] = cell_top_left[i] + cell_width;
                cell_bottom_left[i] = cell_top_left[i] + row_width * cell_height;
                cell_bottom_right[i] = cell_bottom_left[i] + cell_width;
                *(feat_val++) = *(cell_bottom_right[i]) + *(cell_top_left[i]) -
                                *(cell_top_right[i]) - *(cell_bottom_left[i]);
                tmp_cell_top_right[i] = cell_bottom_right[i];
            }

            // cells in 1st row
            for (int32_t i = 1; i < feat.num_cell_per_row; i++) {
                for (int32_t j = 0; j < feat_map_->kNumIntChannel; j++) {
                    cell_top_left[j] = cell_top_right[j];
                    cell_top_right[j] += cell_width;
                    cell_bottom_left[j] = cell_bottom_right[j];
                    cell_bottom_right[j] += cell_width;
                    *(feat_val++) = *(cell_bottom_right[j]) + *(cell_top_left[j]) -
                                    *(cell_top_right[j]) - *(cell_bottom_left[j]);
                }
            }

            for (int32_t i = 0; i < feat_map_->kNumIntChannel; i++)
                cell_top_right[i] = tmp_cell_top_right[i];
        } else {
            const int32_t *tmp_cell_top_right[feat_map_->kNumIntChannel];

            // cell #1
            offset = row_width * (init_cell_y - 1) + cell_width - feat_map_->kNumIntChannel;
            for (int32_t i = 0; i < feat_map_->kNumIntChannel; i++) {
                cell_top_right[i] = int_img + (offset++);
                cell_bottom_right[i] = cell_top_right[i] + row_width * cell_height;
                tmp_cell_top_right[i] = cell_bottom_right[i];
                *(feat_val++) = *(cell_bottom_right[i]) - *(cell_top_right[i]);
            }

            // cells in 1st row
            for (int32_t i = 1; i < feat.num_cell_per_row; i++) {
                for (int32_t j = 0; j < feat_map_->kNumIntChannel; j++) {
                    cell_top_left[j] = cell_top_right[j];
                    cell_top_right[j] += cell_width;
                    cell_bottom_left[j] = cell_bottom_right[j];
                    cell_bottom_right[j] += cell_width;
                    *(feat_val++) = *(cell_bottom_right[j]) + *(cell_top_left[j]) -
                                    *(cell_top_right[j]) - *(cell_bottom_left[j]);
                }
            }

            for (int32_t i = 0; i < feat_map_->kNumIntChannel; i++)
                cell_top_right[i] = tmp_cell_top_right[i];
        }
    } else {
        if (init_cell_x != 0) {
            // cell #1
            offset = row_width * (cell_height - 1) +
                     (init_cell_x - 1) * feat_map_->kNumIntChannel;
            for (int32_t i = 0; i < feat_map_->kNumIntChannel; i++) {
                cell_bottom_left[i] = int_img + (offset++);
                cell_bottom_right[i] = cell_bottom_left[i] + cell_width;
                *(feat_val++) = *(cell_bottom_right[i]) - *(cell_bottom_left[i]);
                cell_top_right[i] = cell_bottom_right[i];
            }

            // cells in 1st row
            for (int32_t i = 1; i < feat.num_cell_per_row; i++) {
                for (int32_t j = 0; j < feat_map_->kNumIntChannel; j++) {
                    cell_bottom_left[j] = cell_bottom_right[j];
                    cell_bottom_right[j] += cell_width;
                    *(feat_val++) = *(cell_bottom_right[j]) - *(cell_bottom_left[j]);
                }
            }
        } else {
            // cell #1
            offset = row_width * (cell_height - 1) + cell_width - feat_map_->kNumIntChannel;
            for (int32_t i = 0; i < feat_map_->kNumIntChannel; i++) {
                cell_bottom_right[i] = int_img + (offset++);
                *(feat_val++) = *(cell_bottom_right[i]);
                cell_top_right[i] = cell_bottom_right[i];
            }

            // cells in 1st row
            for (int32_t i = 1; i < feat.num_cell_per_row; i++) {
                for (int32_t j = 0; j < feat_map_->kNumIntChannel; j++) {
                    cell_bottom_left[j] = cell_bottom_right[j];
                    cell_bottom_right[j] += cell_width;
                    *(feat_val++) = *(cell_bottom_right[j]) - *(cell_bottom_left[j]);
                }
            }
        }
    }

    // from BR of last cell in current row to BR of first cell in next row
    offset = cell_height * row_width - feat.patch.width *
                                       feat_map_->kNumIntChannel + cell_width;

    // cells in following rows
    for (int32_t i = 1; i < feat.num_cell_per_row; i++) {
        // cells in 1st column
        if (init_cell_x == 0) {
            for (int32_t j = 0; j < feat_map_->kNumIntChannel; j++) {
                cell_bottom_right[j] += offset;
                *(feat_val++) = *(cell_bottom_right[j]) - *(cell_top_right[j]);
            }
        } else {
            for (int32_t j = 0; j < feat_map_->kNumIntChannel; j++) {
                cell_bottom_right[j] += offset;
                cell_top_left[j] = cell_top_right[j] - cell_width;
                cell_bottom_left[j] = cell_bottom_right[j] - cell_width;
                *(feat_val++) = *(cell_bottom_right[j]) + *(cell_top_left[j]) -
                                *(cell_top_right[j]) - *(cell_bottom_left[j]);
            }
        }

        // cells in following columns
        for (int32_t j = 1; j < feat.num_cell_per_row; j++) {
            for (int32_t k = 0; k < feat_map_->kNumIntChannel; k++) {
                cell_top_left[k] = cell_top_right[k];
                cell_top_right[k] += cell_width;

                cell_bottom_left[k] = cell_bottom_right[k];
                cell_bottom_right[k] += cell_width;

                *(feat_val++) = *(cell_bottom_right[k]) + *(cell_top_left[k]) -
                                *(cell_bottom_left[k]) - *(cell_top_right[k]);
            }
        }

        for (int32_t j = 0; j < feat_map_->kNumIntChannel; j++)
            cell_top_right[j] += offset;
    }
}

void NormalizeFeatureVectorL2(const int32_t *feat_vec,
                              float *feat_vec_normed, int32_t len) {
    double prod = 0.0;
    float norm_l2 = 0.0f;

    for (int32_t i = 0; i < len; i++)
        prod += static_cast<double>(feat_vec[i] * feat_vec[i]);
    if (prod != 0) {
        norm_l2 = static_cast<float>(std::sqrt(prod));
        for (int32_t i = 0; i < len; i++)
            feat_vec_normed[i] = feat_vec[i] / norm_l2;
    } else {
        for (int32_t i = 0; i < len; i++)
            feat_vec_normed[i] = 0.0f;
    }
}


void GetFeatureVector(SURFFeatureMap *feat_map_, int32_t feat_id, float *feat_vec) {
    if (feat_map_->buf_valid_[feat_id] == 0) {
        ComputeFeatureVector(feat_map_, feat_map_->feat_pool_.pool_[feat_id],
                             feat_map_->feat_vec_buf_[feat_id].data());
        NormalizeFeatureVectorL2(feat_map_->feat_vec_buf_[feat_id].data(),
                                 feat_map_->feat_vec_normed_buf_[feat_id].data(),
                                 static_cast<int32_t>(feat_map_->feat_vec_normed_buf_[feat_id].size()));
        feat_map_->buf_valid_[feat_id] = 1;
        feat_map_->buf_valid_reset_ = true;
    }

    std::memcpy(feat_vec, feat_map_->feat_vec_normed_buf_[feat_id].data(),
                feat_map_->feat_vec_normed_buf_[feat_id].size() * sizeof(float));
}


int32_t GetFeatureVectorDim(SURFFeatureMap *feat_map_, int32_t feat_id) {
    return (feat_map_->feat_pool_.pool_[feat_id].num_cell_per_col *
            feat_map_->feat_pool_.pool_[feat_id].num_cell_per_row * feat_map_->kNumIntChannel);
}


inline float Sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(x));
}

inline float ReLU(float x) {
    return (x > 0.0f ? x : 0.0f);
}

void Compute21(MLPLayer *mlpLayer, const float *input, float *output) {
    {
        for (int32_t i = 0; i < mlpLayer->output_dim_; i++) {
            output[i] = VectorInnerProduct(input, mlpLayer->weights_.data() + i * mlpLayer->input_dim_,
                                           mlpLayer->input_dim_) + mlpLayer->bias_[i];
            output[i] = (mlpLayer->act_func_type_ == 1 ? ReLU(output[i]) : Sigmoid(-output[i]));
        }
    }
}

void Compute2(MLP &mlp, const float *input, float *output) {
    mlp.layer_buf_[0].resize(mlp.layers_[0]->output_dim_);
    Compute21(mlp.layers_[0], input, mlp.layer_buf_[0].data());

    size_t i; /**< layer index */
    for (i = 1; i < mlp.layers_.size() - 1; i++) {
        mlp.layer_buf_[i % 2].resize(mlp.layers_[i]->output_dim_);
        Compute21(mlp.layers_[i], mlp.layer_buf_[(i + 1) % 2].data(), mlp.layer_buf_[i % 2].data());
    }
    Compute21(mlp.layers_.back(), mlp.layer_buf_[(i + 1) % 2].data(), output);
}

bool Classify2(SURFMLP &surfmlp, float *score, float *outputs) {
    float *dest = surfmlp.input_buf_.data();
    for (size_t i = 0; i < surfmlp.feat_id_.size(); i++) {
        GetFeatureVector(surfmlp.feat_map_, surfmlp.feat_id_[i] - 1, dest);
        dest += GetFeatureVectorDim(surfmlp.feat_map_, surfmlp.feat_id_[i]);
    }
    surfmlp.output_buf_.resize(surfmlp.model_.layers_.back()->output_dim_);
    Compute2(surfmlp.model_, surfmlp.input_buf_.data(), surfmlp.output_buf_.data());

    if (score != nullptr)
        *score = surfmlp.output_buf_[0];
    if (outputs != nullptr) {
        std::memcpy(outputs, surfmlp.output_buf_.data(),
                    surfmlp.model_.layers_.back()->output_dim_ * sizeof(float));
    }

    return (surfmlp.output_buf_[0] > surfmlp.thresh_);
}


void AddFeature(int32_t x, int32_t y, LABBoostedClassifier &labBoostedClassifier) {
    LABFeature feat;
    feat.x = x;
    feat.y = y;
    labBoostedClassifier.feat_.push_back(feat);
    //feat_.push_back(feat);
}

void ReadFeatureParam0(std::istream *input,
                       int num_base_classifer_, LABBoostedClassifier &labBoostedClassifier) {
    int32_t x;
    int32_t y;
    for (int32_t i = 0; i < num_base_classifer_; i++) {
        input->read(reinterpret_cast<char *>(&x), sizeof(int32_t));
        input->read(reinterpret_cast<char *>(&y), sizeof(int32_t));
        AddFeature(x, y, labBoostedClassifier);
    }

    //return !input->fail();
}

void SetWeights(const float *weights, int32_t num_bin, LABBaseClassifier *classifier) {
    classifier->weights_.resize(num_bin + 1);
    classifier->num_bin_ = num_bin;
    std::copy(weights, weights + classifier->num_bin_ + 1, classifier->weights_.begin());
}

void SetThreshold(float thresh, LABBaseClassifier *classifier) {
    classifier->thresh_ = thresh;
}
//bool LABBoostedClassifier_Classify(float* score, float* outputs,LABBaseClassifier *classifier) {
//    bool isPos = true;
//    float s = 0.0f;
//
//    for (size_t i = 0; isPos && i < base_classifiers_.size();) {
//        for (int32_t j = 0; j < kFeatGroupSize; j++, i++) {
//            uint8_t featVal = feat_map_->GetFeatureVal(feat_[i].x, feat_[i].y);
//            s += base_classifiers_[i]->weights(featVal);
//        }
//        if (s < base_classifiers_[i - 1]->threshold())
//            isPos = false;
//    }
//    isPos = isPos && ((!use_std_dev_) || feat_map_->GetStdDev() > kStdDevThresh);
//
//    if (score != nullptr)
//        *score = s;
//    if (outputs != nullptr)
//        *outputs = s;
//
//    return isPos;
//}

void AddBaseClassifier(const float *weights,
                       int32_t num_bin, float thresh, LABBoostedClassifier &labBoostedClassifier) {
    LABBaseClassifier *classifier = new LABBaseClassifier();
    SetWeights(weights, num_bin, classifier);
    SetThreshold(thresh, classifier);
    labBoostedClassifier.base_classifiers_.push_back(classifier);
}

void ReadBaseClassifierParam0(std::istream *input, int num_base_classifer_, int num_bin_,
                              LABBoostedClassifier &labBoostedClassifier) {
    std::vector<float> thresh;
    thresh.resize(num_base_classifer_);
    input->read(reinterpret_cast<char *>(thresh.data()),
                sizeof(float) * num_base_classifer_);

    int32_t weight_len = sizeof(float) * (num_bin_ + 1);
    std::vector<float> weights;
    weights.resize(num_bin_ + 1);
    for (int32_t i = 0; i < num_base_classifer_; i++) {
        input->read(reinterpret_cast<char *>(weights.data()), weight_len);
        //model->AddBaseClassifier(weights.data(), num_bin_, thresh[i]);
    }

    //return !input->fail();
}


void Read0(std::istream *input, LABBoostedClassifier &labBoostedClassifier) {
    bool is_read;
    int num_base_classifer_;
    int num_bin_;

    input->read(reinterpret_cast<char *>(&num_base_classifer_), sizeof(int32_t));
    input->read(reinterpret_cast<char *>(&num_bin_), sizeof(int32_t));
    ReadFeatureParam0(input, num_base_classifer_, labBoostedClassifier);

    ReadBaseClassifierParam0(input, num_base_classifer_, num_bin_, labBoostedClassifier);

//return is_read;
}


inline void MLP_SetSize(int32_t inputDim, int32_t outputDim, MLPLayer *layer) {

    layer->input_dim_ = inputDim;
    layer->output_dim_ = outputDim;
    layer->weights_.resize(inputDim * outputDim);
    layer->bias_.resize(outputDim);
}

inline void MLP_SetWeights(const float *weights, int32_t len, MLPLayer *layer) {

    std::copy(weights, weights + layer->input_dim_ * layer->output_dim_, layer->weights_.begin());
}

inline void MLP_SetBias(const float *bias, int32_t len, MLPLayer *layer) {

    std::copy(bias, bias + layer->output_dim_, layer->bias_.begin());
}

void MLP_AddLayer(int32_t inputDim, int32_t outputDim, const float *weights,
                  const float *bias, bool is_output, SURFMLP &surfmlp) {


    MLPLayer *layer = new MLPLayer(is_output ? 0 : 1);
    MLP_SetSize(inputDim, outputDim, layer);
    MLP_SetWeights(weights, inputDim * outputDim, layer);
    MLP_SetBias(bias, outputDim, layer);
    surfmlp.model_.layers_.push_back(layer);
    //layers_.push_back(layer);
}

void AddLayer(int32_t input_dim, int32_t output_dim,
              const float *weights, const float *bias, bool is_output, SURFMLP &surfmlp) {
    if (surfmlp.model_.layers_.size() == 0)
        surfmlp.input_buf_.resize(input_dim);
    MLP_AddLayer(input_dim, output_dim, weights, bias, is_output, surfmlp);
}

void Read1(std::istream *input, SURFMLP &surfmlp) {
    bool is_read = false;

    int32_t num_layer;
    int32_t num_feat;
    int32_t input_dim;
    int32_t output_dim;
    float thresh;
    std::vector<int32_t> feat_id_buf_;
    std::vector<float> weights_buf_;
    std::vector<float> bias_buf_;

    input->read(reinterpret_cast<char *>(&num_layer), sizeof(int32_t));
    if (num_layer <= 0) {
        is_read = false;  // @todo handle the errors and the following ones!!!
    }
    input->read(reinterpret_cast<char *>(&num_feat), sizeof(int32_t));
    if (num_feat <= 0) {
        is_read = false;
    }


    feat_id_buf_.resize(num_feat);
    input->read(reinterpret_cast<char*>(feat_id_buf_.data()),
                sizeof(int32_t) * num_feat);
    for (int32_t i = 0; i < num_feat; i++)
        surfmlp.feat_id_.push_back(feat_id_buf_[i]);

    input->read(reinterpret_cast<char*>(&thresh), sizeof(float));
    surfmlp.thresh_ = (thresh);

    input->read(reinterpret_cast<char *>(&input_dim), sizeof(int32_t));
    if (input_dim <= 0) {
        is_read = false;
    }

    for (int32_t i = 1; i < num_layer; i++) {
        input->read(reinterpret_cast<char *>(&output_dim), sizeof(int32_t));
        if (output_dim <= 0) {
            is_read = false;
        }

        int32_t len = input_dim * output_dim;
        weights_buf_.resize(len);
        input->read(reinterpret_cast<char *>(weights_buf_.data()),
                    sizeof(float) * len);

        bias_buf_.resize(output_dim);
        input->read(reinterpret_cast<char *>(bias_buf_.data()),
                    sizeof(float) * output_dim);

        if (i < num_layer - 1) {
            printf("okok\n");
            AddLayer(input_dim, output_dim, weights_buf_.data(),
                     bias_buf_.data(), false, surfmlp);
        } else {
            printf("okaaa\n");
            AddLayer(input_dim, output_dim, weights_buf_.data(),
                     bias_buf_.data(), true, surfmlp);
        }
        input_dim = output_dim;
    }
//
//    is_read = !input->fail();
//
//    return is_read;
}

std::vector<FaceInfo> Detect(FuStDetector *fuStDetector, ImagePyramid *img_pyramid) {

    float score;
    FaceInfo wnd_info;
    Rect wnd;
    float scale_factor = 0.0;
    const ImageData *img_scaled =
            GetNextScaleImage(img_pyramid, &scale_factor);

    wnd.height = wnd.width = wnd_size_;


    std::vector<std::vector<FaceInfo> > proposals(3);


    while (img_scaled != 0) {


        LABFeatureMap feat_map_;
        Compute(&feat_map_, img_scaled->data, img_scaled->width,
                img_scaled->height);

        wnd_info.bbox.width = static_cast<int32_t>(wnd_size_ / scale_factor + 0.5);
        wnd_info.bbox.height = wnd_info.bbox.width;

        int32_t max_x = img_scaled->width - wnd_size_;
        int32_t max_y = img_scaled->height - wnd_size_;
        for (int32_t y = 0; y <= max_y; y += slide_wnd_step_y_) {
            wnd.y = y;
            for (int32_t x = 0; x <= max_x; x += slide_wnd_step_x_) {
                wnd.x = x;
                SetROI(&feat_map_, wnd);

                wnd_info.bbox.x = static_cast<int32_t>(x / scale_factor + 0.5);
                wnd_info.bbox.y = static_cast<int32_t>(y / scale_factor + 0.5);



                //for (int32_t i = 0; i < hierarchy_size_[0]; i++) {
                for (int32_t i = 0; i < 3; i++) {

                    printf("%d,%d,%d\n",y,x,i);

                    if (Classify0(&score, 0, lABBoostedClassifierS[i])) {
                        wnd_info.score = static_cast<double>(score);
                        proposals[i].push_back(wnd_info);
                    }
                }
            }
        }



        img_scaled = GetNextScaleImage(&scale_factor, img_pyramid);
    }




    std::vector<std::vector<FaceInfo> > proposals_nms(3);
    for (int32_t i = 0; i < 3; i++) {
        NonMaximumSuppression(&(proposals[i]),
                              &(proposals_nms[i]), 0.8f);
        proposals[i].clear();
    }




    // Following classifiers

    ImageData img = iimage1x(img_pyramid);
    Rect roi;
    std::vector<float> mlp_predicts(4);  // @todo no hard-coded number!
    roi.x = roi.y = 0;
    roi.width = roi.height = wnd_size_;

    int32_t cls_idx = 3;//hierarchy_size_[0];
    int32_t model_idx = 3;//hierarchy_size_[0];
    std::vector<int32_t> buf_idx;

    int hierarchy_size_[3] = {3, 1, 1};
    int num_stage_[5] = {1, 1, 1, 2, 1};

    printf("eeee\n");


    std::vector<std::vector<int> > wnd_src_id_;
    std::vector<int> v1;
    wnd_src_id_.push_back(v1);
    std::vector<int> v2;
    wnd_src_id_.push_back(v2);
    std::vector<int> v3;
    wnd_src_id_.push_back(v3);
    std::vector<int> v4;
    v4.push_back(0);
    v4.push_back(1);
    v4.push_back(2);
    wnd_src_id_.push_back(v4);

    std::vector<int> v5;
    v5.push_back(0);
    wnd_src_id_.push_back(v5);


    //wnd_src_id_

//    wnd_src_id_[3][0] = 0;
//    wnd_src_id_[3][0] = 1;
//    wnd_src_id_[3][1] = 2;
//    wnd_src_id_[4][0] = 0;

    printf("fff\n");


    for (int32_t i = 1; i < 3; i++) {
        buf_idx.resize(hierarchy_size_[i]);
        for (int32_t j = 0; j < hierarchy_size_[i]; j++) {
            int32_t num_wnd_src = static_cast<int32_t>(wnd_src_id_[cls_idx].size());
            std::vector<int32_t> &wnd_src = wnd_src_id_[cls_idx];
            buf_idx[j] = wnd_src[0];
            proposals[buf_idx[j]].clear();
            for (int32_t k = 0; k < num_wnd_src; k++) {
                proposals[buf_idx[j]].insert(proposals[buf_idx[j]].end(),
                                             proposals_nms[wnd_src[k]].begin(), proposals_nms[wnd_src[k]].end());
            }

//            std::shared_ptr<seeta::fd::FeatureMap> & feat_map =
//                    feat_map_[cls2feat_idx_[model_[model_idx]->type()]];

            // printf("feat_map:%d,%d,%d\n",i,j,model_[model_idx]->type());
            for (int32_t k = 0; k < num_stage_[cls_idx]; k++) {
                int32_t num_wnd = static_cast<int32_t>(proposals[buf_idx[j]].size());
                std::vector<FaceInfo> &bboxes = proposals[buf_idx[j]];
                int32_t bbox_idx = 0;

                for (int32_t m = 0; m < num_wnd; m++) {
                    if (bboxes[m].bbox.x + bboxes[m].bbox.width <= 0 ||
                        bboxes[m].bbox.y + bboxes[m].bbox.height <= 0)
                        continue;
                    GetWindowData(fuStDetector, img, bboxes[m].bbox);


                    SURFFeatureMap feat_map_;

                    Compute2(&feat_map_, fuStDetector->wnd_data_.data(), wnd_size_, wnd_size_);
                    SetROI2(&feat_map_, roi);

                    printf("%d\n",sURFMLPS.size());

                    if (Classify2(sURFMLPS[0], &score, mlp_predicts.data())) {
                        float x = static_cast<float>(bboxes[m].bbox.x);
                        float y = static_cast<float>(bboxes[m].bbox.y);
                        float w = static_cast<float>(bboxes[m].bbox.width);
                        float h = static_cast<float>(bboxes[m].bbox.height);

                        bboxes[bbox_idx].bbox.width =
                                static_cast<int32_t>((mlp_predicts[3] * 2 - 1) * w + w + 0.5);
                        bboxes[bbox_idx].bbox.height = bboxes[bbox_idx].bbox.width;
                        bboxes[bbox_idx].bbox.x =
                                static_cast<int32_t>((mlp_predicts[1] * 2 - 1) * w + x +
                                                     (w - bboxes[bbox_idx].bbox.width) * 0.5 + 0.5);
                        bboxes[bbox_idx].bbox.y =
                                static_cast<int32_t>((mlp_predicts[2] * 2 - 1) * h + y +
                                                     (h - bboxes[bbox_idx].bbox.height) * 0.5 + 0.5);
                        bboxes[bbox_idx].score = score;
                        bbox_idx++;
                    }
                }
                proposals[buf_idx[j]].resize(bbox_idx);

                if (k < num_stage_[cls_idx] - 1) {
                    NonMaximumSuppression(&(proposals[buf_idx[j]]),
                                          &(proposals_nms[buf_idx[j]]), 0.8f);
                    proposals[buf_idx[j]] = proposals_nms[buf_idx[j]];
                } else {
                    if (i == 3 - 1) {
                        NonMaximumSuppression(&(proposals[buf_idx[j]]),
                                              &(proposals_nms[buf_idx[j]]), 0.3f);
                        proposals[buf_idx[j]] = proposals_nms[buf_idx[j]];
                    }
                }
                model_idx++;
            }

            cls_idx++;
        }

        for (int32_t j = 0; j < hierarchy_size_[i]; j++)
            proposals_nms[j] = proposals[buf_idx[j]];
    }

    return proposals_nms[0];


}

void readModel() {

    std::ifstream model_file("/home/yanhao/lib/modelData/model_200.bin", std::ifstream::binary);
    FILE *fp = fopen("/home/yanhao/project/FD/1.txt", "w");
    if (!fp) {
        printf("aaa");
        exit(-1);
    }

    int num_hierarchy_;
    int hierarchy_size;
    int num_stage;
    int type_id;
    int num_wnd_src;
    std::vector<int> hierarchy_size_;
    std::vector<int> num_stage_;

    std::vector<std::vector<int32_t> > wnd_src_id_;

    model_file.read(reinterpret_cast<char *>(&num_hierarchy_), sizeof(int32_t));

    fprintf(fp, "%s:\n", "num_hierarchy_");
    fprintf(fp, "%d\n", num_hierarchy_);

    for (int32_t i = 0; i < num_hierarchy_; i++) {
        model_file.read(reinterpret_cast<char *>(&hierarchy_size),
                        sizeof(int32_t));


        hierarchy_size_.push_back(hierarchy_size);
        for (int32_t j = 0; j < hierarchy_size; j++) {
            model_file.read(reinterpret_cast<char *>(&num_stage), sizeof(int32_t));
            num_stage_.push_back(num_stage);
            for (int32_t k = 0; k < num_stage; k++) {
                model_file.read(reinterpret_cast<char *>(&type_id), sizeof(int32_t));
                //classifier_type = static_cast<seeta::fd::ClassifierType>(type_id);
                //reader = CreateModelReader(classifier_type);

                //classifier = CreateClassifier(classifier_type);

                //is_loaded = !model_file.fail() &&
                // reader->Read(&model_file, classifier.get());

                printf("%d,%d,%d\n", num_hierarchy_, hierarchy_size, num_stage);
                printf("haha:%d\n", type_id);

                if (type_id == 0) {
                    LABBoostedClassifier labBoostedClassifier;
                    Read0(&model_file, labBoostedClassifier);
                    lABBoostedClassifierS.push_back(labBoostedClassifier);

                }


                if (type_id == 1) {
                    SURFMLP surfmlp;
                    Read1(&model_file, surfmlp);
                    sURFMLPS.push_back(surfmlp);
                }

                //model_.push_back(classifier);
//                    std::shared_ptr<seeta::fd::FeatureMap> feat_map;
//                    if (cls2feat_idx_.count(classifier_type) == 0) {
//                        feat_map_.push_back(CreateFeatureMap(classifier_type));
//                        cls2feat_idx_.insert(
//                                std::map<seeta::fd::ClassifierType, int32_t>::value_type(
//                                        classifier_type, feat_map_index++));
//                        //printf("type:%d,%d,%d,%d,%d\n",i,j,k,classifier_type,feat_map_index-1);
//                    }
//                    feat_map = feat_map_[cls2feat_idx_.at(classifier_type)];
//                    model_.back()->SetFeatureMap(feat_map.get());

            }

            wnd_src_id_.push_back(std::vector<int32_t>());
            model_file.read(reinterpret_cast<char *>(&num_wnd_src), sizeof(int32_t));
            if (num_wnd_src > 0) {
                wnd_src_id_.back().resize(num_wnd_src);
                for (int32_t k = 0; k < num_wnd_src; k++) {
                    model_file.read(reinterpret_cast<char *>(&(wnd_src_id_.back()[k])),
                                    sizeof(int32_t));
                }
            }
        }
    }


    fprintf(fp, "%s\n", "hierarchy_size_");
    fprintf(fp, "%d\n", hierarchy_size_.size());
    for (int l = 0; l < hierarchy_size_.size(); ++l) {
        fprintf(fp, "%d ", hierarchy_size_[l]);
    }
    fprintf(fp, "\n");

    fprintf(fp, "%s\n", "num_stage_");
    fprintf(fp, "%d\n", num_stage_.size());
    for (int m = 0; m < num_stage_.size(); ++m) {
        fprintf(fp, "%d ", num_stage_[m]);
    }
    fprintf(fp, "\n");

    fprintf(fp, "%s\n", "wnd_src_id_");
    fprintf(fp, "%d\n", wnd_src_id_.size());
    for (int n = 0; n < wnd_src_id_.size(); ++n) {
        for (int k = 0; k < wnd_src_id_[n].size(); ++k) {
            fprintf(fp, "%d ", wnd_src_id_[n][k]);
        }
        fprintf(fp, "\n");
    }


    //fprintf(fp, "feat_map_=%d\n", feat_map_.size());

    model_file.close();
    fclose(fp);
    fp = 0;
}


void SetImage1x(ImagePyramid &img_pyramid_, const uint8_t *img_data, int32_t width,
                int32_t height) {
    if (width > img_pyramid_.buf_img_width_ || height > img_pyramid_.buf_img_height_) {
        delete[] img_pyramid_.buf_img_;

        img_pyramid_.buf_img_width_ = width;
        img_pyramid_.buf_img_height_ = height;
        img_pyramid_.buf_img_ = new uint8_t[width * height];
    }

    img_pyramid_.width1x_ = width;
    img_pyramid_.height1x_ = height;
    std::memcpy(img_pyramid_.buf_img_, img_data, width * height * sizeof(uint8_t));
    img_pyramid_.scale_factor_ = img_pyramid_.max_scale_;
    if (img_pyramid_.width1x_ == 0 || img_pyramid_.height1x_ == 0)
        return;

    int32_t max_width = static_cast<int32_t>(img_pyramid_.width1x_ * img_pyramid_.max_scale_ + 0.5);
    int32_t max_height = static_cast<int32_t>(img_pyramid_.height1x_ * img_pyramid_.max_scale_ + 0.5);

    if (max_width > img_pyramid_.buf_scaled_width_ || max_height > img_pyramid_.buf_scaled_height_) {
        delete[] img_pyramid_.buf_img_scaled_;

        img_pyramid_.buf_scaled_width_ = max_width;
        img_pyramid_.buf_scaled_height_ = max_height;
        img_pyramid_.buf_img_scaled_ = new uint8_t[max_width * max_height];

        img_pyramid_.img_scaled_.data = 0;
        img_pyramid_.img_scaled_.width = 0;
        img_pyramid_.img_scaled_.height = 0;
    }
}

int main() {


//    const int kWndSize = 40;
//    int min_face_size_ = 20;
//    int max_face_size_ = -1;
//    int slide_wnd_step_x_ = 4;
//    int slide_wnd_step_y_ = 4;
//    float cls_thresh_ = 3.85f;
//
//    int wnd_size_ = 40;
//    //int slide_wnd_step_x_=4;
//    //int slide_wnd_step_y_=4;
//
//    int num_hierarchy_ = 0;
//
//    std::vector<int> hierarchy_size_;
//    std::vector<int> num_stage_;
//    std::vector<std::vector<int> > wnd_src_id_;

    readModel();

    FuStDetector fuStDetector;

    ImageData img;
    img.data = (unsigned char *) malloc(100 * 100 * sizeof(unsigned char));
    img.width=100;
    img.height=100;
    img.num_channels=1;


    int32_t min_img_size = img.height <= img.width ? img.height : img.width;

    min_img_size = (max_face_size_ > 0 ?
                    (min_img_size >= max_face_size_ ? max_face_size_ : min_img_size) :
                    min_img_size);


    ImagePyramid img_pyramid_;

    SetImage1x(img_pyramid_, img.data, img.width, img.height);
    img_pyramid_.min_scale_ = (static_cast<float>(kWndSize) / min_img_size);

    vector<FaceInfo> faces = Detect(&fuStDetector,&img_pyramid_);
    printf("%d\n",faces.size());


    printf("hello,world!\n");
}

