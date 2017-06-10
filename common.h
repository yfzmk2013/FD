

#ifndef SEETA_COMMON_H_
#define SEETA_COMMON_H_




#define SEETA_NUM_THREADS 4


typedef struct ImageData {
    ImageData() {
        data = nullptr;
        width = 0;
        height = 0;
        num_channels = 0;
    }

    ImageData(int img_width, int img_height,
              int img_num_channels = 1) {
        data = nullptr;
        width = img_width;
        height = img_height;
        num_channels = img_num_channels;
    }

    unsigned char *data;
    int width;
    int height;
    int num_channels;
} ImageData;

typedef struct Rect {
    int x;
    int y;
    int width;
    int height;
} Rect;

typedef struct FaceInfo {
    Rect bbox;
    double roll;
    double pitch;
    double yaw;
    double score; /**< Larger score should mean higher confidence. */
} FaceInfo;

typedef struct {
    double x;
    double y;
} FacialLandmark;


#endif  // SEETA_COMMON_H_
