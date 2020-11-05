#pragma once
#include <iostream>
#include <fstream>
#include <iomanip> // to format image names using setw() and setfill()
//#include <io.h>    // to check file existence using POSIX function access(). On Linux include <unistd.h>.
#include <set>

#include "KalmanTracker.hpp"
#include "Hungarian.hpp"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

typedef struct TrackingBox
{
    int frame;
    int id;
    Rect_<float> box;  // xyx2y2, Rect is xywh
}TrackingBox;


// Computes IOU between two bounding boxes
double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt);


class PersonTrackerSort
{
public:

    PersonTrackerSort();
    ~PersonTrackerSort();

    vector<TrackingBox> get_track_predict(vector<TrackingBox> detFrameData);

    int frame_count;
    int max_age;
    int min_hits;
    double iouThreshold;
    vector<KalmanTracker> trackers;
    vector<TrackingBox> frameTrackingResult;


private:

    // variables used in the for-loop
    vector<Rect_<float>> predictedBoxes;
    vector<vector<double>> iouMatrix;
    vector<int> assignment;
    set<int> unmatchedDetections;
    set<int> unmatchedTrajectories;
    set<int> allItems;
    set<int> matchedItems;
    vector<cv::Point> matchedPairs;

    unsigned int trkNum = 0;
    unsigned int detNum = 0;
};   
