#pragma once
#ifndef _SIFT_DETECTOR_

#define _SIFT_DETECTOR_

#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <cstdio>
using namespace cv;
using namespace std;
#define RED_CHANNEL 2
#define GREEN_CHANNEL 3
#define BLUE_CHANNEL 4

class SiftDetector {
public:
	Mat octaves[3];
	Mat blurImage[3][5];
	Mat doGImage[3][4];
	double sigma = 1.0;
	double k = sqrt(2);
	Mat convolution(const Mat& srcImg, double filter[][5], double m);
	Mat difference(const Mat& m1, const Mat& m2);
	double generateGaussFilter(double kernel[][5], double sigma);
	
	void CalcDoGImage();
	void CalcBlurImage();
	SiftDetector(const Mat& srcImg);
	SiftDetector();
	~SiftDetector();
	void detect();


};


#endif 
