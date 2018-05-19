#pragma once
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
#define PI 3.1415926535897932384626433832795
#include <math.h>
class MyBlob
{
protected:
	double maxThreshold, minThreshold, delta; // Ngưỡng cho sigma
	Mat srcImg;
private:
	vector<Mat> loGImg; // vector store Laplacian of Gauss images

protected:
	double area(double r); // Tính diện tích ứng với bán kính r
	virtual Mat DetectMaximaKeyPoint();
private:
	// Tích chập với laplace 5*5;
	Mat Laplace(const Mat& srcImg);
public:
	MyBlob();
	MyBlob(const Mat& image, double _minThreshold = 2.4, double _maxThreshold = 3.2, double _delta = 0.02);
	~MyBlob();
	//blob detection algorithm
	Mat Detect();
};

