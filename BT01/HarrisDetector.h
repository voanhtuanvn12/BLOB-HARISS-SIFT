#pragma once
#ifndef _HARRIS_DETECTOR_

#define _HARRIS_DETECTOR_

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

float sobelX[3][3] = { { 1,0,-1 },{ 2,0,-2 },{ 1,0,-1 } };
float sobelY[3][3] = { { 1,2,1 },{ 0,0,0 },{ -1,-2,-1 } };
float gaussFilter[5][5] = { { 1,4,7,1,4 },{ 4,16,26,16,4 },{ 7,26,41,26,7 },{ 4,16,26,16,4 },{ 1,4,7,4,1 } };
const int sobelWindowSize = 3;
const int gaussWindowSize = 5;
const float sobelConstant = 1 / 4.0;
const float gaussConstant = 1 / 273.0;


class HarrisDetector
{
public:
	int srcCol;
	int srcRow;
	Mat srcImg;
	Mat ixSquareDerivative;
	Mat iySquareDerivative;
	Mat ixIyDerivative;
	Mat harrisImg;
	Mat MatrixM;
	Mat tempIx;
	Mat tempIy;
	Mat tempIxIy;
public:
	void ConvolutionMatrix();
	void ConvolutionGauss();
	void ConvolutionGauss(Mat& srcImg, Mat& dstImg);
	Mat Detector();
	HarrisDetector();
	HarrisDetector(const Mat& temp);
	~HarrisDetector();

private:

};


void HarrisDetector::ConvolutionMatrix()
{
	float windowSize, constant;
	windowSize = sobelWindowSize;
	constant = sobelConstant;
	// tìm số dòng và số cột
	int srcCol = srcImg.cols;
	int srcRow = srcImg.rows;
	uchar* src_data = (uchar*)srcImg.data;
	//float* dst_Ix_data = (float*)ixDerivative.data;
	//float* dst_Iy_data = (float*)iyDerivative.data;

	float * Ix2_data = (float*)tempIx.data;
	float * Iy2_data = (float*)tempIy.data;
	float * IxIy_data = (float*)tempIxIy.data;

	int src_stepWidth = srcImg.step[0];
	int src_nchannels = srcImg.step[1];

	int haftWindowSize = windowSize / 2;
	for (int y = 0; y < srcRow; ++y, Ix2_data += srcCol, Iy2_data += srcCol, IxIy_data += srcCol) {
		float *Ix2_pRow = Ix2_data;
		float *Iy2_pRow = Iy2_data;
		float *IxIy_pRow = IxIy_data;
		for (int x = 0; x < srcCol; ++x, Ix2_pRow += 1, Iy2_pRow += 1, IxIy_pRow += 1) {
			int yy, xx;
			float resX = 0;
			float resY = 0;
			for (int i = -haftWindowSize; i <= haftWindowSize; i++) {
				for (int j = -haftWindowSize; j <= haftWindowSize; j++) {
					yy = y + i;
					xx = x + j;
					if (yy >= 0 && yy < srcRow && xx >= 0 && xx < srcCol) {
						resX += (float)src_data[yy*src_stepWidth + xx * src_nchannels] * sobelX[1 - i][1 - j];
						resY += (float)src_data[yy*src_stepWidth + xx * src_nchannels] * sobelY[1 - i][1 - j];
					}
				}
			}
			
			resX *= constant;
			resY *= constant;
			//resX /= 255.0;
			//resY /= 255.0;
			
			Ix2_pRow[0] = resX* resX;
			Iy2_pRow[0] = resY* resY;
			IxIy_pRow[0] = resX * resY;
		}
	}

}

void HarrisDetector::ConvolutionGauss()
{
	float windowSize, constant;
	windowSize = gaussWindowSize;
	constant = gaussConstant;




	int srcCol = tempIx.cols;
	int srcRow = tempIx.rows;

	int src_stepWidth = tempIx.step[0];
	int src_nchannels = tempIx.step[1];

	float* src_Ix2_data = (float*)tempIx.data;
	float* src_Iy2_data = (float*)tempIy.data;
	float* src_IxIy_data = (float*)tempIxIy.data;

	float* dst_Ix2_data = (float*)ixSquareDerivative.data;
	float* dst_Iy2_data = (float*)iySquareDerivative.data;
	float* dst_IxIy_data = (float*)ixIyDerivative.data;

	int haftWindowSize = windowSize / 2;
	for (int y = 0; y < srcRow; ++y, dst_Ix2_data += srcCol, dst_Iy2_data += srcCol, dst_IxIy_data += srcCol) {
		float *Ix2_pRow = dst_Ix2_data;
		float *Iy2_pRow = dst_Iy2_data;
		float *IxIy_pRow = dst_IxIy_data;
		for (int x = 0; x < srcCol; ++x, Ix2_pRow += 1, Iy2_pRow += 1, IxIy_pRow += 1) {
			int yy, xx;
			float resX = 0;
			float resY = 0;
			float resXY = 0;
			for (int i = -haftWindowSize; i <= haftWindowSize; i++) {
				for (int j = -haftWindowSize; j <= haftWindowSize; j++) {
					yy = y + i;
					xx = x + j;
					if (yy >= 0 && yy < srcRow && xx >= 0 && xx < srcCol) {
						resX += src_Ix2_data[yy*srcCol + xx] * gaussFilter[2 - i][2 - j];
						resY += src_Iy2_data[yy*srcCol + xx] * gaussFilter[2 - i][2 - j];
						resXY += src_IxIy_data[yy*srcCol + xx] * gaussFilter[2 - i][2 - j];
					}
				}
			}
			resX *= constant;
			resY *= constant;
			resXY *= constant;
			Ix2_pRow[0] = resX;
			Iy2_pRow[0] = resY;
			IxIy_pRow[0] = resXY;
		}
	}
}
void HarrisDetector::ConvolutionGauss(Mat & srcImg, Mat & dstImg)
{
	float windowSize, constant;
	vector<vector<float> > filter;
	windowSize = gaussWindowSize;
	constant = gaussConstant;
	filter.resize(5, vector<float>(5));
	for (int i = 0; i < windowSize; i++) {
		for (int j = 0; j < windowSize; j++) {
			filter[i][j] = gaussFilter[i][j];
		}
	}


	int Col = srcImg.cols;
	int Row = srcImg.rows;

	//int dstCol = dstImg.cols;
	//int dstRow = dstImg.rows;

	uchar* src_data = (uchar*)srcImg.data;
	uchar* dst_data = (uchar*)dstImg.data;

	int haftWindowSize = windowSize / 2;
	for (int y = 0; y < Row; ++y, dst_data += Col) {
		uchar *pRow = dst_data;
		for (int x = 0; x < Col; ++x, pRow += 1) {
			int yy, xx;
			float res = 0;
			for (int i = -haftWindowSize; i <= haftWindowSize; i++) {
				for (int j = -haftWindowSize; j <= haftWindowSize; j++) {
					yy = y + i;
					xx = x + j;
					if (yy >= 0 && yy < Row && xx >= 0 && xx < Col) {
						res += src_data[yy*Col + xx ] * filter[2 - i][2 - j];
					}
				}
			}
			res *= constant;
			pRow[0] = res;
		}
	}
}

Mat HarrisDetector::Detector()
{
	namedWindow("Display", WINDOW_AUTOSIZE);
	Mat temp = srcImg.clone();
	ConvolutionMatrix();
	MatrixM = Mat(srcRow, srcCol, CV_32FC1);
	ConvolutionGauss();

	int srcCol = srcImg.cols;
	int srcRow = srcImg.rows;
	float k = 0.05;

	int src_stepWidth = srcImg.step[0];
	int src_nchannels = srcImg.step[1];

	float* m_data = (float*)MatrixM.data;
	float* Ix2_data = (float*)ixSquareDerivative.data;
	float* Iy2_data = (float*)iySquareDerivative.data;
	float* IxIy_data = (float*)ixIyDerivative.data;
	uchar* dst_data = (uchar *)harrisImg.data;

	for (int y = 0; y < srcRow; ++y, m_data += srcCol, Ix2_data += srcCol, Iy2_data += srcCol, IxIy_data += srcCol) {
		float* mRow = m_data;
		float* Ix2Row = Ix2_data;
		float* Iy2Row = Iy2_data;
		float* IxIyRow = IxIy_data;
		for (int x = 0; x < srcCol; ++x, mRow += 1, Ix2Row += 1, Iy2Row += 1, IxIyRow += 1) {
			float det, trace, r;
			det = Ix2Row[0] * Iy2Row[0] - IxIyRow[0] * IxIyRow[0];
			trace = Ix2Row[0] + Iy2Row[0];
			r = det - k * trace*trace;
			mRow[0] = r;
		}
	}

	m_data = (float*)MatrixM.data;
	float* value = (float*)MatrixM.data;
	dst_data = (uchar *)harrisImg.data;

	for (int y = 0; y < srcRow; ++y, m_data += srcCol) {
		float* mRow = m_data;
		for (int x = 0; x < srcCol; ++x, mRow += 1) {
			float r = mRow[0];
			bool flag;
			if (r < 100000) {
				flag = false;
			}
			// so sánh cực đại địa phương(lân cận 8);
			else {
				flag = true;
				for (int i = -1; i <= 1; i++) {
					if (flag == false)
						break;
					for (int j = -1; j <= 1; j++) {
						int yy = y + i;
						int xx = x + j;
						if (yy < 0 || yy >= srcCol || xx < 0 || xx >= srcRow)
							continue;
						if (i != 0 && j != 0 && value[(y + i)*srcCol + (x + j)] > r) {
							flag = false;
							break;
						}
					}
				}
			}
			if (flag) {
				drawMarker(harrisImg, cv::Point(x,y), Scalar(0, 0, 255));
			}
		}
	}
	//imwrite("ahihi.png", harrisImg);
	//imshow("Display", harrisImg);
	//waitKey(0);
	return harrisImg;
}

HarrisDetector::HarrisDetector()
{
}

HarrisDetector::HarrisDetector(const Mat& temp)
{
	harrisImg = temp.clone();
	cv::cvtColor(temp, srcImg, cv::COLOR_RGB2GRAY);
	if (srcImg.type() == CV_8UC1) {
		cout << "Yes";
	}
	srcCol = srcImg.cols;
	srcRow = srcImg.rows;
	ixSquareDerivative = Mat(srcRow, srcCol, CV_32FC1);
	iySquareDerivative = Mat(srcRow, srcCol, CV_32FC1);
	ixIyDerivative = Mat(srcRow, srcCol, CV_32FC1);
	//ixDerivative = Mat(srcRow, srcCol, CV_32FC1);
	//iyDerivative = Mat(srcRow, srcCol, CV_32FC1);
	
	tempIx = Mat(srcRow, srcCol, CV_32FC1);
	tempIy = Mat(srcRow, srcCol, CV_32FC1);
	tempIxIy = Mat(srcRow, srcCol, CV_32FC1);
}

HarrisDetector::~HarrisDetector()
{
}


#endif
