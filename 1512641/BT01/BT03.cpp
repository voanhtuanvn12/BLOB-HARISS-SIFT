// BT01.cpp : Defines the entry point for the console application.
//
#include "stdafx.h" 
#include "myLibrary.h"
#include "HarrisDetector.h"
#include "BlobDetector.h"
#include "SiftDetector.h"
#include "Blob.h"
#include "MyBlob.h"

Mat detectHarriss(const Mat& srcImg) {
	Mat res;
	HarrisDetector *harris = new HarrisDetector(srcImg);
	res = harris->Detector();
	return res;
}
Mat detectBlob(const Mat& srcImg) {
	Mat res;
	//BlobDetector *blob = new BlobDetector();
	//res = blob->test(srcImg);
	//MyBlob b = MyBlob(srcImg);
	//res = b.Detect();
	MyBlob b = MyBlob(srcImg, 1.8, 3.0, 0.02);
	b.Detect();
	return res;

}
Mat detectDOG(const Mat& srcImg) {
	SiftDetector *sift = new SiftDetector(srcImg);
	sift->detect();
	return Mat();
}
double matchBySIFT(const Mat& img1, const Mat& img2, int detector) {
	return 0.0;
}

int main(int argc, char **argv) {
	if (argc != 3) {
		cout << "Chuong trinh mo va hien thi anh" << endl;
		return -1;
	}
	
	int malenh = atoi(argv[2]);
	Mat image;
	string imageName;
	imageName = argv[1];
	image = imread(imageName, IMREAD_ANYCOLOR);//2
	namedWindow("Display", WINDOW_AUTOSIZE);
	if (!image.data) {
		cout << "Khong the mo anh" << endl;
		return -1;
	}
	if (malenh == 1) {
		Mat res = detectHarriss(image);
		imshow("Display", res);
		waitKey(0);
	}
	else if (malenh == 2) {
		Mat res = detectBlob(image);
		//imshow("Display", res);
		waitKey(0);
	}
	else if (malenh == 3) {
		cout << " Not completed yet !" << endl;
	}
	else if (malenh == 4) {
		cout << " Not completed yet !" << endl;
	}
	else {
		cout << "Loi tham so";
		return -1;
	}
	
	

	
	image.release();
	destroyAllWindows();
	return 0;
}
