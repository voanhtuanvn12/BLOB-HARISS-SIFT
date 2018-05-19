#include "stdafx.h"
#include "MyBlob.h"

double MyBlob::area(double r)
{
	return r * r*PI;
}

Mat MyBlob::DetectMaximaKeyPoint()
{
	vector<int> scale; // vector is used to store scale level of a maxima point found.
	vector<Point> maximaPoints; // vector is used to store maximum points
	int rows = srcImg.rows;
	int cols = srcImg.cols;

	//we use image with level i and compare its each point with 
	//9 points of level(i-1) and level(i+1) images
	//and 8 points of level-i image (itself)
	// to find maxima
	// if image with level 0 will compare with image level 1 and itself
	// if image with level-last( the last one) will compare with image level(last-1) and itself
	Mat up, down, middle;
	// Tìm số lượng khung ảnh 
	int numScales = (int)((maxThreshold - minThreshold) / delta) + 1;

	//find maxima points
	for (int i = 0; i < numScales; i++)
	{
		middle = loGImg[i];
		if (i != 0) // image with level from 1 to numScale - 1 (the last one)
			down = loGImg[i - 1];
		if (i != numScales - 1)// image with level from 0 to numScale - 2
			up = loGImg[i + 1];

		for (int ri = 1; ri < rows - 1; ri++)
		{
			for (int ci = 1; ci < cols - 1; ci++)
			{
				bool isMax = true;
				bool isMin = true;



				double currentPixel = middle.at<double>(ri, ci);
				//if (currentPixel < 1.f) // threshold
				//	continue;
				// compare among 8 adjacent pixels at current image
				if (!(currentPixel > middle.at<double>(ri - 1, ci) &&
					currentPixel > middle.at<double>(ri + 1, ci) &&
					currentPixel > middle.at<double>(ri, ci - 1) &&
					currentPixel > middle.at<double>(ri, ci + 1) &&
					currentPixel > middle.at<double>(ri - 1, ci - 1) &&
					currentPixel > middle.at<double>(ri - 1, ci + 1) &&
					currentPixel > middle.at<double>(ri + 1, ci + 1) &&
					currentPixel > middle.at<double>(ri + 1, ci - 1)))
				{
					isMax = false;
					//continue;
				}


				if (!(currentPixel < middle.at<double>(ri - 1, ci) &&
					currentPixel < middle.at<double>(ri + 1, ci) &&
					currentPixel < middle.at<double>(ri, ci - 1) &&
					currentPixel < middle.at<double>(ri, ci + 1) &&
					currentPixel < middle.at<double>(ri - 1, ci - 1) &&
					currentPixel < middle.at<double>(ri - 1, ci + 1) &&
					currentPixel < middle.at<double>(ri + 1, ci + 1) &&
					currentPixel < middle.at<double>(ri + 1, ci - 1)))
				{
					isMin = false;
					//continue;
				}


				if (i != numScales - 1)
				{
					// compare among 9 adjacent pixels at up image
					if (!(currentPixel > up.at<double>(ri, ci) &&
						currentPixel > up.at<double>(ri - 1, ci) &&
						currentPixel > up.at<double>(ri + 1, ci) &&
						currentPixel > up.at<double>(ri, ci - 1) &&
						currentPixel > up.at<double>(ri, ci + 1) &&
						currentPixel > up.at<double>(ri - 1, ci - 1) &&
						currentPixel > up.at<double>(ri - 1, ci + 1) &&
						currentPixel > up.at<double>(ri + 1, ci + 1) &&
						currentPixel > up.at<double>(ri + 1, ci - 1)))
					{
						isMax = false;
						//continue;
					}

					if (!(currentPixel < up.at<double>(ri, ci) &&
						currentPixel < up.at<double>(ri - 1, ci) &&
						currentPixel < up.at<double>(ri + 1, ci) &&
						currentPixel < up.at<double>(ri, ci - 1) &&
						currentPixel < up.at<double>(ri, ci + 1) &&
						currentPixel < up.at<double>(ri - 1, ci - 1) &&
						currentPixel < up.at<double>(ri - 1, ci + 1) &&
						currentPixel < up.at<double>(ri + 1, ci + 1) &&
						currentPixel < up.at<double>(ri + 1, ci - 1)))
					{
						isMin = false;
						//continue;
					}
				}
				if (i != 0)
				{
					// compare among 9 adjacent pixels at down image
					if (!(currentPixel > down.at<double>(ri, ci) &&
						currentPixel > down.at<double>(ri - 1, ci) &&
						currentPixel > down.at<double>(ri + 1, ci) &&
						currentPixel > down.at<double>(ri, ci - 1) &&
						currentPixel > down.at<double>(ri, ci + 1) &&
						currentPixel > down.at<double>(ri - 1, ci - 1) &&
						currentPixel > down.at<double>(ri - 1, ci + 1) &&
						currentPixel > down.at<double>(ri + 1, ci + 1) &&
						currentPixel > down.at<double>(ri + 1, ci - 1)))
					{
						isMax = false;
						//continue;
					}

					if (!(currentPixel < down.at<double>(ri, ci) &&
						currentPixel < down.at<double>(ri - 1, ci) &&
						currentPixel < down.at<double>(ri + 1, ci) &&
						currentPixel < down.at<double>(ri, ci - 1) &&
						currentPixel < down.at<double>(ri, ci + 1) &&
						currentPixel < down.at<double>(ri - 1, ci - 1) &&
						currentPixel < down.at<double>(ri - 1, ci + 1) &&
						currentPixel < down.at<double>(ri + 1, ci + 1) &&
						currentPixel < down.at<double>(ri + 1, ci - 1)))
					{
						isMin = false;
						//continue;
					}
				}
				// if it's a maxima
				if (isMax || isMin) {
					maximaPoints.push_back(Point(ci, ri));
					scale.push_back(i); // store its scale level
				}
			}
		}

	}
	for (int i = 0; i < scale.size(); i++)
	{
		Point tmp = maximaPoints[i];
		// the radius of a circle
		// r = sqrt(2) * sigma
		//sigma = exp(tmin + scale[i] * tdelta)
		double r = sqrt(2.0f) * exp(minThreshold + scale[i] * delta);
		circle(srcImg, tmp, r, Scalar(0, 0, 255),1);
	}
	namedWindow("Display", WINDOW_AUTOSIZE);
	imshow("Display", srcImg);
	waitKey(0);
	return srcImg;
}

Mat MyBlob::Laplace(const Mat & srcImg) //CV_64FC1
{
	double filter[5][5] = { {1,1,1,1,1}, { 1,1,1,1,1 }, { 1,1,-24,1,1 }, { 1,1,1,1,1 }, { 1,1,1,1,1 }
};
	int Row = srcImg.rows;
	int Col = srcImg.cols;
	Mat dstImg = Mat(Row, Col, CV_64FC1);

	int src_stepWidth = srcImg.step[0];
	int src_nchannels = srcImg.step[1];


	double * src_data = (double*)srcImg.data;
	double * dst_data = (double*)dstImg.data;
	for (int y = 0; y < Row; ++y, dst_data += Col) {
		double *pRow = dst_data;
		for (int x = 0; x < Col; ++x, pRow += 1) {
			int yy, xx;
			double res = 0;
			for (int i = -2; i <= 2; i++) {
				for (int j = -2; j <= 2; j++) {
					yy = y + i;
					xx = x + j;
					if (yy >= 0 && yy < Row && xx >= 0 && xx < Col) {
						res += (double)src_data[yy*Col + xx] * filter[2 - i][2 - j];
					}
				}
			}
			pRow[0] = res;
		}
	}

	//dstImg.convertTo(dstImg, CV_8UC1);
	return dstImg;
}

MyBlob::MyBlob()
{
}

MyBlob::MyBlob(const Mat& image, double _minThreshold, double _maxThreshold, double _delta)
{
	srcImg = image.clone();
	maxThreshold = _maxThreshold;
	minThreshold = _minThreshold;
	delta = _delta;
}

MyBlob::~MyBlob()
{
}

Mat MyBlob::Detect()
{
	Mat grayScaleImg;// = convertRgbToGrayscale(m_srcImage);
	cv::cvtColor(srcImg, grayScaleImg, cv::COLOR_RGB2GRAY);
	grayScaleImg.convertTo(grayScaleImg, CV_64FC1);

	// Chuẩn hóa về 0...1
	for (int r = 0; r < grayScaleImg.rows; r++)
	{
		for (int c = 0; c < grayScaleImg.cols; c++)
		{
			grayScaleImg.at<double>(r, c) = grayScaleImg.at<double>(r, c) / 255.0f;
		}
	}

	// Tìm số lượng khung ảnh 
	int numScales = (int)((maxThreshold - minThreshold) / delta) + 1;

	// Tạo khung ảnh đầu tiên
	double t = minThreshold;
	double sigma = exp(t); // Tìm sigma
	Mat GaussImg;
	GaussianBlur(grayScaleImg, GaussImg, Size(0, 0), sigma); // Làm mờ ảnh với hệ số sigma
	loGImg.push_back(GaussImg.clone()); // push_back the level-1 image
	t += delta;

	// Tạo các khung ảnh còn lại

	for (int i = 1; i < numScales; i++) {
		// Dùng ảnh trước đó để tiếp tục làm mờ
		double sigma_now = exp(t);

		double sigma_new = sqrt(sigma_now*sigma_now - sigma * sigma);
		GaussianBlur(loGImg[i - 1], GaussImg, Size(0, 0), sigma_new); // Làm mờ ảnh
		loGImg.push_back(GaussImg.clone());


		// Cập nhật sigma và t
		sigma = sigma_now;
		t += delta;
	}
	// Nhân các ảnh với hệ số sigma^2
	t = minThreshold;
	for (int i = 0; i < numScales; i++)
	{
		loGImg[i] *= exp(t)*exp(t);
		t += delta;
	}
	// áp bộ lọc laplace
	for (int i = 0; i < numScales; i++)
	{
		loGImg[i] = Laplace(loGImg[i]);
	}
	// Tìm các keypoint
	return DetectMaximaKeyPoint();
}
