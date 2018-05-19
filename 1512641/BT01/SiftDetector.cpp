#include "stdafx.h"
#include "SiftDetector.h"

SiftDetector::SiftDetector(const Mat & srcImg)
{
	Mat grayImg;
	cv::cvtColor(srcImg, grayImg, cv::COLOR_RGB2GRAY);
	resize(grayImg, octaves[0], Size(), 2,2, INTER_NEAREST);
	octaves[1] = grayImg.clone();
	resize(grayImg, octaves[2], Size(), 0.5, 0.5, INTER_NEAREST);
}

SiftDetector::SiftDetector()
{
}

SiftDetector::~SiftDetector()
{
}

Mat SiftDetector::convolution(const Mat & srcImg, double filter[][5], double m)
{

	int nrows = srcImg.rows;
	int ncols = srcImg.cols;
	int src_stepWidth = srcImg.step[0];
	int src_nchannels = srcImg.step[1];
	Mat dstImg = Mat(nrows, ncols, CV_32FC1);

	uchar* src_data = (uchar*)srcImg.data;
	float * dst_data = (float*)dstImg.data;
	for (int y = 0; y < nrows; ++y, dst_data += ncols) {
		float *pRow = dst_data;
		for (int x = 0; x < ncols; ++x, pRow += 1) {
			int yy, xx;
			float res = 0;
			for (int i = -2; i <= 2; i++) {
				for (int j = -2; j <= 2; j++) {
					yy = y + i;
					xx = x + j;
					if (yy >= 0 && yy < nrows && xx >= 0 && xx < ncols) {
						res += (float)src_data[yy*src_stepWidth + xx * src_nchannels] * filter[2 - i][2 - j];
					}
				}
			}
			res *= m;
			pRow[0] = res;
		}
	}
	return dstImg;
}

Mat SiftDetector::difference(const Mat & m1, const Mat & m2)
{
	int nrows = m1.rows;
	int ncols = m1.cols;
	Mat dstImg = Mat(nrows, ncols, CV_32FC1);

	float* src1_data = (float*)m1.data;
	float* src2_data = (float*)m2.data;
	float * dst_data = (float*)dstImg.data;
	for (int y = 0; y < nrows; ++y, dst_data += ncols, src1_data+= ncols, src2_data+= ncols) {
		float *pRow = dst_data;
		float *pRow_src1 = src1_data;
		float *pRow_src2 = src2_data;
		for (int x = 0; x < ncols; ++x, pRow += 1, pRow_src1 += 1, pRow_src2+=1) {
			pRow[0] = pRow_src1[0] - pRow_src2[0];
		}
	}
	return dstImg;
}

double SiftDetector::generateGaussFilter(double kernel[][5], double sigma)
{
	
	double r, s = 2.0 * sigma * sigma;
	double m = 1e9;
	for (int x = -2; x <= 2; ++x) {
		for (int y = -2; y <= 2; ++y) {
			r = (x * x + y * y)*1.0;
			kernel[x + 2][y + 2] =
				((exp(-r / s)) / (3.14 * s));
			//m = min(kernel[x + 2][y + 2], m);
		}
	}
	//m /= 10;
	//for (int i = 0; i < 5; i++) {
	//	for (int j = 0; j < 5; j++) {
	//		kernel[i][j] =(int)round(kernel[i][j] / m);
	//	}
	//}
	return 1;
}

void SiftDetector::CalcDoGImage()
{
	// với mỗi octaves
	for (int oc = 0; oc < 3; oc++) {
		// với mỗi ảnh trong từng octaves
		for (int im = 4; im >= 1; im--) {
			doGImage[oc][4 - im] = difference(blurImage[oc][im], blurImage[oc][im - 1]);
		}
	}
}

void SiftDetector::CalcBlurImage()
{
	double sig = sigma;
	double kernel[5][5][5];
	double m[5];
	for (int i = 0; i < 5; i++) {
		m[i] = generateGaussFilter(kernel[i], sig);
		sig *= k;
		cout << "==========================================\n";
		for (int j = 0; j < 5; j++) {
			for (int k = 0; k < 5; k++) {
				cout << kernel[i][j][k] << " ";
			}
			cout << endl;
		}
	}

	cvNamedWindow("Display");
	// với mỗi octaves
	for (int oc = 0; oc < 3; oc++) {
		// với mỗi ảnh trong octave
		for (int im = 0; im < 5; im++) {
			imshow("Display", octaves[oc]);
			waitKey(0);
			// tích chập từng ảnh trong octave với bộ lọc gauss tương ứng
			blurImage[oc][im] = convolution(octaves[oc], kernel[im], m[im]);
			//Mat temp;
			//temp = blurImage[oc][im].clone();
			//temp.convertTo(temp, CV_8UC1);
			//imshow("Display", temp);
			waitKey(0);
		}
	}
}

void SiftDetector::detect()
{
	CalcBlurImage();
	CalcDoGImage();

}
