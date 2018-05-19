#ifndef  _Tuan_Kyou_

#define _Tuan_Kyou_
#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <cstdio>
using namespace cv;
using namespace std;
#define RED_CHANNEL 2
#define GREEN_CHANNEL 3
#define BLUE_CHANNEL 4

// tunable parameter for the Harris
constexpr float k_param = 0.04f;   // k parameter (usually 0.02 - 0.04)
constexpr float threshold = 0.5f;  // threhold parameter
constexpr int windowSize = 7;      // window size for non-maximal suppresion
constexpr int halfWindowSize = windowSize / 2;  // half window size

// non-maximal suppresion, supress all values which are not the maximum in a
// neighbourhood
struct NonMaximalSuppresion {
	template <typename T>
	float operator()(const T &im) {
		float currentPixel{ im.at(im.I_c, im.I_r) };
		for (int i = -halfWindowSize; i <= halfWindowSize; i++) {
			for (int j = -halfWindowSize; j <= halfWindowSize; j++) {
				if (currentPixel < im.at(im.I_c + i, im.I_r + j)) {
					return 0.0f;
				}
			}
		}
		return currentPixel;
	}
};


// apply threshold operation to the image
struct Thresh {
	template <typename T, typename Thresh>
	float operator()(const T &t, const Thresh &thresh) {
		return t > thresh ? 1.0f : 0.0f;
	}
};

// Convert from float to unsigned char one channel
struct FloatToU8C1 {
	uchar operator()(const float &t) {
		return uchar(static_cast<unsigned char>(t * 255));
	}
};

// convolution for a custom filter in a one-dimensional image
struct Filter2D {
	template <typename T1, typename T2>
	float operator()(const T1 &nbr, const T2 &fltr) {
		int hs_c = (fltr.cols / 2);
		int hs_r = (fltr.rows / 2);

		float out = 0;
		for (int i2 = -hs_c, i = 0; i2 <= hs_c; i2++, i++)
			for (int j2 = -hs_r, j = 0; j2 <= hs_r; j2++, j++)
				out += (nbr.at(nbr.I_c + i2, nbr.I_r + j2) * fltr.at(i, j));
		return out;
	}
};



// operator created to perform a power of 2 of the image
struct PowerOf2 {
	template <typename T>
	float operator()(T t) {
		return t * t;
	}
};

// operator for element-wise multiplication between two images
struct Mul {
	template <typename T1, typename T2>
	float operator()(T1 t1, T2 t2) {
		return t1 * t2;
	}
};

// operator to add two images
struct Add {
	template <typename T1, typename T2>
	float operator()(T1 t1, T2 t2) {
		return t1 + t2;
	}
};

// operator to subtract two images
struct Sub {
	template <typename T1, typename T2>
	float operator()(T1 t1, T2 t2) {
		return t1 - t2;
	}
};




// tinh dao ham anh theo huong x	
// kerneals
double Wx[3][3] = { { 1, 0, -1 },
{ 1, 0, -1 },
{ 1, 0, -1 } };
double Wy[3][3] = { { -1, -1, -1 },
{ 0, 0, 0 },
{ 1, 1, 1 } };

int rgbToGray(Mat &srcImg, Mat &dstImg) {
	// Mục tiêu dùng để chuyển đổi ảnh màu sang ảnh grayscale
	// input
	// srcImg  : Ảnh gốc
	// dstImg : Ảnh sau khi chuyển
	// output
	// Trả về 1 nếu ảnh được chuyển về ảnh grayscale thành công,  trả về 0
	// nếu ảnh đã là ảnh grayscale
	CV_Assert(srcImg.type() == CV_8UC3); // Kiểm tra xem có phải ảnh màu không
	if (srcImg.type() != CV_8UC3) {
		return -1;
	}
	if (srcImg.type() == CV_8UC1) {
		dstImg = srcImg;
		return 1;
	}
	int rows = srcImg.rows, cols = srcImg.cols;
	dstImg.create(srcImg.size(), CV_8UC1); // Tạo một biến lưu ảnh grayscale
	if (srcImg.isContinuous() && dstImg.isContinuous()) // kiểm tra xem cả 2 ảnh có lưu data dưới dạng liên tục hay không
	{
		cols = rows * cols;
		rows = 1;
	}
	for (int row = 0; row < rows; row++)
	{
		const uchar* src_ptr = srcImg.ptr<uchar>(row); // trả về con trỏ mảng
		uchar* dst_ptr = dstImg.ptr<uchar>(row);

		for (int col = 0; col < cols; col++)
		{
			dst_ptr[col] = (uchar)(src_ptr[0] * 0.114f + src_ptr[1] * 0.587f + src_ptr[2] * 0.299f);
			src_ptr += 3;
		}
	}
	return 1;
}	

int calcHistgogram(Mat srcImg, vector<double> &hists) {
	// Mục tiêu : Tính histogram của một ảnh màu hoặc xám
	// Nếu là ảnh màu thì chuyển về ảnh xám và tính histogram
	// input:
	// srcImg : Ảnh gốc
	// vector<double> &hists : histogram trả về 
	// Trả về 1 nếu thành công, 0 nếu thất bại
	Mat grayImg;
	rgbToGray(srcImg, grayImg); // Chuyển về ảnh grayscale
	hists.resize(256); // khởi tạo giá trị histogram
	int rows = grayImg.rows, cols = grayImg.cols; 
	if (grayImg.isContinuous()) { // nếu mảng data được lưu liên tục
		cols *= rows;
		rows = 1;
	}
	const uchar* gray_ptr = grayImg.ptr<uchar>(0); // con trỏ đến data 0
	for (int i = 0; i < cols; i++) {
		hists[gray_ptr[i]]++; // Đến số lượng pixel có giá trị tương ứng lên 1 nếu có
	}
	return 1;
}

void printHistogram(vector<double> hists) {
	// Mục tiêu : in ra vector histogram
	size_t n = hists.size();
	for (int i = 0; i < n; i++) {
		cout << hists[i] << " ";
	}
	cout << endl;
}


double compareHist(Mat &srcImg1, Mat &srcImg2) {
	// So sánh 2 ảnh dựa vào histogram
	// dựa trên khoảng cách euclid
	// công thức được tham khảo trên 
	double ret = 0;
	vector<double> hist1, hist2;
	calcHistgogram(srcImg1, hist1);
	calcHistgogram(srcImg2, hist2);
	int n = hist1.size();
	for (int i = 0; i < n; i++) {
		ret += pow(hist1[i] - hist2[i],2);
	}

	return pow(ret,0.5);
	//return 0;
}

bool isValidBin(int bin) {
	if (256 % bin == 0 && 1 <= bin && bin <= 256) {
		return true;
	}
	return false;
}
//Tính histogram lượng hóa màu của ảnh RGB
// các mã bin phải là ước số của 256 và không âm
vector < double> calcHistogram(Mat srcImg, int rBin, int gBin, int bBin) {
	vector<double> ret;
	// Trả về số lượng giá trị màu trong 1 bin
	if (!isValidBin(rBin) || !isValidBin(gBin) || !isValidBin(bBin))
		return ret;
	bBin = 256 / bBin;
	gBin = 256 / gBin;
	rBin = 256 / rBin;

	

	vector <double> bhist, ghist, rhist;
	bhist.resize(256);
	ghist.resize(256);
	rhist.resize(256);

	CV_Assert(srcImg.type() == CV_8UC3); // Kiểm tra xem có phải ảnh màu không
	if (srcImg.type() != CV_8UC3) {
		return ret;
	}
	int rows = srcImg.rows, cols = srcImg.cols;

	if (srcImg.isContinuous()) // kiểm tra xem ảnh có lưu data dưới dạng liên tục hay không
	{
		cols = rows * cols;
		rows = 1;
	}
	for (int row = 0; row < rows; row++)
	{
		const uchar* src_ptr = srcImg.ptr<uchar>(row); // trả về con trỏ mảng

		for (int col = 0; col < cols; col++)
		{
			bhist[(uchar)src_ptr[0]]++;
			ghist[(uchar)src_ptr[1]]++;
			rhist[(uchar)src_ptr[2]]++;
			src_ptr += 3;
		}
	}
	// Gom thành từng bin	

	

	int toBin = 0;
	int c = 0;
	int index = 0;
	while (index < 256) {
		c += bhist[index];
		index++;
		toBin++;
		if (toBin == bBin - 1) {
			ret.push_back(c);
			toBin = 0;
			c = 0;
		}
	}
	toBin = 0;
	index = 0;
	while (index < 256) {
		c += ghist[index];
		index++;
		toBin++;
		if (toBin == gBin - 1) {
			ret.push_back(c);
			toBin = 0;
			c = 0;
		}
	}
	toBin = 0;
	index = 0;
	while (index < 256) {
		c += rhist[index];
		index++;
		toBin++;
		if (toBin == rBin - 1) {
			ret.push_back(c);
			toBin = 0;
			c = 0;
		}
	}
	return ret;
}


//Tính histogram lượng hóa màu của ảnh xám
vector < double> calcHistogram(Mat srcImg, int gBin) {
	vector<double> ret;

	// lấy số lượng mã màu trong 1 bin
	if (!isValidBin(gBin))
		return ret;
	gBin = 256 / gBin;

	

	vector <double> ghist;
	ghist.resize(256);

	Mat grayscale; // Chuyển thành ảnh grayscale nếu ảnh là ảnh màu
	rgbToGray(srcImg, grayscale);

	int rows = grayscale.rows, cols = grayscale.cols;

	if (grayscale.isContinuous()) // kiểm tra xem ảnh có lưu data dưới dạng liên tục hay không
	{
		cols = rows * cols;
		rows = 1;
	}
	for (int row = 0; row < rows; row++)
	{
		const uchar* src_ptr = grayscale.ptr<uchar>(row); // trả về con trỏ mảng

		for (int col = 0; col < cols; col++)
		{
			ghist[(uchar)src_ptr[0]]++;
			src_ptr ++;
		}
	}
	// Gom thành từng bin	

	int toBin = 0;
	int c = 0;
	int index = 0;
	while (index < 256) {
		c += ghist[index];
		index++;
		toBin++;
		if (toBin == gBin - 1) {
			ret.push_back(c);
			toBin = 0;
			c = 0;
		}
	}
	return ret;
}


//So sánh 2 ảnh dựa vào lược đồ lượng hóa ảnh màu

double compareHist1( Mat srcImg1, Mat srcImg2, int rBin, int gBin, int bBin ) {
	vector<double> hist1 = calcHistogram(srcImg1, rBin, gBin, bBin);
	vector<double> hist2 = calcHistogram(srcImg2, rBin, gBin, bBin);
	double ret = 0;
	int n = hist1.size();
	for (int i = 0; i < n; i++) {
		ret += pow(hist1[i] - hist2[i],2);
	}
	return pow(ret, 0.5);
}


double compareHist1(Mat srcImg1, Mat srcImg2, int gBin) {
	vector<double> hist1 = calcHistogram(srcImg1, gBin);
	vector<double> hist2 = calcHistogram(srcImg2, gBin);
	double ret = 0;
	int n = hist1.size();
	for (int i = 0; i < n; i++) {
		ret += pow(hist1[i] - hist2[i], 2);
	}
	return pow(ret, 0.5);
}


int xGrad(Mat image, int x, int y)
{
	int sum = 0;
	for (int i = -1; i <= 1; ++i)
		for (int j = -1; j <= 1; ++j) {
			sum += image.at<uchar>(y + i, x + j)*Wx[1 - i][1 - j];
		}
	return sum;
}

// Computes the y component of the gradient vector
// at a given point in a image
// returns gradient in the y direction

int yGrad(Mat image, int x, int y)
{
	int sum = 0;
	for (int i = -1; i <= 1; ++i)
		for (int j = -1; j <= 1; ++j) {
			sum += image.at<uchar>(y + i, x + j)*Wy[1 - i][1 - j];
		}
	return sum;
}



Mat xGradient(const Mat &src, double k) {
	double coeff = 1 / (k + 2); //hệ số nhân ma trận
	// số k ứng với các bộ lọc khác nhau 
	// k = 1 : Prewitt
	// k = 2 : Sobel
	// k = sqrt(2) pei-chen
	int w = src.cols;
	int h = src.rows;



	Wx[1][0] = k; Wx[1][2] = k * (-1);
	Mat dstImg = src.clone();

	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
			dstImg.at<uchar>(y, x) = 0.0;

	for (int y = 1; y < h - 1; ++y) {
		for (int x = 0; x < w - 1; ++x) {
			int gx = xGrad(src, x, y);
			gx = gx > 255 ? 255 : gx;
			gx = gx < 0 ? 0 : gx;
			dstImg.at<uchar>(y, x) = gx * coeff;
		}
	}
	return dstImg;




}

// tinh dao ham theo huong y
Mat yGradient(const Mat &src, double k) {
	double coeff = 1 / (k + 2); //hệ số nhân ma trận
								// số k ứng với các bộ lọc khác nhau 
								// k = 1 : Prewitt
								// k = 2 : Sobel
								// k = sqrt(2) pei-chen
	int w = src.cols;
	int h = src.rows;

	Wy[0][1] = k * (-1); Wy[2][1] = k;
	Mat dstImg = src.clone();

	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
			dstImg.at<uchar>(y, x) = 0.0;

	for (int y = 1; y < h - 1; ++y) {
		for (int x = 1; x < w - 1; ++x) {
			int gy = yGrad(src, x, y)*coeff;
			gy = gy > 255 ? 255 : gy;
			gy = gy < 0 ? 0 : gy;
			dstImg.at<uchar>(y, x) = gy;
		}
	}
	return dstImg;
}


// tinh dao ham hai huong theo magnitude
Mat magnitude(const Mat &src, double k) {
	double coeff = 1 / (k + 2); //hệ số nhân ma trận
								// số k ứng với các bộ lọc khác nhau 
								// k = 1 : Prewitt
								// k = 2 : Sobel
								// k = sqrt(2) pei-chen

	Wy[0][1] = k * (-1); Wy[2][1] = k;
	Wx[1][0] = k; Wx[1][2] = k * (-1);

	int h = src.rows, w = src.cols;
	Mat dstImg = src.clone();

	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
			dstImg.at<uchar>(y, x) = 0.0;
	for (int y = 1; y < h - 1; y++) {
		for (int x = 1; x < w - 1; x++) {
			int gx = xGrad(src, x, y);
			int gy = yGrad(src, x, y);
			int	sum = sqrt(pow(gx, 2) + pow(gy, 2)) / 4;
			sum = sum > 255 ? 255 : sum;
			sum = sum < 0 ? 0 : sum;
			dstImg.at<uchar>(y, x) = sum;
		}
	}
	return dstImg;
}



#endif