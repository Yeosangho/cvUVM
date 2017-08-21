#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/gpu/gpu.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "memoryAccessor.h"
using namespace cv;
using namespace std;
using namespace std;
int main(int argc, char** argv){
	

	cuSetDeviceFlags();
	unsigned char *h_img;
	unsigned char *h_img2;
	unsigned char *h_img3;

	int rows = 720;
	int cols = 1280;
	Mat mat = cv::imread("../720p.jpg", CV_8UC1);

	
	cuMallocManaged((void **)&h_img, rows, cols);
	cuMallocManaged((void **)&h_img2, rows, cols);
	cuMallocManaged((void **)&h_img3, rows, cols);
	
	//h_img = mat.data;
	for(int i=0; i<rows*cols; i++){
		h_img[i] = mat.data[i];
	}	
	cout << "1" << endl;
	cout << h_img[0] << endl;
	cout << "1" << endl;
	Mat hmat;
	for(int i=0; i<rows*cols; i++){
			//cout << h_img[i] << endl;
			//h_img[i] = 0;
	}
	cout << "1" << endl;	
	hmat = Mat(cvSize(cols, rows),  CV_8UC1,  h_img);
	Mat hmat2 = Mat(cvSize(cols, rows),  CV_8UC1,  h_img2);
	Mat hmat3 = Mat(cvSize(cols, rows),  CV_8UC1,  h_img3);	
	cout << "1" << endl;
	//cout << hmat << endl;

	Mat result;
	//hmat = cv::imread("../720p.jpg");

	//cout << h_img[0] << endl;
	cout << "1" << endl;	
	gpu::GpuMat dmat(cvSize(cols, rows), CV_8UC1, h_img);
	gpu::GpuMat dmat2(cvSize(cols, rows), CV_8UC1, h_img2);
	gpu::GpuMat dmat3(cvSize(cols, rows), CV_8UC1, h_img3);	
	for(int i=0; i<rows*cols; i++){
			//h_img[i] = 125;
	}
	double timeSec = 0;
	int64 startUVM = getTickCount();
	for(int i=0; i<100; i++){
		h_img++;
		//cout << *h_img	<< endl;
		//cout << dmat.channels() << endl;	
		timeSec += (getTickCount() - startUVM) / getTickFrequency();
	}
	gpu::multiply(dmat, dmat, dmat3);

 	
	cv::imshow("test", hmat3);
	waitKey(0);
	return 0;
}
