#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/core/cuda.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "memoryAccessor.h"
using namespace cv;
using namespace std;
using namespace std;
int main(int argc, char** argv){
	

	cuSetDeviceFlags();
	float *h_img;
	int rows = 1280;
	int cols = 720;
	Mat mat = cv::imread("../720p.jpg", CV_32FC1);

	
	cuMallocManaged((void **)&h_img, rows, cols);
	mat.convertTo(mat,CV_32FC1);
	h_img = (float *)mat.data;
	cout << "1" << endl;
	cout << h_img[0] << endl;
	cout << "1" << endl;
	Mat hmat;
	for(int i=0; i<rows*cols; i++){
			//cout << h_img[i] << endl;
			h_img[i] = 0;
	}
	cout << "1" << endl;	
	hmat = Mat(cvSize(rows, cols),  CV_32FC1,  h_img, 0);
	cout << "1" << endl;
	//cout << hmat << endl;

	Mat result;
	//hmat = cv::imread("../720p.jpg");

	//cout << h_img[0] << endl;
	cout << "1" << endl;	
	cuda::GpuMat dmat(cvSize(rows, cols), CV_32FC1, h_img, 0);
	cuda::GpuMat d_dst;
	for(int i=0; i<rows*cols; i++){
			h_img[i] = 125;
	}
	cuDeviceSynchronize();
	double timeSec = 0;
	int64 startUVM = getTickCount();
	for(int i=0; i<100; i++){
		h_img++;
		//cout << *h_img	<< endl;
		//cout << dmat.channels() << endl;	
		timeSec += (getTickCount() - startUVM) / getTickFrequency();
	}
	//cv::cuda::bilateralFilter( dmat, dmat, -1, 50, 7 );
    Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector( 35.0, 200.0 );
    dmat.convertTo(d_dst, CV_8UC1);
    canny->detect( d_dst, d_dst );
 	
	cout << "1" << endl;
	result = Mat(d_dst);
	cuDeviceSynchronize();
	cv::imshow("test", result);
	waitKey(0);
	return 0;
}
