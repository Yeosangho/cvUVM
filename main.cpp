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
	float* h_img;
	int rows = 1280;
	int cols = 720;

	cuMallocManaged(&h_img, rows, cols);

	Mat hmat;
	hmat = Mat(cvSize(cols, rows), CV_32F,  h_img);
	Mat result;
	hmat = cv::imread("../720p.jpg");
	//cout << hmat << endl;
	gpu::GpuMat dmat(cvSize(rows, cols), CV_32F, h_img);
	cuDeviceSynchronize();
	double timeSec = 0;
	int64 startUVM = getTickCount();
	for(int i=0; i<100; i++){
		h_img++;
		//cout << *h_img	<< endl;
		//cout << dmat.channels() << endl;	
		timeSec += (getTickCount() - startUVM) / getTickFrequency();
	}
	dmat.download(hmat);
	cuDeviceSynchronize();
	//cv::imshow("test", hmat);
	cout << timeSec << endl;
	
	float* h_img2 = (float*)malloc(sizeof(float)*rows*cols);
	float* d_img2;
	
	cuMalloc((void **)&d_img2, rows, cols);
	
	Mat hmat2(cvSize(rows, cols), CV_32F, h_img2);
	
	gpu::GpuMat dmat2(cvSize(rows, cols), CV_32F, d_img2);
	timeSec = 0;
	startUVM = getTickCount();
	for(int i=0; i<100; i++)
	{
		dmat2.upload(hmat2);
		dmat2.download(hmat2);
		timeSec += (getTickCount() - startUVM) / getTickFrequency();
	}
	cout << timeSec << endl;
	//https://stackoverflow.com/questions/14361322/accessing-image-pixels-as-float-array
	VideoCapture cap("../NORWAY720P.mp4");
    if(!cap.isOpened())  // check if we succeeded
        return -1;
	float* frameMem;
	cuMallocManaged(&frameMem, rows, cols);
	Mat frame2;
	Mat frame(cvSize(cols, rows), CV_32F,  frameMem);
	Mat result2;
	gpu::GpuMat dframe(cvSize(cols, rows), CV_32F, frameMem);
	for(;;){
		cap >> frame2;
		frameMem = (float *)frame2.data;
		cout << *frameMem << endl;
		dframe.download(result2);
		cuDeviceSynchronize(); 
		cv::imshow("test", result2);
		 
		waitKey(1); 
	}

	//cout << dmat << endl;
	return 0;
}
