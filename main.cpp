
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "utils.cpp"
#include <opencv2/imgproc.hpp>
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/core/cuda.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>
#include "memoryAccessor.h"

using namespace cv;
using namespace std;
//================================
void LoadJPEG(char* FileName, unsigned char ** buffer)
//================================
{
  unsigned long x, y;
  unsigned int texture_id;
  unsigned long data_size;     // length of the file
  int channels;               //  3 =>RGB   4 =>RGBA 
  unsigned int type;  
  unsigned char * rowptr[1];    // pointer to an array
  unsigned char * jdata;        // data for the image
  struct jpeg_decompress_struct info; //for our jpeg info
  struct jpeg_error_mgr err;          //the error handler

  FILE* file = fopen(FileName, "rb");  //open the file

  info.err = jpeg_std_error(& err);     
  jpeg_create_decompress(& info);   //fills info structure

  //if the jpeg file doesn't load
  if(!file) {
     fprintf(stderr, "Error reading JPEG file %s!", FileName);
  }

  jpeg_stdio_src(&info, file);    
  jpeg_read_header(&info, FALSE);   // read jpeg file header

  jpeg_start_decompress(&info);    // decompress the file

  //set width and height
  x = info.output_width;
  y = info.output_height;
  channels = info.num_components;

  data_size = x * y * 3;

  //--------------------------------------------
  // read scanlines one at a time & put bytes 
  //    in jdata[] array. Assumes an RGB image
  //--------------------------------------------
 
  cuMallocManaged((void **)(buffer), data_size, 1);
  while (info.output_scanline < info.output_height) // loop
  {
    // Enable jpeg_read_scanlines() to fill our jdata array
    rowptr[0] = (unsigned char *)*buffer +  // secret to method
            3* info.output_width * info.output_scanline; 

    jpeg_read_scanlines(&info, rowptr, 1);

  }
  //cout << *buffer << endl;
  //---------------------------------------------------

  jpeg_finish_decompress(&info);   //finish decompressing

  //----- create OpenGL tex map (omit if not needed) --------

  jpeg_destroy_decompress(&info);
  fclose(file);                    //close the file

}




int main(int argc, char** argv){
	

	cuSetDeviceFlags();
	unsigned char *h_img2;
	unsigned char *h_img3;
	unsigned char *src;
	int rows = 720;
	int cols = 1280;
	
	LoadJPEG("../720p.jpg", &src);
	

	cuMallocManaged((void **)&h_img2, rows, cols);
	cuMallocManaged((void **)&h_img3, rows, cols);


	Mat hmat = Mat(cvSize(cols, rows),  CV_8UC3,  src);
	Mat hmat2 = Mat(cvSize(cols, rows),  CV_8UC1,  h_img2);
	Mat hmat3 = Mat(cvSize(cols, rows),  CV_8UC1,  h_img3);	


	cuda::GpuMat dmat(cvSize(cols, rows), CV_8UC3, src);
	cuda::GpuMat dmat2(cvSize(cols, rows), CV_8UC1, h_img2);
	cuda::GpuMat dmat3(cvSize(cols, rows), CV_8UC1, h_img3);
	cuda::GpuMat dmat4(cvSize(cols, rows), CV_8UC3);
	cuda::GpuMat dmat5(cvSize(cols, rows), CV_8UC1);
	cuda::GpuMat dmat6(cvSize(cols, rows), CV_8UC1);
	int64 startUVM;
	double timeSec = 0;
	startUVM = getTickCount();
	for(int i=0; i<110; i++){
		cv::cuda::cvtColor(dmat, dmat2, CV_BGR2GRAY);
		Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector( 35.0, 200.0 );
		canny->detect( dmat2, dmat3 );
		if(i >= 10){
			if(i == 10)
				startUVM = getTickCount();
			timeSec += (getTickCount() - startUVM) / getTickFrequency();
		}			
	}
    std::cout << "uvm 720p"<< endl;
    std::cout << "whole tme "<< timeSec/1000 << endl;
    std::cout << "average" << timeSec/100000 << endl;
	timeSec = 0;

	
	startUVM = getTickCount();
	
	for(int i=0; i<100; i++){
		dmat4.upload(hmat);
		cv::cuda::cvtColor(dmat4, dmat5, CV_BGR2GRAY);
		Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector( 35.0, 200.0 );
		canny->detect( dmat5, dmat6 );
		dmat6.download(hmat3);
		timeSec += (getTickCount() - startUVM) / getTickFrequency();			
	}
    std::cout << "original 720p"<< endl;
    std::cout << "whole tme "<< timeSec/1000 << endl;
    std::cout << "average" << timeSec/100000 << endl;
//////////////////////////////////////////////////1080p/////////////////////

    LoadJPEG("../1080p.jpg", &src);
	rows = 1080;
	cols = 1920;    
    cuMallocManaged((void **)&h_img2, rows, cols);
	cuMallocManaged((void **)&h_img3, rows, cols);
    hmat = Mat(cvSize(cols, rows),  CV_8UC3,  src);
	hmat2 = Mat(cvSize(cols, rows),  CV_8UC1,  h_img2);
	hmat3 = Mat(cvSize(cols, rows),  CV_8UC1,  h_img3);	


	dmat = cuda::GpuMat(cvSize(cols, rows), CV_8UC3, src);
	dmat2 = cuda::GpuMat(cvSize(cols, rows), CV_8UC1, h_img2);
	dmat3 = cuda::GpuMat(cvSize(cols, rows), CV_8UC1, h_img3);
	dmat4 = cuda::GpuMat(cvSize(cols, rows), CV_8UC3);
	dmat5 = cuda::GpuMat(cvSize(cols, rows), CV_8UC1);
	dmat6 = cuda::GpuMat(cvSize(cols, rows), CV_8UC1);    
    
	timeSec = 0;
	startUVM = getTickCount();
	for(int i=0; i<110; i++){
		cv::cuda::cvtColor(dmat, dmat2, CV_BGR2GRAY);
		Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector( 35.0, 200.0 );
		canny->detect( dmat2, dmat3 );
		if(i >= 10){
			if(i == 10)
				startUVM = getTickCount();
			timeSec += (getTickCount() - startUVM) / getTickFrequency();
		}			
	}
    std::cout << "uvm 1080p"<< endl;
    std::cout << "whole tme "<< timeSec/1000 << endl;
    std::cout << "average" << timeSec/100000 << endl;
	timeSec = 0;


	startUVM = getTickCount();
	
	for(int i=0; i<100; i++){
		dmat4.upload(hmat);
		cv::cuda::cvtColor(dmat4, dmat5, CV_BGR2GRAY);
		Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector( 35.0, 200.0 );
		canny->detect( dmat5, dmat6 );
		dmat6.download(hmat3);
		timeSec += (getTickCount() - startUVM) / getTickFrequency();			
	}
    std::cout << "original 1080p"<< endl;
    std::cout << "whole tme "<< timeSec/1000 << endl;
    std::cout << "average" << timeSec/100000 << endl;
    
   //////////////////////////////////4k///////////////////////////////////
    LoadJPEG("../4k.jpg", &src);
	rows = 2160;
	cols = 3840;    
    cuMallocManaged((void **)&h_img2, rows, cols);
	cuMallocManaged((void **)&h_img3, rows, cols);
    hmat = Mat(cvSize(cols, rows),  CV_8UC3,  src);
	hmat2 = Mat(cvSize(cols, rows),  CV_8UC1,  h_img2);
	hmat3 = Mat(cvSize(cols, rows),  CV_8UC1,  h_img3);	


	dmat = cuda::GpuMat(cvSize(cols, rows), CV_8UC3, src);
	dmat2 = cuda::GpuMat(cvSize(cols, rows), CV_8UC1, h_img2);
	dmat3 = cuda::GpuMat(cvSize(cols, rows), CV_8UC1, h_img3);
	dmat4 = cuda::GpuMat(cvSize(cols, rows), CV_8UC3);
	dmat5 = cuda::GpuMat(cvSize(cols, rows), CV_8UC1);
	dmat6 = cuda::GpuMat(cvSize(cols, rows), CV_8UC1);    
    
	timeSec = 0;
	startUVM = getTickCount();
	for(int i=0; i<110; i++){
		cv::cuda::cvtColor(dmat, dmat2, CV_BGR2GRAY);
		Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector( 35.0, 200.0 );
		canny->detect( dmat2, dmat3 );
		if(i >= 10){
			if(i == 10)
				startUVM = getTickCount();
			timeSec += (getTickCount() - startUVM) / getTickFrequency();
		}			
	}
    std::cout << "uvm 4K"<< endl;
    std::cout << "whole tme "<< timeSec/1000 << endl;
    std::cout << "average" << timeSec/100000 << endl;
	timeSec = 0;


	startUVM = getTickCount();
	
	for(int i=0; i<100; i++){
		dmat4.upload(hmat);
		cv::cuda::cvtColor(dmat4, dmat5, CV_BGR2GRAY);
		Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector( 35.0, 200.0 );
		canny->detect( dmat5, dmat6 );
		dmat6.download(hmat3);
		timeSec += (getTickCount() - startUVM) / getTickFrequency();			
	}
    std::cout << "original 4k"<< endl;
    std::cout << "whole tme "<< timeSec/1000 << endl;
    std::cout << "average" << timeSec/100000 << endl;    
	//cuda::bilateralFilter(dmat2, dmat3, 21, 150, 150);
 
 	//cout << hmat << endl;
	//cv::imshow("test", hmat3);
	waitKey(0);
	
	
	//next is video
	/////////////////////////////////////////720P/////////////////////////////////////////////////////////////////////////////////
    VideoCapture cap = VideoCapture("../NORWAY720P.mp4"); // open video
    rows = 720;
	cols = 1280;

    if(!cap.isOpened())  // check if we succeeded
        return -1;
    double fps = cap.get(CV_CAP_PROP_FPS);
 
    // For OpenCV 3, you can also use the following
    // double fps = video.get(CAP_PROP_FPS);
	unsigned char *srcMem;
	unsigned char *grayMem;
	unsigned char *edgeMem;
	cuMallocManaged((void **)&srcMem, rows, cols*3);
	cuMallocManaged((void **)&grayMem, rows, cols);
	cuMallocManaged((void **)&edgeMem, rows, cols);
    cout << "Frames per second using video.get(CV_CAP_PROP_FPS) : " << fps << endl;
   	cout << cap.get(CV_CAP_PROP_FRAME_WIDTH) << endl;
	cout << cap.get(CV_CAP_PROP_FRAME_HEIGHT) << endl; 
    //namedWindow("edges",1);
    Mat frameSrc;
	Mat frame;
	Mat resultframe;
	frame = Mat(cvSize(cols, rows),  CV_8UC3,  srcMem);
	resultframe = Mat(cvSize(cols, rows),  CV_8UC1,  edgeMem);
	Size ksize;
	ksize.width =3;
	ksize.height =3;
	cv::cuda::GpuMat color;
	cv::cuda::GpuMat gray;
	cv::cuda::GpuMat edged;
	cv::cuda::GpuMat color2;
	cv::cuda::GpuMat gray2;
	cv::cuda::GpuMat edged2;	
	color = cv::cuda::GpuMat(cvSize(cols, rows), CV_8UC3, srcMem);
	gray = cv::cuda::GpuMat(cvSize(cols, rows), CV_8UC1, grayMem);
	edged = cv::cuda::GpuMat(cvSize(cols, rows), CV_8UC1, edgeMem);
	color2 = cv::cuda::GpuMat(cvSize(cols, rows), CV_8UC3);
	gray2 = cv::cuda::GpuMat(cvSize(cols, rows), CV_8UC1);
	edged2= cv::cuda::GpuMat(cvSize(cols, rows), CV_8UC1);
	timeSec =0;

	double wholeTime = 0;
    for(int i=0; i<110; i++)
    {
		
	cap >> frameSrc; // get a new frame from camera
	
	const int64 startWhole = getTickCount();
	frameSrc.copyTo(frame);		

	const int64 startCvt = getTickCount();
        cv::cuda::cvtColor(color, gray, CV_BGR2GRAY);
	timeSec = (getTickCount() - startCvt) / getTickFrequency();
	//std::cout << "		Convert Time : " << timeSec << " sec" << std::endl;


	const int64 startCanny = getTickCount();
	Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector( 35.0, 200.0 );
    canny->detect( gray, edged );
	timeSec = (getTickCount() - startCanny) / getTickFrequency();
	//std::cout << "		Canny Time : " << timeSec << " sec" << std::endl;

	timeSec = (getTickCount() - startWhole) / getTickFrequency();
	//std::cout << "	Process Time : " << timeSec << " sec" << std::endl;
	
   //  imshow("edges", frame);
   if(i>=10)
	 wholeTime += timeSec;

	//std::cout << "Whole Time : " << timeSec << " sec" << std::endl;
	

		
      // if(waitKey(30) >= 0) break;
    }
    std::cout << "uvm 720P"<< endl;
    std::cout << "whole tme "<< wholeTime << endl;
    std::cout << "average" << wholeTime/100 << endl;
	std::cout << 1/(wholeTime/100) << endl;
	
	wholeTime = 0;
    for(int i=0; i<110; i++)
    {
		

	cap >> frameSrc; // get a new frame from camera
	const int64 startWhole = getTickCount();		
		color.upload(frameSrc);
	const int64 startCvt = getTickCount();
        cv::cuda::cvtColor(color2, gray2, CV_BGR2GRAY);
	timeSec = (getTickCount() - startCvt) / getTickFrequency();
	//std::cout << "		Convert Time : " << timeSec << " sec" << std::endl;


	const int64 startCanny = getTickCount();
	Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector( 35.0, 200.0 );
    canny->detect( gray2, edged2 );
	timeSec = (getTickCount() - startCanny) / getTickFrequency();
	//std::cout << "		Canny Time : " << timeSec << " sec" << std::endl;

	
	//std::cout << "	Process Time : " << timeSec << " sec" << std::endl;
	edged2.download(resultframe);
	timeSec = (getTickCount() - startWhole) / getTickFrequency();
		
     //imshow("edges", resultframe);
	if(i>= 10)
		wholeTime += timeSec;
	

	//std::cout << "Whole Time : " << timeSec << " sec" << std::endl;
	

		
       // if(waitKey(30) >= 0) break;
    }
    std::cout << "original 720P"<< endl;
    std::cout << "whole tme "<< wholeTime << endl;
    std::cout << "average" << wholeTime/100 << endl;
	std::cout << 1/(wholeTime/100) << endl;	
	////////////////////////////////////////////////////////////////////////////////////////////////////
		
	
	///////////////////////////////2K///////////////////////////////////////////////////////////////////////////////////////////
    cap = VideoCapture("../NORWAY2K.mp4"); // open video
    rows = 1440;
	cols = 2560;


    if(!cap.isOpened())  // check if we succeeded
        return -1;

 

	cuMallocManaged((void **)&srcMem, rows, cols*3);
	cuMallocManaged((void **)&grayMem, rows, cols);
	cuMallocManaged((void **)&edgeMem, rows, cols);
    cout << "Frames per second using video.get(CV_CAP_PROP_FPS) : " << fps << endl;
   	cout << cap.get(CV_CAP_PROP_FRAME_WIDTH) << endl;
	cout << cap.get(CV_CAP_PROP_FRAME_HEIGHT) << endl; 
    //namedWindow("edges",1);

	frame = Mat(cvSize(cols, rows),  CV_8UC3,  srcMem);
	resultframe = Mat(cvSize(cols, rows),  CV_8UC1,  edgeMem);

	color = cv::cuda::GpuMat(cvSize(cols, rows), CV_8UC3, srcMem);
	gray = cv::cuda::GpuMat(cvSize(cols, rows), CV_8UC1, grayMem);
	edged = cv::cuda::GpuMat(cvSize(cols, rows), CV_8UC1, edgeMem);

	timeSec =0;
	color2 = cv::cuda::GpuMat(cvSize(cols, rows), CV_8UC3);
	gray2 = cv::cuda::GpuMat(cvSize(cols, rows), CV_8UC1);
	edged2= cv::cuda::GpuMat(cvSize(cols, rows), CV_8UC1);
	timeSec =0;
	wholeTime = 0;
    for(int i=0; i<100; i++)
    {
		

	cap >> frameSrc; // get a new frame from camera
	const int64 startWhole = getTickCount();		
	color2.upload(frameSrc);
	const int64 startCvt = getTickCount();
        cv::cuda::cvtColor(color2, gray2, CV_BGR2GRAY);
	timeSec = (getTickCount() - startCvt) / getTickFrequency();
	//std::cout << "		Convert Time : " << timeSec << " sec" << std::endl;


	const int64 startCanny = getTickCount();
	Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector( 35.0, 200.0 );
    canny->detect( gray2, edged2 );
	timeSec = (getTickCount() - startCanny) / getTickFrequency();
	//std::cout << "		Canny Time : " << timeSec << " sec" << std::endl;

	
	timeSec = (getTickCount() - startWhole) / getTickFrequency();
	//std::cout << "	Process Time : " << timeSec << " sec" << std::endl;
	edged2.download(resultframe);
		
     //imshow("edges", resultframe);
	 timeSec = (getTickCount() - startWhole) / getTickFrequency();
	
	wholeTime += timeSec;
	

	//std::cout << "Whole Time : " << timeSec << " sec" << std::endl;
	

		
       // if(waitKey(30) >= 0) break;
    }
    std::cout << "original 2k"<< endl;
    std::cout << "whole tme "<< wholeTime << endl;
    std::cout << "average" << wholeTime/100 << endl;
	std::cout << 1/(wholeTime/100) << endl;
	wholeTime = 0;
    for(int i=0; i<110; i++)
    {
		
	cap >> frameSrc; // get a new frame from camera
	const int64 startWhole = getTickCount();			
	frameSrc.copyTo(frame);
	const int64 startCvt = getTickCount();
        cv::cuda::cvtColor(color, gray, CV_BGR2GRAY);
	timeSec = (getTickCount() - startCvt) / getTickFrequency();
	//std::cout << "		Convert Time : " << timeSec << " sec" << std::endl;


	const int64 startCanny = getTickCount();
	Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector( 35.0, 200.0 );
    canny->detect( gray, edged );
	timeSec = (getTickCount() - startCanny) / getTickFrequency();
	//std::cout << "		Canny Time : " << timeSec << " sec" << std::endl;

	timeSec = (getTickCount() - startWhole) / getTickFrequency();
	//std::cout << "	Process Time : " << timeSec << " sec" << std::endl;
	
     //imshow("edges", frame);
	 timeSec = (getTickCount() - startWhole) / getTickFrequency();
	 if(i >= 10){
		wholeTime += timeSec;
	 }
	//std::cout << "Whole Time : " << timeSec << " sec" << std::endl;
	

		
       // if(waitKey(30) >= 0) break;
    }
    std::cout << "uvm 2k"<< endl;
    std::cout << "whole tme "<< wholeTime << endl;
    std::cout << "average" << wholeTime/100 << endl;
	std::cout << 1/(wholeTime/100) << endl;
	////////////////////////////////////////////////////////////////////////////////////////////////////
	
	///////////////////////////////1080P//////////////////////////////////////////////////////////////////////////////////////////
    cap = VideoCapture("../NORWAY1080P.mp4"); // open video
	rows = 1080;
	cols = 1920;   


    if(!cap.isOpened())  // check if we succeeded
        return -1;

 

	cuMallocManaged((void **)&srcMem, rows, cols*3);
	cuMallocManaged((void **)&grayMem, rows, cols);
	cuMallocManaged((void **)&edgeMem, rows, cols);
    cout << "Frames per second using video.get(CV_CAP_PROP_FPS) : " << fps << endl;
   	cout << cap.get(CV_CAP_PROP_FRAME_WIDTH) << endl;
	cout << cap.get(CV_CAP_PROP_FRAME_HEIGHT) << endl; 
    //namedWindow("edges",1);

	frame = Mat(cvSize(cols, rows),  CV_8UC3,  srcMem);
	resultframe = Mat(cvSize(cols, rows),  CV_8UC1,  edgeMem);

	color = cv::cuda::GpuMat(cvSize(cols, rows), CV_8UC3, srcMem);
	gray = cv::cuda::GpuMat(cvSize(cols, rows), CV_8UC1, grayMem);
	edged = cv::cuda::GpuMat(cvSize(cols, rows), CV_8UC1, edgeMem);

	timeSec =0;
	color2 = cv::cuda::GpuMat(cvSize(cols, rows), CV_8UC3);
	gray2 = cv::cuda::GpuMat(cvSize(cols, rows), CV_8UC1);
	edged2= cv::cuda::GpuMat(cvSize(cols, rows), CV_8UC1);
	timeSec =0;
	wholeTime = 0;
    for(int i=0; i<110; i++)
    {
		

	cap >> frameSrc; // get a new frame from camera
	const int64 startWhole = getTickCount();		
	color2.upload(frameSrc);
	const int64 startCvt = getTickCount();
        cv::cuda::cvtColor(color2, gray2, CV_BGR2GRAY);
	timeSec = (getTickCount() - startCvt) / getTickFrequency();
	//std::cout << "		Convert Time : " << timeSec << " sec" << std::endl;


	const int64 startCanny = getTickCount();
	Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector( 35.0, 200.0 );
    canny->detect( gray2, edged2 );
	timeSec = (getTickCount() - startCanny) / getTickFrequency();
	//std::cout << "		Canny Time : " << timeSec << " sec" << std::endl;

	
	timeSec = (getTickCount() - startWhole) / getTickFrequency();
	//std::cout << "	Process Time : " << timeSec << " sec" << std::endl;
	edged2.download(resultframe);
		
     //imshow("edges", resultframe);
	 timeSec = (getTickCount() - startWhole) / getTickFrequency();
	if(i>= 10)
		wholeTime += timeSec;
	

	//std::cout << "Whole Time : " << timeSec << " sec" << std::endl;
	

		
       // if(waitKey(30) >= 0) break;
    }
    std::cout << "original 1080"<< endl;
    std::cout << "whole tme "<< wholeTime << endl;
    std::cout << "average" << wholeTime/100 << endl;
	std::cout << 1/(wholeTime/100) << endl;
	wholeTime = 0;
    for(int i=0; i<110; i++)
    {
		
	cap >> frameSrc; // get a new frame from camera
	const int64 startWhole = getTickCount();			
	frameSrc.copyTo(frame);
	const int64 startCvt = getTickCount();
        cv::cuda::cvtColor(color, gray, CV_BGR2GRAY);
	timeSec = (getTickCount() - startCvt) / getTickFrequency();
	//std::cout << "		Convert Time : " << timeSec << " sec" << std::endl;


	const int64 startCanny = getTickCount();
	Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector( 35.0, 200.0 );
    canny->detect( gray, edged );
	timeSec = (getTickCount() - startCanny) / getTickFrequency();
	//std::cout << "		Canny Time : " << timeSec << " sec" << std::endl;

	timeSec = (getTickCount() - startWhole) / getTickFrequency();
	//std::cout << "	Process Time : " << timeSec << " sec" << std::endl;
	
     //imshow("edges", frame);
	 timeSec = (getTickCount() - startWhole) / getTickFrequency();
	 if(i>= 10)
		wholeTime += timeSec;

	//std::cout << "Whole Time : " << timeSec << " sec" << std::endl;
	

		
       // if(waitKey(30) >= 0) break;
    }
    std::cout << "uvm 1080"<< endl;
    std::cout << "whole tme "<< wholeTime << endl;
    std::cout << "average" << wholeTime/100 << endl;
	std::cout << 1/(wholeTime/100) << endl;
	////////////////////////////////////////////////////////////////////////////////////////////////////	
	
	
	//write atomicity test
	cv::cuda::Stream stream;

     for(; ;)
    {
        cv::cuda::cvtColor(dmat, dmat2, CV_BGR2GRAY,0, stream);
		Ptr<cv::cuda::Filter> filter = cv::cuda::createBoxFilter(gray.type(), gray.type(), Size(3,3));
	        filter->apply(dmat2, dmat2, stream);
		cv::boxFilter(hmat2, hmat3,-1, ksize);
		
		//cv::Canny( hmat2, hmat3, 35.0, 200.0 );
		stream.waitForCompletion();
		imshow("edges", hmat3);
		waitKey(0);
	} 
//////////////////////////////////////////////////////////////////////////////////////////
	//video capture buffer test.
	return 0;
}
