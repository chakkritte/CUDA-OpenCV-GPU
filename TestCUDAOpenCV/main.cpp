#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include <vector>
#include <iostream>
#include <iomanip>
#include "opencv2/contrib/contrib.hpp"
#include <opencv2/ocl/ocl.hpp>

#ifdef _DEBUG
#pragma comment(lib, "opencv_core2411d.lib")
#pragma comment(lib, "opencv_imgproc2411d.lib")   //MAT processing
#pragma comment(lib, "opencv_objdetect2411d.lib") //HOGDescriptor
#pragma comment(lib, "opencv_gpu2411d.lib")
#pragma comment(lib, "opencv_highgui2411d.lib")
#else
#pragma comment(lib, "opencv_core2411.lib")
#pragma comment(lib, "opencv_imgproc2411.lib")
#pragma comment(lib, "opencv_objdetect2411.lib")
#pragma comment(lib, "opencv_gpu2411.lib")
#pragma comment(lib, "opencv_highgui2411.lib")
#endif

using namespace std;
using namespace cv;
using namespace cv::gpu;
using namespace cv::ocl;

/** Global variables */
String face_cascade_name = "haarcascade_frontalface_alt2.xml";
CascadeClassifier face_cascade;
CascadeClassifier_GPU face_cascade_gpu;

//CascadeClassifier eyes_cascade;
string window_name = "CPU - Face detection";
string window_namegpu = "GPU OpenCL- Face detection";
string window_namegpu_cuda = "GPU CUDA- Face detection";

RNG rng(12345);

float TakeTime;
unsigned long Atime, Btime;

struct getRect { Rect operator ()(const CvAvgComp& e) const { return e.rect; } };

/** @function detectAndDisplay */
void detectAndDisplayCPU(Mat& frame)
{
	vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	Atime = getTickCount();

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));

	Btime = getTickCount();
	TakeTime = (Btime - Atime) / getTickFrequency();
	printf("detected face(cpu version) = %d / %lf sec take.\n", faces.size(), TakeTime);

	if (faces.size() >= 1)
	{
		for (int ji = 0; ji < faces.size(); ++ji)
		{
			rectangle(frame, faces[ji], CV_RGB(0, 0, 255), 4);
		}
	}
	//-- Show what you got
	namedWindow(window_name, WINDOW_AUTOSIZE);
	imshow(window_name, frame);
	//imwrite("cpu.jpg", frame);
}


void detectAndDisplayGPUCUDA(Mat frame)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	GpuMat faceBuf_gpu;
	GpuMat GpuImg;
	GpuImg.upload(frame_gray);
	Atime = getTickCount();
	int detectionNumber = face_cascade_gpu.detectMultiScale(GpuImg, faceBuf_gpu, 1.1, 2, Size(20, 20));
	Btime = getTickCount();
	TakeTime = (Btime - Atime) / getTickFrequency();
	printf("detected face(gpu version) =%d / %lf sec take.\n", detectionNumber, TakeTime);

	Mat faces_downloaded;
	if (detectionNumber >= 1)
	{
		faceBuf_gpu.colRange(0, detectionNumber).download(faces_downloaded);
		Rect* faces = faces_downloaded.ptr< Rect>();

		for (int ji = 0; ji < detectionNumber; ++ji)
		{
			rectangle(frame, Point(faces[ji].x, faces[ji].y), Point(faces[ji].x + faces[ji].width, faces[ji].y + faces[ji].height), CV_RGB(255, 0, 0), 2);
		}
	}

	imshow(window_namegpu_cuda, frame);
}

/*
void detectAndDisplayOpenCL(Mat frame)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

}
*/

void detectAndDraw(Mat& img,
	cv::ocl::OclCascadeClassifier& cascade, CascadeClassifier&,
	double scale)
{
	int i = 0;
	double t = 0;
	vector<Rect> faces;
	const static Scalar colors[] = { CV_RGB(0, 0, 255),
		CV_RGB(0, 128, 255),
		CV_RGB(0, 255, 255),
		CV_RGB(0, 255, 0),
		CV_RGB(255, 128, 0),
		CV_RGB(255, 255, 0),
		CV_RGB(255, 0, 0),
		CV_RGB(255, 0, 255) };
	cv::ocl::oclMat image(img);
	cv::ocl::oclMat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);

	cv::ocl::cvtColor(image, gray, CV_BGR2GRAY);
	cv::ocl::resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
	cv::ocl::equalizeHist(smallImg, smallImg);

	CvSeq* _objects;
	MemStorage storage(cvCreateMemStorage(0));
	//t = (double)cvGetTickCount();
	Atime = getTickCount();
	_objects = cascade.oclHaarDetectObjects(smallImg, storage, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE , Size(20, 20));

	Btime = getTickCount();

	vector<CvAvgComp> vecAvgComp;
	Seq<CvAvgComp>(_objects).copyTo(vecAvgComp);
	faces.resize(vecAvgComp.size());
	std::transform(vecAvgComp.begin(), vecAvgComp.end(), faces.begin(), getRect());
	//t = (double)cvGetTickCount() - t;
	//printf("detection time = %g ms\n", t / ((double)cvGetTickFrequency()*1000.));


	TakeTime = (Btime - Atime) / getTickFrequency();
	printf("detected face(gpu version) = %d / %lf sec take.\n", faces.size(), TakeTime);

	/*
	for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++)
	{
		Mat smallImgROI;
		Point center;
		Scalar color = colors[i % 8];
		int radius;
		center.x = cvRound((r->x + r->width*0.5)*scale);
		center.y = cvRound((r->y + r->height*0.5)*scale);
		radius = cvRound((r->width + r->height)*0.25*scale);
		circle(img, center, radius, color, 3, 8, 0);
	}
	*/

	if (faces.size() >= 1)
	{
		for (int ji = 0; ji < faces.size(); ++ji)
		{
			rectangle(img, faces[ji], CV_RGB(255, 0, 0), 4);
		}
	}
	namedWindow(window_namegpu, WINDOW_AUTOSIZE);
	imshow(window_namegpu, img);
	//imwrite("gpu.jpg", img);
}



/** @function main */
int main(int argc, const char *argv[])
{

	if (getCudaEnabledDeviceCount() == 0)
	{
		return cerr << "No GPU found or the library is compiled without GPU support" << endl, -1;
	}

	//cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());
	//CvCapture* capture;
	//Code use CPU

	Mat frame;
	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)){ printf("--(!)CPU Error loading\n"); return -1; };
	//if (!eyes_cascade.load(eyes_cascade_name)){ printf("--(!)Error loading\n"); return -1; };
	//cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());
	cv::Mat img = cv::imread("MyImages\\6.jpg");
	Mat imgCPU = img.clone();
	Mat imgGPU = img.clone();
	Mat imgGPUCUDA = img.clone();

	detectAndDisplayCPU(imgCPU);

	//Code use OpenCL

	DevicesInfo devices;
	getOpenCLDevices(devices);
	setDevice(devices[0]);
	OclCascadeClassifier cascade_ocl;
	CascadeClassifier  nestedCascade;
	if (!cascade_ocl.load(face_cascade_name)){ printf("--(!)GPU OpenCL Error loading\n"); return -1; };
	double scale = 1;
	detectAndDraw(imgGPU, cascade_ocl, nestedCascade, scale);


	//Code use CUDA

	if (!face_cascade_gpu.load(face_cascade_name)){ printf("--(!)GPU CUDA Error loading\n"); return -1; };
	detectAndDisplayGPUCUDA(imgGPUCUDA);
	//detectAndDisplayGPU(img);

	cv::waitKey(0);
	/*
	//-- 2. Read the video stream
	capture = cvCaptureFromCAM(0);
	if (capture)
	{
	while (true)
	{
	frame = cvQueryFrame(capture);

	//-- 3. Apply the classifier to the frame
	if (!frame.empty())
	{
	detectAndDisplay(frame);
	}
	else
	{
	printf(" --(!) No captured frame -- Break!"); break;
	}

	int c = waitKey(10);
	if ((char)c == 'c') { break; }
	}
	}
	*/
	return 0;
}
