//! [includes]
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <iostream>
#include <string>

using namespace cv;
using namespace std;

void salt(Mat &image, int n) {
	int i, j;
	for (int k = 0; k < n; k++) {
		// rand() is the MFC random number generator
		i = rand() % image.cols;
		j = rand() % image.rows;
		if (image.channels() == 1) { // gray-level image
			image.at<uchar>(j, i) = 255;
		}
		else if (image.channels() == 3) { // color image
			image.at<Vec3b>(j, i)[0] = 255;
			image.at<Vec3b>(j, i)[1] = 255;
			image.at<Vec3b>(j, i)[2] = 255;
		}
	}
}

int exercise1(int argc, char** argv)
{
	string imgName;
	cout << "Please enter the image name: ";
	cin >> imgName;

	imgName += ".jpg";

	
	string imageName(imgName); 
	if (argc > 1)
	{
		imageName = argv[1];
	}
	
	Mat image;

	image = imread(imageName.c_str(), IMREAD_COLOR); 
													

	if (image.empty())                      
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	//Image Dimensions
	cout << endl << "Image Dimensions" << endl;
	cout << "Width->" << image.cols << "px\tHeight->" << image.rows << "px" << endl << endl;

	//Write Image to bmp
	imwrite("New_Image.bmp", image);
	cout << "Bmp Image Created";


	namedWindow("Display window", WINDOW_AUTOSIZE);											
												
	imshow("Display window", image);                

	waitKey(0); 
				
	return 0;
}

int exercise2(int argc, char** argv)
{
	string imageName("HappyFish.jpg"); 
	if (argc > 1)
	{
		imageName = argv[1];
	}

	Mat img1, img2, img3;
	Mat newImg(50, 200, CV_8U, Scalar(100));
	newImg.at<uchar>(25, 100) = 255;

	img1 = imread(imageName.c_str(), IMREAD_COLOR); 											

	if (img1.empty())                      
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	img2 = img1;
	img1.copyTo(img3);
	flip(img3, img2, 1);

	namedWindow("Display window 1", WINDOW_AUTOSIZE); 
	namedWindow("Display window 2", WINDOW_AUTOSIZE);
	namedWindow("Display window 3", WINDOW_AUTOSIZE);
	namedWindow("Display window New Image", WINDOW_AUTOSIZE);

	imshow("Display window 1", img1);                
	imshow("Display window 2", img2);
	imshow("Display window 3", img3);
	imshow("Display window New Image", newImg);

	cout << waitKey(0); 
						
	return 0;
}

int exercise3a(int argc, char** argv)
{
	string imageName("HappyFish.jpg"); 
	if (argc > 1)
	{
		imageName = argv[1];
	}

	Mat imgRGB, imgGray;

	imgRGB = imread(imageName.c_str(), IMREAD_COLOR); 

	if (imgRGB.empty())                     
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	cvtColor(imgRGB, imgGray, CV_BGR2GRAY);

	namedWindow("Display window RGB", WINDOW_AUTOSIZE);
	namedWindow("Display window Gray", WINDOW_AUTOSIZE);

	imshow("Display window RGB", imgRGB);                
	imshow("Display window Gray", imgGray);

	waitKey(0); 

	return 0;
}

int exercise3b(int argc, char** argv)
{
	string imageName("HappyFish.jpg");
	if (argc > 1)
	{
		imageName = argv[1];
	}

	Mat imgRGB, imgNoise;

	imgRGB = imread(imageName.c_str(), IMREAD_COLOR); 

	if (imgRGB.empty())                      
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	int noisePoints = (imgRGB.cols * imgRGB.rows) * .1;
	imgRGB.copyTo(imgNoise);
	salt(imgNoise, noisePoints);

	namedWindow("Display window RGB", WINDOW_AUTOSIZE); 

														
	imshow("Display window RGB", imgRGB);                
	imshow("Display window Noise", imgNoise);

	waitKey(0); 

	return 0;
}

int exercise3c(int argc, char** argv)
{
	string imageName("HappyFish.jpg");
	if (argc > 1)
	{
		imageName = argv[1];
	}

	Mat imgRGB, imgResult;

	// create vector of 3 images
	vector<Mat> planes;

	imgRGB = imread(imageName.c_str(), IMREAD_COLOR); 

	if (imgRGB.empty())                      
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	// split 1 3-channel image into 3 1-channel images
	split(imgRGB, planes);

	planes[0] += 100;

	merge(planes, imgResult);

	namedWindow("Display window RGB", WINDOW_AUTOSIZE); 
														
	imshow("Display window RGB", imgRGB);                
	imshow("Blue Image", planes[0]);
	imshow("Green Image", planes[1]);
	imshow("Red Image", planes[2]);
	imshow("Result", imgResult);

	waitKey(0); 

	return 0;
}

int exercise3d(int argc, char** argv)
{
	string imageName("HappyFish.jpg");
	if (argc > 1)
	{
		imageName = argv[1];
	}

	Mat imgRGB, imgResult;

	// create vector of 3 images
	vector<Mat> planes;

	imgRGB = imread(imageName.c_str(), IMREAD_COLOR); 
													  
	if (imgRGB.empty())                      
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	cvtColor(imgRGB, imgRGB, CV_BGR2HSV);

	// split 1 3-channel image into 3 1-channel images
	split(imgRGB, planes);

	planes[1] += 100;

	merge(planes, imgResult);

	cvtColor(imgResult, imgResult, CV_HSV2BGR);
	cvtColor(imgRGB, imgRGB, CV_HSV2BGR);

	namedWindow("Display window RGB", WINDOW_AUTOSIZE); 
														
	imshow("Display window RGB", imgRGB);                
	imshow("Hue Image", planes[0]);
	imshow("Saturation Image", planes[1]);
	imshow("Value Image", planes[2]);
	imshow("Result", imgResult);

	waitKey(0); 

	return 0;
}

int exercise3(int argc, char** argv)
{
	string exercise;

	while (true)
	{
		cout << "1 -> Read a color image convert it to grayscale" << endl;
		cout << "2 -> Read an image and add \"salt and pepper\" noise" << endl;
		cout << "3 -> Read a color image and split the 3 channel" << endl;
		cout << "4 -> Read a color image and convert it to HSV" << endl;
		cout << "0 -> Exit" << endl;
		cout << "Select exercise: ";
		cin >> exercise;

		int val = atoi(exercise.c_str());

		switch (val) {
		case 0:
			return 0;
		case 1:
			return exercise3a(argc, argv);
		case 2:
			return exercise3b(argc, argv);
		case 3:
			return exercise3c(argc, argv);
		case 4:
			return exercise3d(argc, argv);
		}
	}
}

int exercise4(int argc, char** argv)
{
	int SPACEBAR_KEY = 32;
	int keyPressed = -1;

	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	Mat edges, edgesGray;

	namedWindow("Video Cam", 1);
	namedWindow("Video Cam Gray", 1);

	for (;;)
	{
		Mat frame;
		cap >> frame; // get a new frame from camera
		cvtColor(frame, edges, 0);
		cvtColor(frame, edgesGray, CV_BGR2GRAY);

		flip(edges, edges, 1); //flip video
		flip(edgesGray, edgesGray, 1); //flip video

		imshow("Video Cam", edges);
		imshow("Video Cam Gray", edgesGray);

		keyPressed = waitKey(30);

		if (keyPressed == SPACEBAR_KEY) {
			imwrite("photo.jpg", edges);
		}
		else if (keyPressed >= 0) {
			break;
		}
	}

	return 0;
}

int exercise5a(int argc, char** argv)
{
	Mat src, dst;

	string imageName("LowContrast.jpg"); 
	src = imread(imageName.c_str(), IMREAD_COLOR);

	if (src.empty())                      // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	cvtColor(src, src, CV_BGR2GRAY);

	/// Separate the image in 3 places ( B, G and R )
	vector<Mat> bgr_planes;
	split(src, bgr_planes);

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat b_hist, g_hist, r_hist;

	/// Compute the histograms:
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}

	/// Display
	namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
	imshow("calcHist Demo", histImage);

	imshow("Source image", src);

	waitKey(0);

	return 0;
}

int exercise5b(int argc, char** argv)
{

	string imageName("LowContrast.jpg");
	if (argc > 1)
	{
		imageName = argv[1];
	}

	Mat orgImg, histEqual, claheImg, labImg;

	orgImg = imread(imageName.c_str(), IMREAD_COLOR);

	if (orgImg.empty())
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	Ptr<CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(4);

	cvtColor(orgImg, labImg, CV_BGR2Lab);

	cvtColor(orgImg, orgImg, CV_BGR2GRAY);

	//CLAHE
	vector<Mat> lab_planes(3);
	split(labImg, lab_planes);
	clahe->apply(lab_planes[0], claheImg);

	claheImg.copyTo(lab_planes[0]);
	merge(lab_planes, labImg);

	cvtColor(labImg, claheImg, CV_Lab2BGR);


	/// Apply Histogram Equalization
	equalizeHist(orgImg, histEqual);

	imshow("Original", orgImg);
	imshow("Histogram Equalization", histEqual);
	imshow("Clahe", claheImg);

	waitKey(0);

	return 0;
}

int exercise5(int argc, char** argv)
{
	string exercise;

	while (true)
	{
		cout << "1 -> Take a low contrast image and plot its histogram" << endl;
		cout << "2 -> Enhance the image constrast" << endl;
		cout << "0 -> Exit" << endl;
		cout << "Select exercise: ";
		cin >> exercise;

		int val = atoi(exercise.c_str());

		switch (val) {
		case 0:
			return 0;
		case 1:
			return exercise5a(argc, argv);
		case 2:
			return exercise5b(argc, argv);
		}
	}
}

int exercise6(int argc, char** argv)
{
	int KERNEL_LENGTH = 5;

	string imageName("Noise.jpg");
	if (argc > 1)
	{
		imageName = argv[1];
	}

	Mat src, meanImg, gaussianImg, medianImg, bilateralImg;

	src = imread(imageName.c_str(), IMREAD_COLOR);


	if (src.empty())
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	blur(src, meanImg, Size(KERNEL_LENGTH, KERNEL_LENGTH));

	GaussianBlur(src, gaussianImg, Size(KERNEL_LENGTH, KERNEL_LENGTH), 0, 0);

	medianBlur(src, medianImg, KERNEL_LENGTH);

	bilateralFilter(src, bilateralImg, KERNEL_LENGTH, KERNEL_LENGTH * 2, KERNEL_LENGTH / 2);

	imshow("Original", src);
	imshow("Mean", meanImg);
	imshow("Gaussian", gaussianImg);
	imshow("Median", medianImg);
	imshow("Bilateral", bilateralImg);

	waitKey(0);

	return 0;
}

void exercise7_Sobel(Mat src_gray)
{
	Mat grad;
	Mat abs_grad_x, abs_grad_y;
	Mat grad_x, grad_y;

	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	imshow("Sobel", grad);
}

void exercise7_Canny(Mat src_gray)
{
	int edgeThresh = 1;
	int lowThreshold = 0;
	int const maxThreshold = 100;
	int kernel_size = 3;

	Mat detected_edges, dst;

	/// Reduce noise with a kernel 3x3
	blur(src_gray, detected_edges, Size(3, 3));

	/// Canny detector
	Canny(detected_edges, detected_edges, lowThreshold, maxThreshold, kernel_size);

	/// Using Canny's output as a mask, we display our result
	dst = Scalar::all(0);

	src_gray.copyTo(dst, detected_edges);
	imshow("Canny", dst);

}

void exercise7_Laplace(Mat src_gray)
{
	int kernel_size = 3;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	Mat abs_dst, dst;

	/// Reduce noise with a kernel 3x3
	blur(src_gray, src_gray, Size(3, 3));

	/// Laplace Filter
	Laplacian(src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(dst, abs_dst);

	imshow("Laplace", abs_dst);

}

int exercise7(int argc, char** argv)
{
	string imageName("Lena.jpg");
	if (argc > 1)
	{
		imageName = argv[1];
	}

	Mat src, src_gray;

	src = imread(imageName.c_str(), IMREAD_COLOR);

	if (src.empty())
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);

	cvtColor(src, src_gray, CV_RGB2GRAY);

	exercise7_Sobel(src_gray);

	exercise7_Canny(src_gray);

	exercise7_Laplace(src_gray);

	imshow("Source", src);

	waitKey(0);

	return 0;
}

void exercise8_StandardHough(Mat src)
{
	Mat dst, cdst;
	Canny(src, dst, 50, 200, 3);
	cvtColor(dst, cdst, CV_GRAY2BGR);

	vector<Vec2f> lines;
	// detect lines
	HoughLines(dst, lines, 1, CV_PI / 180, 175, 0, 0);

	// draw lines
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(cdst, pt1, pt2, Scalar(0, 0, 255), 1, CV_AA);
	}
	imshow("Standard Hough", cdst);
}

void exercise8_ProbabilisticHough(Mat src)
{
	Mat dst, cdst;
	Canny(src, dst, 50, 200, 3);
	cvtColor(dst, cdst, CV_GRAY2BGR);

	vector<Vec4i> lines;
	// detect lines
	HoughLinesP(dst, lines, 1, CV_PI / 180, 50, 50, 10);

	// draw lines
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, CV_AA);
	}

	imshow("Probabilistic Hough", cdst);
}

int exercise8_Lines(int argc, char** argv)
{
	string imageName("Building.jpg");
	if (argc > 1)
	{
		imageName = argv[1];
	}

	Mat src;

	src = imread(imageName.c_str(), IMREAD_COLOR);

	if (src.empty())
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);

	imshow("Source", src);

	exercise8_StandardHough(src);

	exercise8_ProbabilisticHough(src);

	waitKey(0);

	return 0;
}

int exercise8_Circle(int argc, char** argv)
{
	string imageName("Olympic.jpg");
	if (argc > 1)
	{
		imageName = argv[1];
	}

	Mat src, src_gray;

	src = imread(imageName.c_str(), IMREAD_COLOR);

	if (src.empty())
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);

	cvtColor(src, src_gray, CV_RGB2GRAY);

	vector<Vec3f> circles;

	Mat dst;

	src.copyTo(dst);

	/// Apply the Hough Transform to find the circles
	HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows / 8, 200, 100, 0, 0);

	/// Draw the circles detected
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle(dst, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// circle outline
		circle(dst, center, radius, Scalar(128, 128, 128), 3, 8, 0);
	}

	imshow("Source", src);
	imshow("Hough Circle", dst);

	waitKey(0);

	return 0;
}

int exercise8(int argc, char** argv)
{
	string exercise;

	while (true)
	{
		cout << "1 -> Lines" << endl;
		cout << "2 -> Circle" << endl;
		cout << "0 -> Exit" << endl;
		cout << "Select exercise: ";
		cin >> exercise;

		int val = atoi(exercise.c_str());

		switch (val) {
		case 0:
			return 0;
		case 1:
			return exercise8_Lines(argc, argv);
		case 2:
			return exercise8_Circle(argc, argv);
		}
	}
}

int main(int argc, char** argv)
{
	string exercise;

	while (true)
	{
		cout << "1 -> Images - read, write and display" << endl;
		cout << "2 -> Images - creation" << endl;
		cout << "3 -> Images - representation, grayscale & color spaces" << endl;
		cout << "4 -> Video - acquisition and simple processing" << endl;
		cout << "5 -> Image enhancement - histogram equalization" << endl;
		cout << "6 -> Image enhancement - filtering" << endl;
		cout << "7 -> Edge detection" << endl;
		cout << "8 -> Hough transform - line and circle detection" << endl;
		cout << "0 -> Exit" << endl;
		cout << "Select exercise: ";
		cin >> exercise;

		int val = atoi(exercise.c_str());

		switch (val) {
		case 0:
			return 0;
		case 1:
			return exercise1(argc, argv);
		case 2:
			return exercise2(argc, argv);
		case 3:
			return exercise3(argc, argv);
		case 4:
			return exercise4(argc, argv);
		case 5:
			return exercise5(argc, argv);
		case 6:
			return exercise6(argc, argv);
		case 7:
			return exercise7(argc, argv);
		case 8:
			return exercise8(argc, argv);
		}
	}
}