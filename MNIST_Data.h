#ifndef MNIST_DATA_H
#define MNIST_DATA_H

#include <opencv2\opencv.hpp>
#include "AllocBlockMem.h"

class MNIST_Data{
	void Create(const char *imgPath, const char *labelPath);
	void Clear();

public:
	MemBlock3D<uchar> sample;
	cv::Mat result;

	MNIST_Data();
	MNIST_Data(const char *path, const char *labelPath);
	~MNIST_Data();

	void Set(const char *path, const char *labelPath);
	void Release();
};

#endif