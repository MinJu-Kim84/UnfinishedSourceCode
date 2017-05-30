#ifndef TRANCE_MAT_TYPE_H
#define TRANCE_MAT_TYPE_H

#include <opencv2\opencv.hpp>

// TranceMatType

class TranceMatType{
public:
	const int typeVal;

	TranceMatType(uchar *data);
	TranceMatType(int *data);
	TranceMatType(float *data);
	TranceMatType(double *data);
};

#endif