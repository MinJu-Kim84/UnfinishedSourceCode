#include "TranceMatType.h"
#include "stdafx.h"


TranceMatType::TranceMatType(uchar *data) : typeVal(CV_8UC1){
}

TranceMatType::TranceMatType(int *data) : typeVal(CV_32SC1){
}

TranceMatType::TranceMatType(float *data) : typeVal(CV_32FC1){
}

TranceMatType::TranceMatType(double *data) : typeVal(CV_64FC1){
}