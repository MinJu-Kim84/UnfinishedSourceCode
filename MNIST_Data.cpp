#include "MNIST_Data.h"
#include "stdafx.h"
#include <iostream>

using namespace std;

MNIST_Data::MNIST_Data(){

}

MNIST_Data::MNIST_Data(const char *imgPath, const char *labelPath){
	Create(imgPath, labelPath);
}

MNIST_Data::~MNIST_Data(){

}

void MNIST_Data::Set(const char *imgPath, const char *labelPath){
	Clear();
	Create(imgPath, labelPath);
}

void MNIST_Data::Release(){
	Clear();
}

void MNIST_Data::Create(const char *imgPath, const char *labelPath){
	FILE *img_fp = NULL;
	FILE *label_fp = NULL;

	union TransVal{
		uchar buff08[4];
		int buff32;
	}data;

	fopen_s(&img_fp, imgPath, "rb");
	if(!img_fp){
		cout << "이미지 파일을 열 수 없습니다.\n";
		return;
	}
	fopen_s(&label_fp, labelPath, "rb");
	if(!label_fp){
		cout << "라벨링 파일을 열 수 없습니다.\n";
		return;
	}

	int param[4];

	for(int i = 0; i < 4; ++i){
		for(int k = 0; k < 4; ++k){
			fread_s(&data.buff08[3 - k], sizeof(uchar), sizeof(uchar), 1, img_fp);
		}
		param[i] = data.buff32;
		cout << data.buff32 << endl;
	}

	int nSample = param[1];
	int width = param[2];
	int height = param[3];

	for(int i = 0; i < 2; ++i){
		for(int k = 0; k < 4; ++k){
			fread_s(&data.buff08[3 - k], sizeof(uchar), sizeof(uchar), 1, label_fp);
		}
		param[i] = data.buff32;
		cout << data.buff32 << endl;
	}

	if(nSample != param[1]){
		cout << "샘플 이미지와 라벨링 데이터 수가 맞지 않습니다.\n";
		fclose(img_fp);
		fclose(label_fp);
		return;
	}

	sample.Set(width, height, nSample);
	uchar *_tmp = new uchar[nSample];
	result = cv::Mat::zeros(nSample, 10, CV_64FC1);

	fread_s(sample.memBlk1D, sample.t_elements * sizeof(uchar), sizeof(uchar), sample.t_elements, img_fp); 
	fread_s(_tmp, sizeof(uchar) * nSample, sizeof(uchar), nSample, label_fp); 

	fclose(img_fp);
	fclose(label_fp);

	for(int i = 0; i < nSample; ++i){
		result.at<double>(i, _tmp[i]) = 1;
	}

	delete[] _tmp;
}

void MNIST_Data::Clear(){
//	sample.clear();
	result.release();
}
