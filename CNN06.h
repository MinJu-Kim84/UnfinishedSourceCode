#ifndef CNN06_H
#define CNN06_H

#include <vector>
#include <opencv2\opencv.hpp>
#include <tbb\tbb.h>
#include <iostream>
#include <fstream>
#include "AllocBlockMem.h"

#define CONV_MODEL 2
#define POOL_MODEL 2
#define COST_MODEL 2

using namespace std;
using namespace tbb;

typedef MemBlock3D<uchar> Mem3UC;
typedef MemBlock2D<double> Mem2D;
typedef MemBlock3D<double> Mem3D;
typedef MemBlock4D<double> Mem4D;

/***************************************************************
*															   *
*					      CvCNN_Info   						   *
*															   *
****************************************************************/

struct CvCNN_Info{
	int upNodes;
	int dwNodes;
};

/***************************************************************
*															   *
*					      CvCNN     						   *
*															   *
****************************************************************/

class CvCNN{

public:
	bool learnFlag;
	bool initFlag; 
	double learnRate;
	double errorRate;

	int t_clayer;
	int t_mlayer;
	int t_sample;

	int itor;

	vector<MemBlock4D<double>> kernel;
	vector<MemBlock4D<double>> ioNode;
	vector<MemBlock4D<double>> convNode;
	vector<MemBlock4D<double>> convDelta;
	vector<MemBlock3D<double>> ioDelta;
	MemBlock4D<double> lastDelta;
	vector<MemBlock3D<double>> padDelta;
	vector<MemBlock4D<uchar>> poolMark;
	vector<double*> cnnBias;

	vector<MemBlock2D<double>> mlpNode;
	vector<MemBlock2D<double>> mlpDelta;
	vector<MemBlock2D<double>> weight;
	vector<double*> mlpBias;

	task_scheduler_init init;

	CvCNN();
	CvCNN(cv::Mat &_cnnNode, cv::Mat &_kernel, cv::Mat _mlpNode,
		int _itor, int nSample, double _learnRate, double _errorRate);
	~CvCNN();

	bool Check(cv::Mat &_cnnNode, cv::Mat &_kernel, cv::Mat _mlpNode);
	void MoveData(const MemBlock3D<uchar> &sample);
	void Trainning(Mem3UC &samples, cv::Mat &target);
	void Predict(cv::Mat &input, cv::Mat &result);
	void Save(const char *path);
	void Load(const char *path);

	/*
	void Convolution(Mem4D &input, Mem4D &_kernel, Mem4D &output, vector<vector<double>> &bias);
	void ActivFunc(CvCNN_Node<float> &currMat, int method);
	void Pooling(CvCNN_Node<float> &input, CvCNN_Node<float> &output, CvCNN_Node<char> &poolMark, int method);
	float CalError(CvCNN_Node<float> &output, cv::Mat &targetMat, CvCNN_Node<float> &deltaMat, int method);
	void InsertPad(CvCNN_Node<float> &inDelta, CvCNN_Node<float> &outDelta);
	void Correlation(CvCNN_Node<float> &inDelta, CvCNN_Node<float> &outDelta, CvCNN_Kernel &kernel);
	*/
	void UnPooling(Mem3D &input, Mem3D &output, Mem3UC &mark);
	/*
	void DeActFunc(CvCNN_Node<float> &upDelta, CvCNN_Node<float> &downDelta, CvCNN_Node<float> &output, int method);
	void ModifyKernel(CvCNN_Node<float> &input, CvCNN_Kernel &kernel, CvCNN_Node<float> &convDelta, CvCNN_Node<float> &moment);

	void ShowMemBlk(CvCNN_Node<float> mat, int nBlock);
	void ShowMemBlk2(CvCNN_Node<char> mat, int nBlock);
	void PrintInfo(CvCNN_Node<float> &currMat);
	*/
};

#endif