#include "CNN06.h"
#include "stdafx.h"
#include <time.h>



//*********************CvCNN*****************************************

CvCNN::CvCNN()
{
	learnFlag = false;
	initFlag = false;
	learnRate = 0.f;
	errorRate = 0.f;

	t_clayer = 0;
	t_mlayer = 0;
	t_sample = 0;

	itor = 0;
}

CvCNN::CvCNN(cv::Mat &_cnnNode, cv::Mat &_kernel, cv::Mat _mlpNode,
		int _itor, int nSample, double _learnRate, double _errorRate)
{
	if(!Check(_cnnNode, _kernel, _mlpNode)) return;

	learnFlag = false;
	initFlag = true;
	learnRate = _learnRate;
	errorRate = _errorRate;

	t_clayer = _cnnNode.rows - 1;
	t_mlayer = _mlpNode.rows - 1;
	t_sample = nSample;

	itor = _itor;

	int out_w = 0, out_h = 0, out_m = 0;
	for(int i = 0; i < t_clayer; ++i){
		int in_w = _cnnNode.at<int>(i, 0);
		int in_h = _cnnNode.at<int>(i, 1);
		int in_m = _cnnNode.at<int>(i, 2);
		int kw = _kernel.at<int>(i, 0);
		out_w = _cnnNode.at<int>(i + 1, 0);
		out_h = _cnnNode.at<int>(i + 1, 1);
		out_m = _cnnNode.at<int>(i + 1, 2);

		ioNode.push_back(MemBlock4D<double>(in_w, in_h, in_m, t_sample));
		kernel.push_back(MemBlock4D<double>(kw, kw, in_m, out_m));
		convNode.push_back(MemBlock4D<double>(in_w - kw + 1, in_h - kw + 1, out_m, t_sample));
		convDelta.push_back(MemBlock4D<double>(in_w - kw + 1, in_h - kw + 1, out_m, t_sample));
		ioDelta.push_back(MemBlock3D<double>(out_w, out_h, out_m));
		padDelta.push_back(MemBlock3D<double>(in_w + kw - 1, in_h + kw - 1, out_m));
		poolMark.push_back(MemBlock4D<uchar>(out_w, out_h, out_m, t_sample));
		cnnBias.push_back(new double[out_m]);
	}
	ioNode.push_back(MemBlock4D<double>(out_w, out_h, out_m, t_sample));
	lastDelta.Set(out_w, out_h, out_m, t_sample);

	int upper = 0;
	for(int i = 0; i < t_mlayer; ++i){
		int under = _mlpNode.at<int>(i);
		upper = _mlpNode.at<int>(i + 1);
		
		if(i == 0){
			mlpNode.push_back(MemBlock2D<double>(under, t_sample, ioNode[t_clayer].memBlk1D));
			mlpDelta.push_back(Mem2D(under, t_sample, lastDelta.memBlk1D));
		}
		else{
			mlpNode.push_back(MemBlock2D<double>(under, t_sample));
		}
		mlpDelta.push_back(MemBlock2D<double>(upper, t_sample));
		weight.push_back(MemBlock2D<double>(under, upper));
		mlpBias.push_back(new double[upper]);
	}
	mlpNode.push_back(MemBlock2D<double>(upper, t_sample));
	
	cv::RNG rng(time(NULL));
	for(int i = 0; i < t_clayer; ++i){
		rng.fill(kernel[i].mat, cv::RNG::UNIFORM, -0.5, 0.5);
		for(int k = 0; k < kernel[i].dim4; ++k)
			cnnBias[i][k] = rng.uniform(-0.5, 0.5);
	}
	
	for(int i = 0; i < t_mlayer; ++i){
		if(i < t_mlayer - 1){
			rng.fill(weight[i].mat, cv::RNG::UNIFORM, -0.5, 0.5);
			for(int k = 0; k < weight[i].dim2; ++k)
				mlpBias[i][k] = rng.uniform(-0.5, 0.5);
		}
		else{
			rng.fill(weight[i].mat, cv::RNG::UNIFORM, 0, 1);
			for(int k = 0; k < weight[i].dim2; ++k)
				mlpBias[i][k] = rng.uniform(0, 1);
		}
	}
}

CvCNN::~CvCNN(){
	if(initFlag){
		for(int i = 0; i < t_clayer; ++i) 
			delete[] cnnBias[i];
		for(int i = 0; i < t_mlayer; ++i)
			delete[] mlpBias[i];
	}
}


bool CvCNN::Check(cv::Mat &_cnnNode, cv::Mat &_kernel, cv::Mat _mlpNode){
	bool _initFlag = true;

	if(_cnnNode.rows - 1 != _kernel.rows){
		cout << "노드 개수와 커널 개수가 맞지 않습니다.\n";
		_initFlag = false;
	}
	else if(_cnnNode.cols != 3){
		cout << "노드 구성이 잘못 됐습니다.\n";
		_initFlag = false;
	}
	else if(_kernel.cols != 1){
		cout << "커널 구성이 잘못 됐습니다.\n";
		_initFlag = false;
	}
	else if(_cnnNode.type() != CV_32SC1 || _kernel.type() != CV_32SC1 || _mlpNode.type() != CV_32SC1){
		cout << "노드와 커널 구성 인자가 int형이 아닙니다.\n";
		_initFlag = false;
	}
	else{
		// 인풋 개수
		int in_m = _cnnNode.at<int>(0, 2);
		// 인풋 가로
		int in_w = _cnnNode.at<int>(0, 0);
		// 인풋 세로
		int in_h = _cnnNode.at<int>(0, 1);

		if(in_m != 1){
			cout << "입력 채널은 1채널 이어야 합니다.\n";
			_initFlag = false;
		}
		else{
			for(int i = 0, n = 1; i < _kernel.rows; ++i, ++n){
				// 커널 가로 및 세로
				int kw = _kernel.at<int>(i, 0);
				//출력 개수
				int out_m = _cnnNode.at<int>(n, 2);
				// 출력 가로
				int out_w = _cnnNode.at<int>(n, 0);
				// 출력 세로
				int out_h = _cnnNode.at<int>(n, 1);

				if(in_w < kw || in_h < kw){
					cout << i << "번 레이어 커널이 입력 사이즈 보다 큽니다.\n";
					_initFlag = false;
					break;
				}
				else if(out_w * 2 != in_w - kw + 1){
					cout << i + 1 << "번 레이어 입력 가로 사이즈가 맞지 않습니다.\n";
					_initFlag = false;
					break;
				}
				else if(out_h * 2 != in_h - kw + 1){
					cout << i + 1 << "번 레이어 입력 세로 사이즈가 맞지 않습니다.\n";
					_initFlag = false;
					break;
				}
				in_m = out_m;
				in_w = out_w;
				in_h = out_w;
			}
			if(_mlpNode.at<int>(0) != (in_w * in_h * in_m)){
				cout << "CNN 출력과 MLP 입력 개수가 안맞습니다.\n";
				_initFlag = false;
			}
		}
	}

	return _initFlag;
}

void CvCNN::MoveData(const MemBlock3D<uchar> &sample){
	tick_count t0 = tick_count::now();

	parallel_for(blocked_range<int>(0, t_sample, (t_sample + 3) / 4),
		[&](blocked_range<int> q){
			Mem4D &input = ioNode[0];

			for(int i = q.begin(); i < q.end(); ++i){
				double **inPt = input.memBlk4D[i][0];
				uchar **sPt = sample.memBlk3D[i];
				for(int h = 0; h < input.dim2; ++h){
					for(int w = 0; w < input.dim1; ++w){
						inPt[h][w] = (double)sPt[h][w] / 255;
					}
				}
			}
	});

	cout << "샘플 변환 완료" << (tick_count::now() - t0).seconds() * 1000 << "ms" << endl;
}

void CvCNN::Trainning(Mem3UC &samples, cv::Mat &target){
	if(!initFlag) return;

	MoveData(samples);

	vector<Mem2D> note;
	for(int i = 0; i < 4; ++i) note.push_back(Mem2D(128, 64));
	tick_count start = tick_count::now();
	tick_count t0, t1;

	for(int n = 0; n < itor; ++n){
		t0 = tick_count::now();

		learnFlag = true;
		
		//****************Foreward**********************
		for(int cl = 0; cl < t_clayer; ++cl){
			//****************CNN**********************			
#if (CONV_MODEL == 1)	
			//****************Convolution MODEL1*********
			parallel_for(blocked_range<int>(0, t_sample, (t_sample + 3) / 4),
				[&](blocked_range<int> q){
					MemBlock4D<double> &input = ioNode[cl];
					MemBlock4D<double> &kern = kernel[cl];
					MemBlock4D<double> &conv = convNode[cl];
					double *biasPt = cnnBias[cl];
					double _kernel[7][7];
					
					for(int up = 0; up < kern.dim4; ++up){
						double _bias = biasPt[up];
						for(int dw = 0; dw < kern.dim3; ++dw){
							double **kernelPt = kern.memBlk4D[up][dw];
							for(int h = 0; h < kern.dim2; ++h){
								for(int w = 0; w < kern.dim1; ++w){
									_kernel[h][w] = kernelPt[h][w];
								}
							}

							for(int s = q.begin(); s < q.end(); ++s){
								double **inputPt = input.memBlk4D[dw][s];
								double **convPt = conv.memBlk4D[up][s];
								for(int y = 0; y < conv.dim2; ++y){
									for(int x = 0; x < conv.dim1; ++x){
										double _sum = 0;
										for(int h = 0; h < kern.dim2; ++h){
											for(int w = 0; w < kern.dim1; ++w){
												_sum += inputPt[y + h][x + w] * _kernel[h][w];
											}
										}
										if(up) convPt[y][x] += _sum;
										else convPt[y][x] = _sum + _bias;
									}
								}
							}
						}
					}
			});	

#elif (CONV_MODEL == 2)
			//****************Convolution MODEL2****************
			parallel_for(blocked_range<int>(0, t_sample, (t_sample + 3) / 4),
				[&](blocked_range<int> q){
					MemBlock4D<double> &input = ioNode[cl];
					MemBlock4D<double> &_kernel = kernel[cl];
					MemBlock4D<double> &output = convNode[cl];
					Mem2D &notePt = note[q.begin() / (t_sample / 4)];
					double *biasPt = cnnBias[cl];

					for(int i = q.begin(); i < q.end(); ++i){
						for(int om = 0; om < output.dim3; ++om){
							for(int im = 0; im < input.dim3; ++im){
								for(int y = 0; y < output.dim2; ++y){
									for(int h = 0; h < _kernel.dim2; ++h){
										for(int x = 0; x < output.dim1; ++x){
											for(int w = 0; w < _kernel.dim1; ++w){
												if(im) output.memBlk4D[i][om][y][x] += input.memBlk4D[i][im][y + h][x + w] * _kernel.memBlk4D[om][im][h][w];
												else output.memBlk4D[i][om][y][x] = input.memBlk4D[i][im][y + h][x + w] * _kernel.memBlk4D[om][im][h][w];
											}
										}
									}
								}
							}
							for(int y = 0; y < output.dim2; ++y){
								for(int x = 0; x < output.dim1; ++x){
									output.memBlk4D[i][om][y][x] += biasPt[om];
								}
							}
						}
					}
			});
#endif
			//****************Active**************
			parallel_for(blocked_range<int>(0, convNode[cl].t_elements, (convNode[cl].t_elements + 3) / 4),
				[&](blocked_range<int> q){
					double *convPt = convNode[cl].memBlk1D;

					for(int i = q.begin(); i < q.end(); ++i) convPt[i] = max(0, convPt[i]);
			});

#if (POOL_MODEL == 1)
			//****************Pooling MODEL1*************
			parallel_for(blocked_range<int>(0, t_sample, (t_sample + 3) / 4),
				[&](blocked_range<int> q){
					MemBlock4D<double> &conv = convNode[cl];
					MemBlock4D<double> &output = ioNode[cl + 1];
					MemBlock4D<uchar> &mark = poolMark[cl];

					for(int m = 0; m < output.dim4; ++m){
						for(int s = q.begin(); s < q.end(); ++s){
							double **convPt = conv.memBlk4D[m][s];
							double **outputPt = output.memBlk4D[m][s];
							uchar **markPt = mark.memBlk4D[m][s];
							for(int y = 0; y < output.dim2; ++y){
								for(int x = 0; x < output.dim1; ++x){
									double _max = -1;
									uchar _count = 0;
									uchar idx = 0;
									for(int h = 0; h < 2; ++h){
										for(int w = 0; w < 2; ++w){
											double current = convPt[y * 2 + h][x * 2 + w];
											if(current > _max){
												_max = current;
												idx = _count;
											}
											++_count;
										}
									}
									outputPt[y][x] = _max;
									markPt[y][x] = idx;
								}
							}
						}
					}
			});

#elif (POOL_MODEL == 2)
			//****************Pooling MODEL2*************
			parallel_for(blocked_range<int>(0, t_sample, (t_sample + 3) / 4),
				[&](blocked_range<int> q){
					MemBlock4D<double> &input = convNode[cl];
					MemBlock4D<double> &output = ioNode[cl + 1];
					MemBlock4D<uchar> &mark = poolMark[cl];

					for(int i = q.begin(); i < q.end(); ++i){
						for(int om = 0; om < output.dim3; ++om){
							for(int y = 0; y < output.dim2; ++y){
								for(uchar h = 0; h < 2; ++h){
									int in_y = y * 2 + h;
									for(int x = 0; x < output.dim1; ++x){
										if(h == 0) output.memBlk4D[i][om][y][x] = 0;
										for(uchar w = 0; w < 2; ++w){
											int in_x = x * 2 + w;
											if(output.memBlk4D[i][om][y][x] < input.memBlk4D[i][om][in_y][in_x]){
												output.memBlk4D[i][om][y][x] = input.memBlk4D[i][om][in_y][in_x];
												mark.memBlk4D[i][om][y][x] = h * 2 + w;
											}
										}
									}
								}
							}
						}
					}
			});
#endif
		}

		//****************MLP**********************
		for(int ml = 0; ml < t_mlayer; ++ml){
			//***************Multy MODEL1*****************
			parallel_for(blocked_range<int>(0, t_sample, (t_sample + 3) / 4),
				[&](blocked_range<int> q){
					MemBlock2D<double> &input = mlpNode[ml];
					MemBlock2D<double> &_weight = weight[ml];
					MemBlock2D<double> &output = mlpNode[ml + 1];
					double *biasPt = mlpBias[ml];

					for(int i = q.begin(); i < q.end(); ++i){
						double *outPt = output.memBlk2D[i];
						double *inPt = input.memBlk2D[i];
						for(int up = 0; up < _weight.dim2; ++up){
							double _sum = 0;
							for(int dw = 0; dw < _weight.dim1; ++dw){
								_sum += inPt[dw] * _weight.memBlk2D[up][dw];
							}
							outPt[up] = _sum + biasPt[up];
						}
					}
			});
			if(ml < t_mlayer - 1){
				//***************Sigmoid Active****************
				parallel_for(blocked_range<int>(0, mlpNode[ml + 1].t_elements, 
					(mlpNode[ml + 1].t_elements + 3) / 4),
					[&](blocked_range<int> q){
						MemBlock2D<double> &input = mlpNode[ml + 1];
						double *inputPt = input.memBlk1D;
					
						for(int i = q.begin(); i < q.end(); ++i){
							inputPt[i] = 1 / (1 + exp(-inputPt[i]));
						}
				});
			}
			else{
				//***************SoftMax Active****************
				parallel_for(blocked_range<int>(0, t_sample, (t_sample + 3) / 4),
					[&](blocked_range<int> q){
						MemBlock2D<double> &output = mlpNode[ml + 1];
						
						for(int i = q.begin(); i < q.end(); ++i){
							double _sum = 0;
							for(int om = 0; om < output.dim1; ++om) _sum += output.memBlk2D[i][om] = exp(output.memBlk2D[i][om]);
							for(int om = 0; om < output.dim1; ++om) output.memBlk2D[i][om] /= _sum;
						}
				});
			}
		}

		//****************CalculateError****************
		double cost_e = 0;
#if (COST_MODEL == 1)
		//****************CrossEntropy Regression 1****************
		parallel_for(blocked_range<int>(0, t_sample, (t_sample + 3) / 4),
			[&](blocked_range<int> q){
				MemBlock2D<double> &output = mlpNode[t_mlayer - 1];
				double local_e = 0;

				for(int i = q.begin(); i < q.end(); ++i){
					double *outPt = output.memBlk2D[i];
					double *tarPt = target.ptr<double>(i);
					for(int om = 0; om < output.dim1; ++om){
						local_e += tarPt[om] * log(outPt[om]) + (1 - tarPt[om]) * log(1 - outPt[om]);
					}
				}
				costVal += local_e / t_sample;
		});

#elif (COST_MODEL == 2)
		//****************CrossEntropy Regression 2****************
		parallel_for(blocked_range<int>(0, t_sample, (t_sample + 3) / 4),
			[&](blocked_range<int> q){
				MemBlock2D<double> &output = mlpNode[t_mlayer];
				double ce = 0;
				for(int i = q.begin(); i < q.end(); ++i){
					double *outputPt = output.memBlk2D[i];
					double *targetPt = target.ptr<double>(i);
					for(int om = 0; om < output.dim1; ++om){
						if(targetPt[om]) ce -= log(outputPt[om]);
					}
				}
				cost_e += ce / t_sample;
		});
#endif
		if(cost_e < errorRate){
			learnFlag = true;
			t1 = tick_count::now();
			cout << n << "# 학습 종료(" << cost_e << ") " << (t1 - t0).seconds() * 1000 << "ms 경과\n";
			break;
		}
		else{
			//****************Backward*****************
			//****************MLP*****************
			parallel_for(blocked_range<int>(0, t_sample, (t_sample + 3) / 4),
				[&](blocked_range<int> q){
					Mem2D &output = mlpNode[t_mlayer];
					Mem2D &delta = mlpDelta[t_mlayer];
					
					for(int i = q.begin(); i < q.end(); ++i){
						double *tarPt = target.ptr<double>(i);
						for(int om = 0; om < delta.dim1; ++om){
							delta.memBlk2D[i][om] = tarPt[om] - output.memBlk2D[i][om];
						}
					}
			});

			int ml = t_mlayer;
			while(ml > 0){
				parallel_for(blocked_range<int>(0, t_sample, (t_sample + 3) / 4),
					[&](blocked_range<int> q){
						Mem2D &upDelta = mlpDelta[ml];
						double **_weight = weight[ml].memBlk2D;
						Mem2D &downDelta = mlpDelta[ml - 1];

						for(int i = q.begin(); i < q.end(); ++i){
							for(int up = 0; up < upDelta.dim1; ++up){
								for(int dw = 0; dw < downDelta.dim1; ++dw){
									if(up) downDelta.memBlk2D[i][dw] += upDelta.memBlk2D[i][up] * _weight[up][dw];
									else downDelta.memBlk2D[i][dw] = upDelta.memBlk2D[i][up] * _weight[up][dw];
								}
							}
						}
				});

				if(ml > 1){
					parallel_for(blocked_range<int>(0, t_sample, (t_sample + 3) / 4),
						[&](blocked_range<int> q){
							Mem2D &output = mlpNode[ml - 1];
							Mem2D &downDelta = mlpDelta[ml - 1];

							for(int i = q.begin(); i < q.end(); ++i){
								for(int om = 0; om < downDelta.dim1; ++om)
									downDelta.memBlk2D[i][om] *= output.memBlk2D[i][om] * (1 - output.memBlk2D[i][om]);
							}
					});
				}
				--ml;
			}

			//****************CNN*****************
			parallel_for(blocked_range<int>(0, t_sample, (t_sample + 3) / 4),
				[&](blocked_range<int> q){
					Mem4D &outDel = convDelta[t_clayer - 1];
					MemBlock4D<uchar> &mark = poolMark[t_clayer - 1];

					for(int i = q.begin(); i < q.end(); ++i){
						for(int om = 0; om < mark.dim3; ++om){
							for(int y = 0; y < mark.dim2; ++y){
								for(uchar h = 0; h < 2; ++h){
									int out_y = y * 2 + h;
									for(int x = 0; x < mark.dim1; ++x){
										for(uchar w = 0; w < 2; ++w){
											int out_x = x * 2 + w;
											if(mark.memBlk4D[i][om][y][x] == (h * 2 + w))
												outDel.memBlk4D[i][om][out_y][out_x] = lastDelta.memBlk4D[i][om][y][x];
											else
												outDel.memBlk4D[i][om][out_y][out_x] = 0;
										}
									}
								}
							}
						}
					}
			});

			ml = t_clayer;
			while (--ml){
				//************************Unpooling****************************
				parallel_for(blocked_range<int>(0, t_sample, (t_sample + 3) / 4),
					[&](blocked_range<int> q){
						Mem3D &delta = ioDelta[ml];

				});
			}

			//****************ModifyKernel******************

			//****************ModyfieWeight*****************

			t1 = tick_count::now();
			cout << n << "# (" << cost_e << ") " << (t1 - t0).seconds() * 1000 << "ms 경과\n";
		}
	}
	cout << "********* 총 경과 시간 " << (t1 - start).seconds() * 1000 << "ms\n";
}

void CvCNN::UnPooling(Mem3D &input, Mem3D &output, Mem3UC &mark){

}

void CvCNN::Predict(cv::Mat &input, cv::Mat &result){

}

void CvCNN::Save(const char *path){

}

void CvCNN::Load(const char *path){

}
