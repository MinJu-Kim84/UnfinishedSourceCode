#ifndef ALLOC_BLOCK_MEM_H
#define ALLOC_BLOCK_MEM_H

#include <opencv2\opencv.hpp>
#include "ObjectID.h"
#include "TranceMatType.h"


/***************************************************************
*															   *
*					      MemBlock2D						   *
*															   *
****************************************************************/

template <typename _type_>
class MemBlock2D{
	static Object_ID_List list;
	Object_ID *ID;

	bool shareFlag;

	bool Create(_type_ *data);
	void Clear();
	void AllZero();

public:
	TranceMatType tp;

	_type_ *memBlk1D;
	_type_ **memBlk2D;

	cv::Mat mat;

	int dim1;
	int dim2;
	int t_elements;

	MemBlock2D();
	MemBlock2D(const MemBlock2D &p);
	MemBlock2D(int _dim1, int _dim2, _type_ *data = NULL);
	MemBlock2D(cv::Mat _mat);
	~MemBlock2D();

	void Set(int _dim1, int _dim2, _type_ *data = NULL);
	void Set(cv::Mat _mat);
	cv::Mat GetMat();
	MemBlock2D Clone();
	MemBlock2D& operator=(MemBlock2D &p);
};


template <typename _type_>
Object_ID_List MemBlock2D<_type_>::list;


/******************************* 내부 초기화 함수 **********************************/

template <typename _type_>
void MemBlock2D<_type_>::AllZero(){
	ID = NULL;
	
	shareFlag = false;
	
	memBlk1D = NULL;
	memBlk2D = NULL;

	mat = cv::Mat();

	dim1 = dim2 = t_elements = 0;
}

template <typename _type_>
bool MemBlock2D<_type_>::Create(_type_ *data){
	bool initFlag = false;

	t_elements = dim1 * dim2;
	if(t_elements){
		if(data){
			memBlk1D = data;
			shareFlag = true;
		}
		else{
			memBlk1D = new _type_[t_elements];
			shareFlag = false;
		}
		memBlk2D = new _type_*[dim2];

		if(!(memBlk1D && memBlk2D)){
			if(!shareFlag) delete[] memBlk1D;
			delete[] memBlk2D;
		}
		else{
			_type_ *mem1D_pt = memBlk1D;
			for(int i = 0; i < dim2; ++i){
				memBlk2D[i] = mem1D_pt;
				mem1D_pt += dim1;
			}

			mat = cv::Mat(dim2, dim1, tp.typeVal, memBlk1D);
			ID = list.Create();

			initFlag = true;
		}
	}

	return initFlag;
}

template <typename _type_>
void MemBlock2D<_type_>::Clear(){
	if(ID){
		if(ID->nCpy > 1) ID->nCpy -= 1;
		else{
			if(!shareFlag) delete[] memBlk1D;
			delete[] memBlk2D;

			list.Erase(ID);
			AllZero();
		}
	}
	else{
		AllZero();
	}
}


/******************************* 생성자 및 소멸자 **********************************/

template <typename _type_>
MemBlock2D<_type_>::MemBlock2D()
	: tp(memBlk1D)
{
	AllZero();
}

template <typename _type_>
MemBlock2D<_type_>::MemBlock2D(const MemBlock2D &p)
	: tp(memBlk1D)
{
	ID = p.ID;

	shareFlag = p.shareFlag;

	memBlk1D = p.memBlk1D;
	memBlk2D = p.memBlk2D;

	mat = p.mat;

	dim1 = p.dim1;
	dim2 = p.dim2;
	t_elements = p.t_elements;

	if(ID) ID->nCpy += 1;
}

template <typename _type_>
MemBlock2D<_type_>::MemBlock2D(int _dim1, int _dim2, _type_ *data)
	: tp(memBlk1D)
{
	dim1 = _dim1;
	dim2 = _dim2;

	if(!Create(data)) AllZero();
}

template<typename _type_>
MemBlock2D<_type_>::MemBlock2D(cv::Mat _mat)
	: tp(memBlk1D)
{
	if((_mat.type() == tp.typeVal) && (_mat.depth() == 2)){
		dim1 = _mat.cols;
		dim2 = _mat.rows;

		if(!Create(_mat.ptr<_type_>(0))) AllZero();
	}
	else AllZero();
}

template <typename _type_>
MemBlock2D<_type_>::~MemBlock2D(){
	Clear();
}


/******************************* 나머지 함수 **********************************/

template <typename _type_>
void MemBlock2D<_type_>::Set(int _dim1, int _dim2, _type_ *data){
	Clear();

	dim1 = _dim1;
	dim2 = _dim2;

	if(!Create(data)) AllZero();
}

template <typename _type_>
void MemBlock2D<_type_>::Set(cv::Mat _mat){
	Clear();

	if(_mat.type() == tp.typeVal){
		dim1 = _mat.cols;
		dim2 = _mat.rows;

		if(!Create(_mat.ptr<_type_>(0))) AllZero();
	}
	else AllZero();
}

template <typename _type_>
MemBlock2D<_type_> MemBlock2D<_type_>::Clone(){
	MemBlock2D<_type_> tmp(dim1, dim2);

	memcpy_s(tmp.memBlk1D, tmp.t_elements * sizeof(_type_), memBlk1D, t_elements * sizeof(_type_));

	return tmp;
}

template <typename _type_>
cv::Mat MemBlock2D<_type_>::GetMat(){
	return mat;
}

template <typename _type_>
MemBlock2D<_type_>& MemBlock2D<_type_>::operator=(MemBlock2D &p){
	if(this == &p) return *this;

	Clear();

	ID = p.ID;

	shareFlag = p.shareFlag;

	memBlk1D = p.memBlk1D;
	memBlk2D = p.memBlk2D;

	mat = p.mat;

	dim1 = p.dim1;
	dim2 = p.dim2;
	t_elements = p.t_elements;

	if(ID) ID->nCpy += 1;

	return *this;
}



/***************************************************************
*															   *
*					      MemBlock3D						   *
*															   *
****************************************************************/

template <typename _type_>
class MemBlock3D{
	static Object_ID_List list;
	Object_ID *ID;

	bool shareFlag;

	bool Create(_type_ *data);
	void Clear();
	void AllZero();

public:
	TranceMatType tp;

	_type_ *memBlk1D;
	_type_ **memBlk2D;
	_type_ ***memBlk3D;

	int dim1;
	int dim2;
	int dim3;
	int t_elements;

	cv::Mat mat;

	MemBlock3D();
	MemBlock3D(const MemBlock3D &p);
	MemBlock3D(int _dim1, int _dim2, int _dim3, _type_ *data = NULL);
	MemBlock3D(cv::Mat _mat);
	~MemBlock3D();

	void Set(int _dim1, int _dim2, int _dim3, _type_ *data = NULL);
	void Set(cv::Mat _mat);
	cv::Mat GetMat();
	MemBlock3D Clone();
	MemBlock3D& operator=(MemBlock3D &p);
};


template <typename _type_>
Object_ID_List MemBlock3D<_type_>::list;


/******************************* 내부 초기화 함수 **********************************/

template <typename _type_>
bool MemBlock3D<_type_>::Create(_type_ *data){
	bool initFlag = false;

	t_elements = dim1 * dim2 * dim3;
	int t_dim2 = dim2 * dim3;

	if(t_elements){
		if(data){
			memBlk1D = data;
			shareFlag = true;
		}
		else{
			memBlk1D = new _type_[t_elements];
			shareFlag = false;
		}
		memBlk2D = new _type_*[t_dim2];
		memBlk3D = new _type_**[dim3];

		if(!(memBlk1D && memBlk2D && memBlk3D)){
			if(!shareFlag) delete[] memBlk1D;
			delete[] memBlk2D;
			delete[] memBlk3D;
		}
		else{
			_type_ *mem1D_pt = memBlk1D;
			for(int i = 0; i < t_dim2; ++i){
				memBlk2D[i] = mem1D_pt;
				mem1D_pt += dim1;
			}

			_type_ **mem2D_pt = memBlk2D;
			for(int i = 0; i < dim3; ++i){
				memBlk3D[i] = mem2D_pt;
				mem2D_pt += dim2;
			}

			int dimArr[3] = {dim3, dim2, dim1};
			mat = cv::Mat(3, dimArr, tp.typeVal, memBlk1D);
			ID = list.Create();

			initFlag = true;
		}
	}

	return initFlag;
}

template <typename _type_>
void MemBlock3D<_type_>::AllZero(){
	ID = NULL;

	shareFlag = false;

	memBlk1D = NULL;
	memBlk2D = NULL;
	memBlk3D = NULL;

	mat = cv::Mat();

	dim1 = dim2 = dim3 = t_elements = 0;
}

template <typename _type_>
void MemBlock3D<_type_>::Clear(){
	if(ID){
		if(ID->nCpy > 1) ID->nCpy -= 1;
		else{
			if(!shareFlag) delete[] memBlk1D;
			delete[] memBlk2D;
			delete[] memBlk3D;

			list.Erase(ID);
			AllZero();
		}
	}
	else{
		AllZero();
	}
}


/******************************* 생성자 및 소멸자 **********************************/

template <typename _type_>
MemBlock3D<_type_>::MemBlock3D()
	: tp(memBlk1D)
{
	AllZero();
}

template <typename _type_>
MemBlock3D<_type_>::MemBlock3D(const MemBlock3D &p)
	: tp(memBlk1D)
{
	ID = p.ID;

	shareFlag = p.shareFlag;

	memBlk1D = p.memBlk1D;
	memBlk2D = p.memBlk2D;
	memBlk3D = p.memBlk3D;

	mat = p.mat;

	dim1 = p.dim1;
	dim2 = p.dim2;
	dim3 = p.dim3;
	t_elements = p.t_elements;

	if(ID) ID->nCpy += 1;
}

template <typename _type_>
MemBlock3D<_type_>::MemBlock3D(int _dim1, int _dim2, int _dim3, _type_ *data)
	: tp(memBlk1D)
{
	dim1 = _dim1;
	dim2 = _dim2;
	dim3 = _dim3;
	
	if(!Create(data)) AllZero();
}

template <typename _type_>
MemBlock3D<_type_>::MemBlock3D(cv::Mat _mat)
	: tp(memBlk1D)
{
	if((_mat.type() == tp.typeVal) && (_mat.depth() == 3)){
		dim1 = _mat.size.p[2];
		dim2 = _mat.size.p[1];
		dim3 = _mat.size.p[0];

		if(!Create(_mat.ptr<_type_>(0))) AllZero();
	}
	else AllZero();
}

template <typename _type_>
MemBlock3D<_type_>::~MemBlock3D(){
	Clear();
}


/******************************* 나머지 함수 **********************************/

template <typename _type_>
void MemBlock3D<_type_>::Set(int _dim1, int _dim2, int _dim3, _type_ *data){
	Clear();

	dim1 = _dim1;
	dim2 = _dim2;
	dim3 = _dim3;
	
	if(!Create(data)) AllZero();
}

template <typename _type_>
void MemBlock3D<_type_>::Set(cv::Mat _mat){
	Clear();

	if((_mat.type() == tp.typeVal) && (_mat.depth() == 3)){
		dim1 = _mat.size.p[2];
		dim2 = _mat.size.p[1];
		dim3 = _mat.size.p[0];

		if(!Create(_mat.ptr<_type_>(0))) AllZero();
	}
	else AllZero();
}

template <typename _type_>
MemBlock3D<_type_> MemBlock3D<_type_>::Clone(){
	MemBlock3D<_type_> tmp(dim1, dim2, dim3);

	memcpy_s(tmp.memBlk1D, tmp.t_elements * sizeof(_type_), memBlk1D, t_elements * sizeof(_type_));

	return tmp;
}

template <typename _type_>
MemBlock3D<_type_>& MemBlock3D<_type_>::operator=(MemBlock3D &p){
	if(this == &p) return *this;

	Clear();

	ID = p.ID;

	shareFlag = p.shareFlag;

	memBlk1D = p.memBlk1D;
	memBlk2D = p.memBlk2D;
	memBlk3D = p.memBlk3D;

	mat = p.mat;

	dim1 = p.dim1;
	dim2 = p.dim2;
	dim3 = p.dim3;
	t_elements = p.t_elements;

	if(ID) ID->nCpy += 1;

	return *this;
}

template <typename _type_>
cv::Mat MemBlock3D<_type_>::GetMat(){
	return mat;
}

/***************************************************************
*															   *
*					      MemBlock4D						   *
*															   *
****************************************************************/

template <typename _type_>
class MemBlock4D{
	static Object_ID_List list;
	Object_ID *ID;

	bool shareFlag;

	bool Create(_type_ *data);
	void Clear();
	void AllZero();

public:
	TranceMatType tp;

	_type_ *memBlk1D;
	_type_ **memBlk2D;
	_type_ ***memBlk3D;
	_type_ ****memBlk4D;

	cv::Mat mat;

	int dim1;
	int dim2;
	int dim3;
	int dim4;
	int t_elements;

	MemBlock4D();
	MemBlock4D(const MemBlock4D &p);
	MemBlock4D(int _dim1, int _dim2, int _dim3, int _dim4, _type_ *data = NULL);
	MemBlock4D(cv::Mat _mat);
	~MemBlock4D();

	void Set(int _dim1, int _dim2, int _dim3, int _dim4, _type_ *data = NULL);
	void Set(cv::Mat _mat);
	cv::Mat GetMat();
	MemBlock4D Clone();
	MemBlock4D& operator=(MemBlock4D &p);
};


template <typename _type_>
Object_ID_List MemBlock4D<_type_>::list;


/******************************* 내부 초기화 함수 **********************************/

template <typename _type_>
bool MemBlock4D<_type_>::Create(_type_ *data){
	bool initFlag = false;

	t_elements = dim1 * dim2 * dim3 * dim4;
	int t_dim2 = dim2 * dim3 * dim4;
	int t_dim3 = dim3 * dim4;

	if(t_elements){
		if(data){
			memBlk1D = data;
			shareFlag = true;
		}
		else{
			memBlk1D = new _type_[t_elements];
			shareFlag = false;
		}
		memBlk2D = new _type_*[t_dim2];
		memBlk3D = new _type_**[t_dim3];
		memBlk4D = new _type_***[dim4];

		if(!(memBlk1D && memBlk2D && memBlk3D && memBlk4D)){
			if(!shareFlag) delete[] memBlk1D;
			delete[] memBlk2D;
			delete[] memBlk3D;
			delete[] memBlk4D;
		}
		else{
			_type_ *mem1D_pt = memBlk1D;
			for(int i = 0; i < t_dim2; ++i){
				memBlk2D[i] = mem1D_pt;
				mem1D_pt += dim1;
			}

			_type_ **mem2D_pt = memBlk2D;
			for(int i = 0; i < t_dim3; ++i){
				memBlk3D[i] = mem2D_pt;
				mem2D_pt += dim2;
			}

			_type_ ***mem3D_pt = memBlk3D;
			for(int i = 0; i < dim4; ++i){
				memBlk4D[i] = mem3D_pt;
				mem3D_pt += dim3;
			}

			int dimArr[4] = {dim4, dim3, dim2, dim1};
			mat = cv::Mat(4, dimArr, tp.typeVal, memBlk1D);
			ID = list.Create();

			initFlag = true;
		}
	}

	return initFlag;
}

template <typename _type_>
void MemBlock4D<_type_>::AllZero(){
	ID = NULL;

	shareFlag = false;

	memBlk1D = NULL;
	memBlk2D = NULL;
	memBlk3D = NULL;
	memBlk4D = NULL;

	mat = cv::Mat();

	dim1 = dim2 = dim3 = dim4 =  t_elements = 0;
}

template <typename _type_>
void MemBlock4D<_type_>::Clear(){
	if(ID){
		if(ID->nCpy > 1) ID->nCpy -= 1;
		else{
			if(!shareFlag) delete[] memBlk1D;
			delete[] memBlk2D;
			delete[] memBlk3D;
			delete[] memBlk4D;

			list.Erase(ID);
			AllZero();
		}
	}
	else{
		AllZero();
	}
}


/******************************* 생성자 및 소멸자 **********************************/

template <typename _type_>
MemBlock4D<_type_>::MemBlock4D()
	: tp(memBlk1D)
{
	AllZero();
}

template <typename _type_>
MemBlock4D<_type_>::MemBlock4D(const MemBlock4D &p)
	: tp(memBlk1D)
{
	ID = p.ID;

	shareFlag = p.shareFlag;

	memBlk1D = p.memBlk1D;
	memBlk2D = p.memBlk2D;
	memBlk3D = p.memBlk3D;
	memBlk4D = p.memBlk4D;

	mat = p.mat;

	dim1 = p.dim1;
	dim2 = p.dim2;
	dim3 = p.dim3;
	dim4 = p.dim4;
	t_elements = p.t_elements;

	if(ID) ID->nCpy += 1;
}

template <typename _type_>
MemBlock4D<_type_>::MemBlock4D(int _dim1, int _dim2, int _dim3, int _dim4, _type_ *data)
	: tp(memBlk1D)
{
	dim1 = _dim1;
	dim2 = _dim2;
	dim3 = _dim3;
	dim4 = _dim4;
	
	if(!Create(data)) AllZero();
}

template <typename _type_>
MemBlock4D<_type_>::MemBlock4D(cv::Mat _mat)
	: tp(memBlk1D)
{
	if((_mat.type() == tp.typeVal) && (_mat.depth() == 4)){
		dim1 = _mat.size.p[3];
		dim2 = _mat.size.p[2];
		dim3 = _mat.size.p[1];
		dim4 = _mat.size.p[0];

		if(!Create(_mat.ptr<_type_>(0))) AllZero();
	}
	else AllZero();
}

template <typename _type_>
MemBlock4D<_type_>::~MemBlock4D(){
	Clear();
}


/******************************* 나머지 함수 **********************************/

template <typename _type_>
void MemBlock4D<_type_>::Set(int _dim1, int _dim2, int _dim3, int _dim4, _type_ *data){
	Clear();

	dim1 = _dim1;
	dim2 = _dim2;
	dim3 = _dim3;
	dim4 = _dim4;
	
	if(!Create(data)) AllZero();
}

template <typename _type_>
void MemBlock4D<_type_>::Set(cv::Mat _mat){
	Clear();

	if((_mat.type() == tp.typeVal) && (_mat.depth() == 4)){
		dim1 = _mat.size.p[3];
		dim2 = _mat.size.p[2];
		dim3 = _mat.size.p[1];
		dim4 = _mat.size.p[0];

		if(!Create(_mat.ptr<_type_>(0))) AllZero();
	}
	else AllZero();
}

template <typename _type_>
MemBlock4D<_type_> MemBlock4D<_type_>::Clone(){
	MemBlock4D<_type_> tmp(dim1, dim2, dim3, dim4);

	memcpy_s(tmp.memBlk1D, tmp.t_elements * sizeof(_type_), memBlk1D, t_elements * sizeof(_type_));

	return tmp;
}

template <typename _type_>
MemBlock4D<_type_>& MemBlock4D<_type_>::operator=(MemBlock4D &p){
	if(this == &p) return *this;

	Clear();

	ID = p.ID;

	shareFlag = p.shareFlag;

	memBlk1D = p.memBlk1D;
	memBlk2D = p.memBlk2D;
	memBlk3D = p.memBlk3D;
	memBlk4D = p.memBlk4D;

	mat = p.mat;

	dim1 = p.dim1;
	dim2 = p.dim2;
	dim3 = p.dim3;
	dim4 = p.dim4;
	t_elements = p.t_elements;

	if(ID) ID->nCpy += 1;

	return *this;
}

template <typename _type_>
cv::Mat MemBlock4D<_type_>::GetMat(){
	return mat;
}

#endif