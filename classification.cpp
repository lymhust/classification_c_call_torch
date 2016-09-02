//============================================================================
// Name        : c_call_torch.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
extern "C"{
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"
#include "luajit.h"
#include "luaT.h"
#include "TH/TH.h"
};

using namespace std;
using namespace cv;

/*
#include "luajit.h"
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"
 */

void init_torch7(lua_State *L)
{
	// Init torch7
	if (luaL_loadfile(L, "./src/luafile/init.lua") || lua_pcall(L, 0, 0, 0))
	{
		printf("lua init error: %s \n", lua_tostring(L, -1));
	}
}

float* run_classify(lua_State *L, THFloatTensor *imgTensor, THFloatTensor *batchTensor,
								  THFloatTensor *boxTensor, THFloatTensor *clsTensor)
{
	luaT_newmetatable(L, "torch.FloatTensor", NULL, NULL, NULL, NULL);

	// Classify
	if (luaL_loadfile(L, "./src/luafile/classify.lua") || lua_pcall(L, 0, 0, 0))
	{
		printf("lua classify file load error: %s \n", lua_tostring(L, -1));
	}

	// Load the lua function
	lua_getglobal(L, "classify");
	if(!lua_isfunction(L,-1))
	{
		cout << "classify is not a function" << endl;
		lua_pop(L,1);
	}

	// This pushes data to the stack to be used as a parameter to the function call
	luaT_pushudata(L, (void *)imgTensor,   "torch.FloatTensor");
	luaT_pushudata(L, (void *)batchTensor, "torch.FloatTensor");
	luaT_pushudata(L, (void *)boxTensor,   "torch.FloatTensor");
	luaT_pushudata(L, (void *)clsTensor,   "torch.FloatTensor");

	// Call the lua function
	if (lua_pcall(L, 4, 1, 0) != 0)
	{
		printf("lua error running function 'classify': %s \n", lua_tostring(L, -1));
	}

	// Get results returned from the lua function
	clsTensor = (THFloatTensor*)luaT_toudata(L, -1, "torch.FloatTensor");
	lua_pop(L, 1);
	THFloatStorage *storage_res =  clsTensor->storage;
	float *clsResult = storage_res->data;

	return clsResult;
}

int main()
{
	Mat img = imread("./src/luafile/test.jpg");
	if(img.empty())
	{
		cout<<"opencv error img is nill";
		return -1;
	}
	cvtColor(img, img, CV_BGR2RGB);
	img.convertTo(img, CV_32FC3);
	img = img / 255;
	float *ptrimg = (float*)img.data; // image pointer

	///////////////////////////////////////////////////////////////////////
	// C call Torch7
	lua_State *L = luaL_newstate();
	luaL_openlibs(L);
	init_torch7(L);

	// Convert the c array to Torch7 specific structure representing a tensor
	THFloatStorage *storage =  THFloatStorage_newWithData(ptrimg, img.rows * img.cols * img.channels());
	THFloatTensor *imgTensor = THFloatTensor_newWithStorage3d(storage, 0, img.rows, img.cols*img.channels(),       //long size0_, long stride0_,
			img.cols, img.channels(),
			img.channels(), 1);
	int boxNum = 20;
	THFloatTensor *batchTensor = THFloatTensor_newWithSize4d(boxNum, 3, 48, 48);
	THFloatTensor *boxTensor =   THFloatTensor_newWithSize2d(boxNum, 4);
	THFloatTensor *clsTensor =   THFloatTensor_newWithSize1d(boxNum);

	float *clsResult = run_classify(L, imgTensor, batchTensor, boxTensor, clsTensor);

	////////////////////////////////////////////////////////////////////////////////

	for(int i=0; i<=boxNum-1; i++)
	{
		cout << clsResult[i] << endl;
	}

	return 0;
}

