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
	// Load network
	if (luaL_loadfile(L, "./src/luafile/init_network.lua") || lua_pcall(L, 0, 0, 0))
	{
		printf("lua init error: %s \n", lua_tostring(L, -1));
	}

	// Load classify
	if (luaL_loadfile(L, "./src/luafile/classify.lua") || lua_pcall(L, 0, 0, 0))
	{
		printf("lua classify file load error: %s \n", lua_tostring(L, -1));
	}

	//luaT_newmetatable(L, "torch.FloatTensor", NULL, NULL, NULL, NULL);
	//luaT_newmetatable(L, "torch.IntTensor", NULL, NULL, NULL, NULL);

}

void run_classify(lua_State *L, THFloatTensor *imgTensor, THIntTensor *boxTensor, THIntTensor *clsTensor)
{
	// Load the lua function
	lua_getglobal(L, "classify");
	if(!lua_isfunction(L,-1))
	{
		cout << "classify is not a function" << endl;
		lua_pop(L,1);
	}

	// This pushes data to the stack to be used as a parameter to the function call
	luaT_pushudata(L, (void *)imgTensor, "torch.FloatTensor");
	luaT_pushudata(L, (void *)boxTensor, "torch.IntTensor");
	luaT_pushudata(L, (void *)clsTensor, "torch.IntTensor");

	cout << "Number of stack before: " << lua_gettop(L) << endl;

	// Call the lua function
	if (lua_pcall(L, 3, 0, 0) != 0)
	{
		printf("lua error running function 'classify': %s \n", lua_tostring(L, -1));
	}

	cout << "Number of stack after: " << lua_gettop(L) << endl;
	//lua_settop(L, 0);
	// Get results returned from the lua function
	// clsTensor = (THIntTensor*)luaT_toudata(L, -1, "torch.IntTensor");
	// lua_pop(L, 1);
}

int main()
{
	const char *label[] = {"light","pedestrain","vehicle","i2","i4","i5","il100","il60","il80","il90",
			"io","ip","p10","p11","p12","p19","p23","p26","p27","p3","p5","p6","pg","ph4",
			"ph4.5","ph5","pl100","pl120","pl20","pl30","pl40","pl5","pl50","pl60","pl70",
			"pl80","pm20","pm30","pm55","pn","pne","po","pr40","w13","w32","w55","w57","w59","wo"};

	///////////////////////////////////////////////////////////////////////
	// C call Torch7 init
	lua_State *L = luaL_newstate();
	luaL_openlibs(L);
	init_torch7(L);
	///////////////////////////////////////////////////////////////////////

	for(int time=0; time<10; time++)
	{
		// system("pwd");
		Mat img = imread("./src/luafile/s.jpg");
		if(img.empty())
		{
			cout<<"opencv error img is nill";
			return -1;
		}
		cvtColor(img, img, CV_BGR2RGB);
		img.convertTo(img, CV_32FC3);
		img = img / 255;
		float *ptrimg = (float*)img.data; // image pointer

		// C call Torch7
		// Convert the c array to Torch7 specific structure representing a tensor
		THFloatStorage *storage =  THFloatStorage_newWithData(ptrimg, img.rows * img.cols * img.channels());
		THFloatTensor *imgTensor = THFloatTensor_newWithStorage3d(storage, 0, img.rows, img.cols*img.channels(),       //long size0_, long stride0_,
				img.cols, img.channels(),
				img.channels(), 1);
		int boxNum = 20;
		THIntTensor *boxTensor =   THIntTensor_newWithSize2d(boxNum, 4);
		THIntTensor *clsTensor =   THIntTensor_newWithSize1d(boxNum);
		THIntTensor_fill(boxTensor, -1);
		THIntTensor_fill(clsTensor, -1);

		// Load bbox data
		int *boxptr = THIntTensor_data(boxTensor);
		for(int i=0; i<boxNum*4; i=i+4)
		{
			boxptr[i] = 1;      // left
			boxptr[i+1] = 1;    // top
			boxptr[i+2] = 187;  // w
			boxptr[i+3] = 191;  // h
		}
		boxptr = NULL;

		run_classify(L, imgTensor, boxTensor, clsTensor);
		////////////////////////////////////////////////////////////////////////////////


		for(int i=0; i<=boxNum-1; i++)
		{
			int indx = THIntTensor_get1d(clsTensor, i) - 1;
			if(indx != -1) { cout << label[indx] << endl; }
			else { break; }
		}

		cout << time << endl;
	}

	lua_close(L);
	return 0;
}



