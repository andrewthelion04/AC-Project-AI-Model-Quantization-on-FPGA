// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.2 (64-bit)
// Tool Version Limit: 2025.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#ifdef SDT
#include "xparameters.h"
#endif
#include "xdense_int8.h"

extern XDense_int8_Config XDense_int8_ConfigTable[];

#ifdef SDT
XDense_int8_Config *XDense_int8_LookupConfig(UINTPTR BaseAddress) {
	XDense_int8_Config *ConfigPtr = NULL;

	int Index;

	for (Index = (u32)0x0; XDense_int8_ConfigTable[Index].Name != NULL; Index++) {
		if (!BaseAddress || XDense_int8_ConfigTable[Index].Control_BaseAddress == BaseAddress) {
			ConfigPtr = &XDense_int8_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XDense_int8_Initialize(XDense_int8 *InstancePtr, UINTPTR BaseAddress) {
	XDense_int8_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XDense_int8_LookupConfig(BaseAddress);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XDense_int8_CfgInitialize(InstancePtr, ConfigPtr);
}
#else
XDense_int8_Config *XDense_int8_LookupConfig(u16 DeviceId) {
	XDense_int8_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XDENSE_INT8_NUM_INSTANCES; Index++) {
		if (XDense_int8_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XDense_int8_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XDense_int8_Initialize(XDense_int8 *InstancePtr, u16 DeviceId) {
	XDense_int8_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XDense_int8_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XDense_int8_CfgInitialize(InstancePtr, ConfigPtr);
}
#endif

#endif

