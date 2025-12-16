// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.2 (64-bit)
// Tool Version Limit: 2025.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef XDENSE_INT8_H
#define XDENSE_INT8_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xdense_int8_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
#ifdef SDT
    char *Name;
#else
    u16 DeviceId;
#endif
    u64 Control_BaseAddress;
} XDense_int8_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XDense_int8;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XDense_int8_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XDense_int8_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XDense_int8_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XDense_int8_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
#ifdef SDT
int XDense_int8_Initialize(XDense_int8 *InstancePtr, UINTPTR BaseAddress);
XDense_int8_Config* XDense_int8_LookupConfig(UINTPTR BaseAddress);
#else
int XDense_int8_Initialize(XDense_int8 *InstancePtr, u16 DeviceId);
XDense_int8_Config* XDense_int8_LookupConfig(u16 DeviceId);
#endif
int XDense_int8_CfgInitialize(XDense_int8 *InstancePtr, XDense_int8_Config *ConfigPtr);
#else
int XDense_int8_Initialize(XDense_int8 *InstancePtr, const char* InstanceName);
int XDense_int8_Release(XDense_int8 *InstancePtr);
#endif

void XDense_int8_Start(XDense_int8 *InstancePtr);
u32 XDense_int8_IsDone(XDense_int8 *InstancePtr);
u32 XDense_int8_IsIdle(XDense_int8 *InstancePtr);
u32 XDense_int8_IsReady(XDense_int8 *InstancePtr);
void XDense_int8_EnableAutoRestart(XDense_int8 *InstancePtr);
void XDense_int8_DisableAutoRestart(XDense_int8 *InstancePtr);

void XDense_int8_Set_input_image(XDense_int8 *InstancePtr, u64 Data);
u64 XDense_int8_Get_input_image(XDense_int8 *InstancePtr);
void XDense_int8_Set_output_scores(XDense_int8 *InstancePtr, u64 Data);
u64 XDense_int8_Get_output_scores(XDense_int8 *InstancePtr);

void XDense_int8_InterruptGlobalEnable(XDense_int8 *InstancePtr);
void XDense_int8_InterruptGlobalDisable(XDense_int8 *InstancePtr);
void XDense_int8_InterruptEnable(XDense_int8 *InstancePtr, u32 Mask);
void XDense_int8_InterruptDisable(XDense_int8 *InstancePtr, u32 Mask);
void XDense_int8_InterruptClear(XDense_int8 *InstancePtr, u32 Mask);
u32 XDense_int8_InterruptGetEnabled(XDense_int8 *InstancePtr);
u32 XDense_int8_InterruptGetStatus(XDense_int8 *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
