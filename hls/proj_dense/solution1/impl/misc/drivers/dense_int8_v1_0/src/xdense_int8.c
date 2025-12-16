// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.2 (64-bit)
// Tool Version Limit: 2025.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
/***************************** Include Files *********************************/
#include "xdense_int8.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XDense_int8_CfgInitialize(XDense_int8 *InstancePtr, XDense_int8_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XDense_int8_Start(XDense_int8 *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDense_int8_ReadReg(InstancePtr->Control_BaseAddress, XDENSE_INT8_CONTROL_ADDR_AP_CTRL) & 0x80;
    XDense_int8_WriteReg(InstancePtr->Control_BaseAddress, XDENSE_INT8_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XDense_int8_IsDone(XDense_int8 *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDense_int8_ReadReg(InstancePtr->Control_BaseAddress, XDENSE_INT8_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XDense_int8_IsIdle(XDense_int8 *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDense_int8_ReadReg(InstancePtr->Control_BaseAddress, XDENSE_INT8_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XDense_int8_IsReady(XDense_int8 *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDense_int8_ReadReg(InstancePtr->Control_BaseAddress, XDENSE_INT8_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XDense_int8_EnableAutoRestart(XDense_int8 *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDense_int8_WriteReg(InstancePtr->Control_BaseAddress, XDENSE_INT8_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XDense_int8_DisableAutoRestart(XDense_int8 *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDense_int8_WriteReg(InstancePtr->Control_BaseAddress, XDENSE_INT8_CONTROL_ADDR_AP_CTRL, 0);
}

void XDense_int8_Set_input_image(XDense_int8 *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDense_int8_WriteReg(InstancePtr->Control_BaseAddress, XDENSE_INT8_CONTROL_ADDR_INPUT_IMAGE_DATA, (u32)(Data));
    XDense_int8_WriteReg(InstancePtr->Control_BaseAddress, XDENSE_INT8_CONTROL_ADDR_INPUT_IMAGE_DATA + 4, (u32)(Data >> 32));
}

u64 XDense_int8_Get_input_image(XDense_int8 *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDense_int8_ReadReg(InstancePtr->Control_BaseAddress, XDENSE_INT8_CONTROL_ADDR_INPUT_IMAGE_DATA);
    Data += (u64)XDense_int8_ReadReg(InstancePtr->Control_BaseAddress, XDENSE_INT8_CONTROL_ADDR_INPUT_IMAGE_DATA + 4) << 32;
    return Data;
}

void XDense_int8_Set_output_scores(XDense_int8 *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDense_int8_WriteReg(InstancePtr->Control_BaseAddress, XDENSE_INT8_CONTROL_ADDR_OUTPUT_SCORES_DATA, (u32)(Data));
    XDense_int8_WriteReg(InstancePtr->Control_BaseAddress, XDENSE_INT8_CONTROL_ADDR_OUTPUT_SCORES_DATA + 4, (u32)(Data >> 32));
}

u64 XDense_int8_Get_output_scores(XDense_int8 *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDense_int8_ReadReg(InstancePtr->Control_BaseAddress, XDENSE_INT8_CONTROL_ADDR_OUTPUT_SCORES_DATA);
    Data += (u64)XDense_int8_ReadReg(InstancePtr->Control_BaseAddress, XDENSE_INT8_CONTROL_ADDR_OUTPUT_SCORES_DATA + 4) << 32;
    return Data;
}

void XDense_int8_InterruptGlobalEnable(XDense_int8 *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDense_int8_WriteReg(InstancePtr->Control_BaseAddress, XDENSE_INT8_CONTROL_ADDR_GIE, 1);
}

void XDense_int8_InterruptGlobalDisable(XDense_int8 *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDense_int8_WriteReg(InstancePtr->Control_BaseAddress, XDENSE_INT8_CONTROL_ADDR_GIE, 0);
}

void XDense_int8_InterruptEnable(XDense_int8 *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XDense_int8_ReadReg(InstancePtr->Control_BaseAddress, XDENSE_INT8_CONTROL_ADDR_IER);
    XDense_int8_WriteReg(InstancePtr->Control_BaseAddress, XDENSE_INT8_CONTROL_ADDR_IER, Register | Mask);
}

void XDense_int8_InterruptDisable(XDense_int8 *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XDense_int8_ReadReg(InstancePtr->Control_BaseAddress, XDENSE_INT8_CONTROL_ADDR_IER);
    XDense_int8_WriteReg(InstancePtr->Control_BaseAddress, XDENSE_INT8_CONTROL_ADDR_IER, Register & (~Mask));
}

void XDense_int8_InterruptClear(XDense_int8 *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDense_int8_WriteReg(InstancePtr->Control_BaseAddress, XDENSE_INT8_CONTROL_ADDR_ISR, Mask);
}

u32 XDense_int8_InterruptGetEnabled(XDense_int8 *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XDense_int8_ReadReg(InstancePtr->Control_BaseAddress, XDENSE_INT8_CONTROL_ADDR_IER);
}

u32 XDense_int8_InterruptGetStatus(XDense_int8 *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XDense_int8_ReadReg(InstancePtr->Control_BaseAddress, XDENSE_INT8_CONTROL_ADDR_ISR);
}

