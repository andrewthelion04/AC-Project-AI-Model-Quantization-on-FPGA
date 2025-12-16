set moduleName dense_int8_Pipeline_VITIS_LOOP_71_6_VITIS_LOOP_72_7_VITIS_LOOP_73_8_VITIS_LOOP_7
set isTopModule 0
set isCombinational 0
set isDatapathOnly 0
set isPipelined 1
set isPipelined_legacy 1
set pipeline_type loop_auto_rewind
set FunctionProtocol ap_ctrl_hs
set restart_counter_num 0
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set hasInterrupt 0
set DLRegFirstOffset 0
set DLRegItemOffset 0
set svuvm_can_support 1
set cdfgNum 6
set C_modelName {dense_int8_Pipeline_VITIS_LOOP_71_6_VITIS_LOOP_72_7_VITIS_LOOP_73_8_VITIS_LOOP_7}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
dict set ap_memory_interface_dict layer2_out { MEM_WIDTH 8 MEM_SIZE 12544 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict layer1_out { MEM_WIDTH 8 MEM_SIZE 6272 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
set C_modelArgList {
	{ layer2_out int 8 regular {array 12544 { 0 3 } 0 1 } {global 1}  }
	{ layer1_out int 8 regular {array 6272 { 1 3 } 1 1 } {global 0}  }
}
set hasAXIMCache 0
set l_AXIML2Cache [list]
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "layer2_out", "interface" : "memory", "bitwidth" : 8, "direction" : "WRITEONLY", "extern" : 0} , 
 	{ "Name" : "layer1_out", "interface" : "memory", "bitwidth" : 8, "direction" : "READONLY", "extern" : 0} ]}
# RTL Port declarations: 
set portNum 25
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ layer2_out_address0 sc_out sc_lv 14 signal 0 } 
	{ layer2_out_ce0 sc_out sc_logic 1 signal 0 } 
	{ layer2_out_we0 sc_out sc_logic 1 signal 0 } 
	{ layer2_out_d0 sc_out sc_lv 8 signal 0 } 
	{ layer1_out_address0 sc_out sc_lv 13 signal 1 } 
	{ layer1_out_ce0 sc_out sc_logic 1 signal 1 } 
	{ layer1_out_q0 sc_in sc_lv 8 signal 1 } 
	{ grp_fu_179_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_179_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_179_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_179_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_187_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_187_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_187_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_183_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_183_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_183_p_opcode sc_out sc_lv 5 signal -1 } 
	{ grp_fu_183_p_dout0 sc_in sc_lv 1 signal -1 } 
	{ grp_fu_183_p_ce sc_out sc_logic 1 signal -1 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "layer2_out_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":14, "type": "signal", "bundle":{"name": "layer2_out", "role": "address0" }} , 
 	{ "name": "layer2_out_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "layer2_out", "role": "ce0" }} , 
 	{ "name": "layer2_out_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "layer2_out", "role": "we0" }} , 
 	{ "name": "layer2_out_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "layer2_out", "role": "d0" }} , 
 	{ "name": "layer1_out_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":13, "type": "signal", "bundle":{"name": "layer1_out", "role": "address0" }} , 
 	{ "name": "layer1_out_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "layer1_out", "role": "ce0" }} , 
 	{ "name": "layer1_out_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "layer1_out", "role": "q0" }} , 
 	{ "name": "grp_fu_179_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_179_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_179_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_179_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_179_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_179_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_179_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_179_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_187_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_187_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_187_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_187_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_187_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_187_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_183_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_183_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_183_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_183_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_183_p_opcode", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "grp_fu_183_p_opcode", "role": "default" }} , 
 	{ "name": "grp_fu_183_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_183_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_183_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_183_p_ce", "role": "default" }}  ]}

set ArgLastReadFirstWriteLatency {
	dense_int8_Pipeline_VITIS_LOOP_71_6_VITIS_LOOP_72_7_VITIS_LOOP_73_8_VITIS_LOOP_7 {
		layer2_out {Type O LastRead -1 FirstWrite 22}
		layer1_out {Type I LastRead 5 FirstWrite -1}
		conv2_w {Type I LastRead -1 FirstWrite -1}
		conv2_b {Type I LastRead -1 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "903191", "Max" : "903191"}
	, {"Name" : "Interval", "Min" : "903169", "Max" : "903169"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	layer2_out { ap_memory {  { layer2_out_address0 mem_address 1 14 }  { layer2_out_ce0 mem_ce 1 1 }  { layer2_out_we0 mem_we 1 1 }  { layer2_out_d0 mem_din 1 8 } } }
	layer1_out { ap_memory {  { layer1_out_address0 mem_address 1 13 }  { layer1_out_ce0 mem_ce 1 1 }  { layer1_out_q0 mem_dout 0 8 } } }
}
