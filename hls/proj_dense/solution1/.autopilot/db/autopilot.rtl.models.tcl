set SynModuleInfo {
  {SRCNAME dense_int8_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_VITIS_LOOP_38_3 MODELNAME dense_int8_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_VITIS_LOOP_38_3 RTLNAME dense_int8_dense_int8_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_VITIS_LOOP_38_3
    SUBMODULES {
      {MODELNAME dense_int8_fmul_32ns_32ns_32_4_max_dsp_1 RTLNAME dense_int8_fmul_32ns_32ns_32_4_max_dsp_1 BINDTYPE op TYPE fmul IMPL maxdsp LATENCY 3 ALLOW_PRAGMA 1}
      {MODELNAME dense_int8_sitofp_32s_32_6_no_dsp_1 RTLNAME dense_int8_sitofp_32s_32_6_no_dsp_1 BINDTYPE op TYPE sitofp IMPL auto LATENCY 5 ALLOW_PRAGMA 1}
      {MODELNAME dense_int8_mul_8s_8s_16_1_1 RTLNAME dense_int8_mul_8s_8s_16_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME dense_int8_sparsemux_7_2_8_1_1 RTLNAME dense_int8_sparsemux_7_2_8_1_1 BINDTYPE op TYPE sparsemux IMPL onehotencoding_realdef}
      {MODELNAME dense_int8_mac_muladd_8s_8s_16s_16_4_1 RTLNAME dense_int8_mac_muladd_8s_8s_16s_16_4_1 BINDTYPE op TYPE all IMPL dsp_slice LATENCY 3}
      {MODELNAME dense_int8_dense_int8_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_VITIS_LOOP_38_3_conv1_b_Rbkb RTLNAME dense_int8_dense_int8_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_VITIS_LOOP_38_3_conv1_b_Rbkb BINDTYPE storage TYPE rom IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME dense_int8_dense_int8_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_VITIS_LOOP_38_3_conv1_w_Rcud RTLNAME dense_int8_dense_int8_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_VITIS_LOOP_38_3_conv1_w_Rcud BINDTYPE storage TYPE rom IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME dense_int8_flow_control_loop_pipe_sequential_init RTLNAME dense_int8_flow_control_loop_pipe_sequential_init BINDTYPE interface TYPE internal_upc_flow_control INSTNAME dense_int8_flow_control_loop_pipe_sequential_init_U}
    }
  }
  {SRCNAME dense_int8_Pipeline_VITIS_LOOP_71_6_VITIS_LOOP_72_7_VITIS_LOOP_73_8_VITIS_LOOP_7 MODELNAME dense_int8_Pipeline_VITIS_LOOP_71_6_VITIS_LOOP_72_7_VITIS_LOOP_73_8_VITIS_LOOP_7 RTLNAME dense_int8_dense_int8_Pipeline_VITIS_LOOP_71_6_VITIS_LOOP_72_7_VITIS_LOOP_73_8_VITIS_LOOP_7
    SUBMODULES {
      {MODELNAME dense_int8_mac_muladd_8s_8s_32s_32_4_1 RTLNAME dense_int8_mac_muladd_8s_8s_32s_32_4_1 BINDTYPE op TYPE all IMPL dsp_slice LATENCY 3}
      {MODELNAME dense_int8_dense_int8_Pipeline_VITIS_LOOP_71_6_VITIS_LOOP_72_7_VITIS_LOOP_73_8_VITIS_LOOdEe RTLNAME dense_int8_dense_int8_Pipeline_VITIS_LOOP_71_6_VITIS_LOOP_72_7_VITIS_LOOP_73_8_VITIS_LOOdEe BINDTYPE storage TYPE rom IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME dense_int8_dense_int8_Pipeline_VITIS_LOOP_71_6_VITIS_LOOP_72_7_VITIS_LOOP_73_8_VITIS_LOOeOg RTLNAME dense_int8_dense_int8_Pipeline_VITIS_LOOP_71_6_VITIS_LOOP_72_7_VITIS_LOOP_73_8_VITIS_LOOeOg BINDTYPE storage TYPE rom IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME dense_int8_Pipeline_VITIS_LOOP_103_12_VITIS_LOOP_108_13_VITIS_LOOP_109_14_VITIS_ MODELNAME dense_int8_Pipeline_VITIS_LOOP_103_12_VITIS_LOOP_108_13_VITIS_LOOP_109_14_VITIS_s RTLNAME dense_int8_dense_int8_Pipeline_VITIS_LOOP_103_12_VITIS_LOOP_108_13_VITIS_LOOP_109_14_VITIS_s
    SUBMODULES {
      {MODELNAME dense_int8_mac_muladd_14ns_4ns_14ns_17_4_1 RTLNAME dense_int8_mac_muladd_14ns_4ns_14ns_17_4_1 BINDTYPE op TYPE all IMPL dsp_slice LATENCY 3}
      {MODELNAME dense_int8_dense_int8_Pipeline_VITIS_LOOP_103_12_VITIS_LOOP_108_13_VITIS_LOOP_109_14_VITfYi RTLNAME dense_int8_dense_int8_Pipeline_VITIS_LOOP_103_12_VITIS_LOOP_108_13_VITIS_LOOP_109_14_VITfYi BINDTYPE storage TYPE rom IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME dense_int8_dense_int8_Pipeline_VITIS_LOOP_103_12_VITIS_LOOP_108_13_VITIS_LOOP_109_14_VITg8j RTLNAME dense_int8_dense_int8_Pipeline_VITIS_LOOP_103_12_VITIS_LOOP_108_13_VITIS_LOOP_109_14_VITg8j BINDTYPE storage TYPE rom IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME dense_int8 MODELNAME dense_int8 RTLNAME dense_int8 IS_TOP 1
    SUBMODULES {
      {MODELNAME dense_int8_fcmp_32ns_32ns_1_2_no_dsp_1 RTLNAME dense_int8_fcmp_32ns_32ns_1_2_no_dsp_1 BINDTYPE op TYPE fcmp IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME dense_int8_layer1_out_RAM_AUTO_1R1W RTLNAME dense_int8_layer1_out_RAM_AUTO_1R1W BINDTYPE storage TYPE ram IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME dense_int8_layer2_out_RAM_AUTO_1R1W RTLNAME dense_int8_layer2_out_RAM_AUTO_1R1W BINDTYPE storage TYPE ram IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME dense_int8_gmem_m_axi RTLNAME dense_int8_gmem_m_axi BINDTYPE interface TYPE adapter IMPL m_axi}
      {MODELNAME dense_int8_control_s_axi RTLNAME dense_int8_control_s_axi BINDTYPE interface TYPE interface_s_axilite}
    }
  }
}
