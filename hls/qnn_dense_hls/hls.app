<AutoPilot:project xmlns:AutoPilot="com.autoesl.autopilot.project" projectType="C/C++" top="dense_int8" name="qnn_dense_hls" ideType="classic">
    <files>
        <file name="dense_int8.cpp" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="../../tb_dense.cpp" sc="0" tb="1" cflags="-Wno-unknown-pragmas" csimflags="" blackbox="false"/>
    </files>
    <solutions>
        <solution name="sol1" status=""/>
    </solutions>
    <Simulation argv="">
        <SimFlow name="csim" setup="false" optimizeCompile="false" clean="false" ldflags="" mflags=""/>
    </Simulation>
</AutoPilot:project>
