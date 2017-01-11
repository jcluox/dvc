README:
-------
description:
NTU-DVC-codec developed by NTU/CSIE/CMLAB/DSP/Han-ping Cheng
                           NTU/CSIE/CMLAB/DSP/Tse-Chung Su

contact:
plokm@cmlab.csie.ntu.edu.tw
jsrf@cmlab.csie.ntu.edu.tw

------------------------------------------------------------


Compile:
--------
1. Windows:(必須安裝CUDA toolkit 4.0環境以後才可以編譯並執行程式):
  以下測試環境為 Win7 notebook  VS2008  CUDA Toolkit 4.0
  1)install Notebook Developer Drivers for WinVista and Win7
  2)install CUDA Toolkit （預設安裝路徑:C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.0)
    *安裝完以後CUDA相關的系統變數，以及cuda rules會被設定好

  3)用vs2008打開專案DvcCodec.sln, 直接建置方案
  
  註1：如果是安裝其他版本的CUDA toolkit，會有打不開DvcDec專案的情形，是因為DvcDec專案找不到適當的cuda rule的系統檔案載入
       (他預設安裝路徑在 C:\Program Files\Microsoft Visual Studio x.0\VC\VCProjectDefaults\NvCudaRuntimeApi.v4.0.rules
                          C:\Program Files\Microsoft Visual Studio x.0\VC\VCProjectDefaults\NvCudaDriverApi.v4.0.rules )
       解決方法是將 \DISPAC_20100617\DvcDec 裡面的 NvCudaRuntimeApi.v4.0.rules   NvCudaDriverApi.v4.0.rules  複製到上述路徑即可
  註2：Windows系統預設Nvidia顯卡為compute capability 1.x (非Fermi架構) 
       如果你的顯卡為Fermi，可嘗試以下步驟（本人沒有這種硬體環境，沒有試過）:
       1)在"DvcDec"專案按右鍵->屬性->CUDA Driver API->GPU->GPU Architecture->選"sm_20".
       2)在"DvcDec"專案按右鍵->屬性->CUDA Runtime API->GPU->GPU Architecture(1)->選"sm_20".
       3)在"DvcDec"專案按右鍵->屬性->C/C++->前置處理器->前置處理器定義->最後面多加上一個加"FERMI".

  如果想要在.cu檔案中享有Visual Studio提供的intellisense功能，請依照下列網址所教的步驟做設定
  http://choorucode.wordpress.com/2011/02/01/visual-studio-intellisense-for-cuda/

2. Linux: 
  (測試環境為cml15   cml23)
  cml23 (compute capability 2.x以上的Fermi顯卡) 打"make FERMI=1"  (clean的指令請打 "make FERMI=1 clean")
  cml15 (compute capability 1.x的顯卡) 直接打"make" 
--------------------------------------------------------------------------------------------------


Execute: (binary都放在"release/"資料夾下)
-----------------------------------------
1. H264 Encoder:
1.1 Usage: H264Enc.exe -d encoder_main.cfg

1.2 H264 encoder config說明:
InputFile: 要壓縮的原始sequence檔名 (.yuv檔)
FramesToBeEncoded: 總共要壓縮幾張frame (WZ and Intra)
SourceWidth: QCIF->176, CIF->352
SourceHeight: QCIF->144, CIF->288
ReconFile: 264 reconstruct yuv的檔名
QPISlice: 根據DVC encoder的qindex來設 (可參考release/H264Enc_JM9.5/QP_reference.txt)

1.3 Output:
1) ReconFile
2) per_frame_stat.txt (紀錄264壓縮的結果, decoder會用到)



2. DVC Encoder:
2.1 Usage:
DvcEnc.exe -d <ConfigFileName> [-ParameterName=ParmeterValue]
All parameters within the configuration file can also be set as commandline parameter:
ex: DvcEnc.exe -Qindex=4 -d encoder.cfg

2.2 DVC encoder config說明:
InputFile: 要壓縮的原始sequence檔名 (.yuv檔)
FramesToBeEncoded: 總共要壓縮幾張frame (WZ and Intra)
SequenceSize: sequence的resolution, "QCIF"(176x144)或"CIF"(352x288)
WZBitstreamFile: 輸出的WZ bitstream檔名 (裡面存sequence size, GOP, qindex, accmulated syndrome, binary decision map...etc)
GOP: 大於1的數字即可
QIndex: 1到8 (WZ frame quantization table的index)
SkipBlock: 0或1 (0:沒有skip block mode, 1:有skip block mode)
IntraMode: 0或1 (0:沒有Intra block mode, 1:有Intra block mode)
KeyFrameSequence: H264 encoder output的ReconFile檔名
IntraBitstreamFile: 輸出的Intra bitstream檔名 (裡面存WZ frame中, Intra block的H264 bitstream)
IntraQP: WZ frame裡面Intra block的QP (可參考release/DvcEnc/IntraQP_reference.txt)
DeblockingFilter: 0或1 (0:沒有Deblocking Filter, 1:有Deblocking Filter)

2.3 Output:
1) WZBitstreamFile
2) IntraBitstreamFile



3. DVC Decoder:
3.1 Usage: DvcDec.exe -d <ConfigFileName> [-ParameterName=ParmeterValue]
All parameters within the configuration file can also be set as commandline parameter:
ex: DvcDec.exe -WZBitstream=../DvcEnc/foreman.wz -d decoder.cfg -UseCoreNum=4

3.2 DVC decoder config說明:
OriginalSequence:	原始要壓縮的sequence (decoder要拿它來算PSNR)
WZBitstream:			DVC encoder output的WZ bitstream檔名
IntraBitstream:		DVC encoder output的Intra block bitstream檔名
KeyFrameSequence:	H264 encoder output的ReconFile檔名
KeyFrameInfo:			H264 encoder output的per_frame_stat.txt檔名
RDFile:						紀錄decode結果的檔案名稱
OutputSequences:	decode出來的yuv檔案名稱(不要加上副檔名，因為程式會加上qindex及副檔名在最後)
FrameRate:				sequence的frame rate (decoder要拿它來算Bitrate)
ParallelMode:			0到3 (0:sequential執行, 1:只使用OpenMP平行, 2:只使用CUDA平行, 3:同時使用OpenMP跟CUDA平行)
UseCoreNum:				要使用的CPU數量, 大於0的數字 (如果超過電腦最大的CPU數量會自動改用最大數量)

*以下config可以設定在argv裡面，不能設定在config裡面：
GPUid:						(如果一台電腦裡有多張Nvidia顯卡)指定使用哪一張Nvidia顯卡
startFrame:				從第幾張frame開始解碼(從0開始數)(必為keyframe)
endFrame:					解到第幾張frame(必為keyframe)


3.3 Output:
1) RDFile
2) OutputSequences
-------------------------------------------------------------------------------------------------------------


Test Sequences: (放在"release/test_sequences/"資料夾下)
-------------------------------------------------------
1) foreman_qcif_15_150.yuv (150 frames, 15 Hz)
2) soccer_qcif_15_150.yuv (150 frames, 15 Hz)
3) coastguard_qcif_15_150.yuv (150 frames, 15 Hz)
4) hall_qcif_15_164.yuv (164 frames, 15 Hz)
