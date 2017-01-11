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
1. Windows:(�����w��CUDA toolkit 4.0���ҥH��~�i�H�sĶ�ð���{��):
  �H�U�������Ҭ� Win7 notebook  VS2008  CUDA Toolkit 4.0
  1)install Notebook Developer Drivers for WinVista and Win7
  2)install CUDA Toolkit �]�w�]�w�˸��|:C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.0)
    *�w�˧��H��CUDA�������t���ܼơA�H��cuda rules�|�Q�]�w�n

  3)��vs2008���}�M��DvcCodec.sln, �����ظm���
  
  ��1�G�p�G�O�w�˨�L������CUDA toolkit�A�|�������}DvcDec�M�ת����ΡA�O�]��DvcDec�M�ק䤣��A��cuda rule���t���ɮ׸��J
       (�L�w�]�w�˸��|�b C:\Program Files\Microsoft Visual Studio x.0\VC\VCProjectDefaults\NvCudaRuntimeApi.v4.0.rules
                          C:\Program Files\Microsoft Visual Studio x.0\VC\VCProjectDefaults\NvCudaDriverApi.v4.0.rules )
       �ѨM��k�O�N \DISPAC_20100617\DvcDec �̭��� NvCudaRuntimeApi.v4.0.rules   NvCudaDriverApi.v4.0.rules  �ƻs��W�z���|�Y�i
  ��2�GWindows�t�ιw�]Nvidia��d��compute capability 1.x (�DFermi�[�c) 
       �p�G�A����d��Fermi�A�i���եH�U�B�J�]���H�S���o�صw�����ҡA�S���չL�^:
       1)�b"DvcDec"�M�׫��k��->�ݩ�->CUDA Driver API->GPU->GPU Architecture->��"sm_20".
       2)�b"DvcDec"�M�׫��k��->�ݩ�->CUDA Runtime API->GPU->GPU Architecture(1)->��"sm_20".
       3)�b"DvcDec"�M�׫��k��->�ݩ�->C/C++->�e�m�B�z��->�e�m�B�z���w�q->�̫᭱�h�[�W�@�ӥ["FERMI".

  �p�G�Q�n�b.cu�ɮפ��ɦ�Visual Studio���Ѫ�intellisense�\��A�Ш̷ӤU�C���}�ұЪ��B�J���]�w
  http://choorucode.wordpress.com/2011/02/01/visual-studio-intellisense-for-cuda/

2. Linux: 
  (�������Ҭ�cml15   cml23)
  cml23 (compute capability 2.x�H�W��Fermi��d) ��"make FERMI=1"  (clean�����O�Х� "make FERMI=1 clean")
  cml15 (compute capability 1.x����d) ������"make" 
--------------------------------------------------------------------------------------------------


Execute: (binary����b"release/"��Ƨ��U)
-----------------------------------------
1. H264 Encoder:
1.1 Usage: H264Enc.exe -d encoder_main.cfg

1.2 H264 encoder config����:
InputFile: �n���Y����lsequence�ɦW (.yuv��)
FramesToBeEncoded: �`�@�n���Y�X�iframe (WZ and Intra)
SourceWidth: QCIF->176, CIF->352
SourceHeight: QCIF->144, CIF->288
ReconFile: 264 reconstruct yuv���ɦW
QPISlice: �ھ�DVC encoder��qindex�ӳ] (�i�Ѧ�release/H264Enc_JM9.5/QP_reference.txt)

1.3 Output:
1) ReconFile
2) per_frame_stat.txt (����264���Y�����G, decoder�|�Ψ�)



2. DVC Encoder:
2.1 Usage:
DvcEnc.exe -d <ConfigFileName> [-ParameterName=ParmeterValue]
All parameters within the configuration file can also be set as commandline parameter:
ex: DvcEnc.exe -Qindex=4 -d encoder.cfg

2.2 DVC encoder config����:
InputFile: �n���Y����lsequence�ɦW (.yuv��)
FramesToBeEncoded: �`�@�n���Y�X�iframe (WZ and Intra)
SequenceSize: sequence��resolution, "QCIF"(176x144)��"CIF"(352x288)
WZBitstreamFile: ��X��WZ bitstream�ɦW (�̭��ssequence size, GOP, qindex, accmulated syndrome, binary decision map...etc)
GOP: �j��1���Ʀr�Y�i
QIndex: 1��8 (WZ frame quantization table��index)
SkipBlock: 0��1 (0:�S��skip block mode, 1:��skip block mode)
IntraMode: 0��1 (0:�S��Intra block mode, 1:��Intra block mode)
KeyFrameSequence: H264 encoder output��ReconFile�ɦW
IntraBitstreamFile: ��X��Intra bitstream�ɦW (�̭��sWZ frame��, Intra block��H264 bitstream)
IntraQP: WZ frame�̭�Intra block��QP (�i�Ѧ�release/DvcEnc/IntraQP_reference.txt)
DeblockingFilter: 0��1 (0:�S��Deblocking Filter, 1:��Deblocking Filter)

2.3 Output:
1) WZBitstreamFile
2) IntraBitstreamFile



3. DVC Decoder:
3.1 Usage: DvcDec.exe -d <ConfigFileName> [-ParameterName=ParmeterValue]
All parameters within the configuration file can also be set as commandline parameter:
ex: DvcDec.exe -WZBitstream=../DvcEnc/foreman.wz -d decoder.cfg -UseCoreNum=4

3.2 DVC decoder config����:
OriginalSequence:	��l�n���Y��sequence (decoder�n�����Ӻ�PSNR)
WZBitstream:			DVC encoder output��WZ bitstream�ɦW
IntraBitstream:		DVC encoder output��Intra block bitstream�ɦW
KeyFrameSequence:	H264 encoder output��ReconFile�ɦW
KeyFrameInfo:			H264 encoder output��per_frame_stat.txt�ɦW
RDFile:						����decode���G���ɮצW��
OutputSequences:	decode�X�Ӫ�yuv�ɮצW��(���n�[�W���ɦW�A�]���{���|�[�Wqindex�ΰ��ɦW�b�̫�)
FrameRate:				sequence��frame rate (decoder�n�����Ӻ�Bitrate)
ParallelMode:			0��3 (0:sequential����, 1:�u�ϥ�OpenMP����, 2:�u�ϥ�CUDA����, 3:�P�ɨϥ�OpenMP��CUDA����)
UseCoreNum:				�n�ϥΪ�CPU�ƶq, �j��0���Ʀr (�p�G�W�L�q���̤j��CPU�ƶq�|�۰ʧ�γ̤j�ƶq)

*�H�Uconfig�i�H�]�w�bargv�̭��A����]�w�bconfig�̭��G
GPUid:						(�p�G�@�x�q���̦��h�iNvidia��d)���w�ϥέ��@�iNvidia��d
startFrame:				�q�ĴX�iframe�}�l�ѽX(�q0�}�l��)(����keyframe)
endFrame:					�Ѩ�ĴX�iframe(����keyframe)


3.3 Output:
1) RDFile
2) OutputSequences
-------------------------------------------------------------------------------------------------------------


Test Sequences: (��b"release/test_sequences/"��Ƨ��U)
-------------------------------------------------------
1) foreman_qcif_15_150.yuv (150 frames, 15 Hz)
2) soccer_qcif_15_150.yuv (150 frames, 15 Hz)
3) coastguard_qcif_15_150.yuv (150 frames, 15 Hz)
4) hall_qcif_15_164.yuv (164 frames, 15 Hz)
