#!/bin/bash
resultDIR="result/"
Seq=$(echo foreman soccer coastguard hall)
#Seq=$(echo hall)
GOP=$(echo 2 4 8)
Qindex=$(seq 1 8)
#Qindex=8
ParallelMode=3
UseCoreNum=12

foremanQP=(0 40 39 38 34 34 32 29 25)
soccerQP=(0 44 43 41 36 36 34 31 25)
coastguardQP=(0 38 37 37 34 33 31 30 26)
hallQP=(0 37 36 36 33 33 31 29 24)
for seqName in $Seq
do
	for gop in $GOP
	do
		for qindex in $Qindex
		do
			if [ "$seqName" == "foreman" ]; then
				#echo ${seqName},QP${foremanQP[${qindex}]},GOP${gop},Qindex${qindex}
				echo ./DvcDec -d decoder.cfg -OriginalSequence=../test_sequences/${seqName}_qcif_15_150.yuv -WZBitstream=../DvcEnc/${seqName}GOP${gop}Q${qindex}.wz -IntraBitstream=../DvcEnc/${seqName}GOP${gop}Q${qindex}.intra -KeyFrameSequence=../H264Enc_JM9.5/264_rec_${seqName}${foremanQP[${qindex}]}.yuv -KeyFrameInfo=../H264Enc_JM9.5/per_frame_stat_${seqName}${foremanQP[${qindex}]}.txt -RDFile=${seqName}G${gop}Q${qindex}_rate_psnr -ParallelMode=$ParallelMode -UseCoreNum=$UseCoreNum
			elif [ "$seqName" == "soccer" ]; then
				#echo ${seqName},QP${soccerQP[${qindex}]},GOP${gop},Qindex${qindex}
				echo ./DvcDec -d decoder.cfg -OriginalSequence=../test_sequences/${seqName}_qcif_15_150.yuv -WZBitstream=../DvcEnc/${seqName}GOP${gop}Q${qindex}.wz -IntraBitstream=../DvcEnc/${seqName}GOP${gop}Q${qindex}.intra -KeyFrameSequence=../H264Enc_JM9.5/264_rec_${seqName}${soccerQP[${qindex}]}.yuv -KeyFrameInfo=../H264Enc_JM9.5/per_frame_stat_${seqName}${soccerQP[${qindex}]}.txt -RDFile=${seqName}G${gop}Q${qindex}_rate_psnr -ParallelMode=$ParallelMode -UseCoreNum=$UseCoreNum
			elif [ "$seqName" == "coastguard" ]; then
				#echo ${seqName},QP${coastguardQP[${qindex}]},GOP${gop},Qindex${qindex}
				echo ./DvcDec -d decoder.cfg -OriginalSequence=../test_sequences/${seqName}_qcif_15_150.yuv -WZBitstream=../DvcEnc/${seqName}GOP${gop}Q${qindex}.wz -IntraBitstream=../DvcEnc/${seqName}GOP${gop}Q${qindex}.intra -KeyFrameSequence=../H264Enc_JM9.5/264_rec_${seqName}${coastguardQP[${qindex}]}.yuv -KeyFrameInfo=../H264Enc_JM9.5/per_frame_stat_${seqName}${coastguardQP[${qindex}]}.txt -RDFile=${seqName}G${gop}Q${qindex}_rate_psnr -ParallelMode=$ParallelMode -UseCoreNum=$UseCoreNum
			elif [ "$seqName" == "hall" ]; then
				#echo ${seqName},QP${hallQP[${qindex}]},GOP${gop},Qindex${qindex}
				echo ./DvcDec -d decoder.cfg -OriginalSequence=../test_sequences/${seqName}_qcif_15_164.yuv -WZBitstream=../DvcEnc/${seqName}GOP${gop}Q${qindex}.wz -IntraBitstream=../DvcEnc/${seqName}GOP${gop}Q${qindex}.intra -KeyFrameSequence=../H264Enc_JM9.5/264_rec_${seqName}${hallQP[${qindex}]}.yuv -KeyFrameInfo=../H264Enc_JM9.5/per_frame_stat_${seqName}${hallQP[${qindex}]}.txt -RDFile=${seqName}G${gop}Q${qindex}_rate_psnr -ParallelMode=$ParallelMode -UseCoreNum=$UseCoreNum
			fi
		done
	done
done
#mv *.txt $resultDIR
