#!/bin/bash

#key frame QP
foremanQP=(0 40 39 38 34 34 32 29 25)
soccerQP=(0 44 43 41 36 36 34 31 25)
coastguardQP=(0 38 37 37 34 33 31 30 26)
hallQP=(0 37 36 36 33 33 31 29 24)

#Intra block QP
foreman_IntraQP=(0 39 38 37 33 33 31 28 24)
soccer_IntraQP=(0 41 40 40 35 35 33 30 24)
coastguard_IntraQP=(0 37 36 36 33 32 30 29 25)
hall_IntraQP=(0 36 35 35 32 32 30 28 23)

for seqName in foreman soccer coastguard hall
do
	for gop in 2 4 8
	do
		for qindex in $(seq 1 8)
		do
			if [ "$seqName" == "hall" ]; then
				./DvcEnc.exe -d encoder.cfg -InputFile=../test_sequences/${seqName}_qcif_15_164.yuv -WZBitstreamFile=${seqName}GOP${gop}Q${qindex}.wz -IntraBitstreamFile=${seqName}GOP${gop}Q${qindex}.intra -QIndex=${qindex} -GOP=${gop} -IntraQP=${hall_IntraQP[${qindex}]} -KeyFrameSequence=../H264Enc_JM9.5/264_rec_${seqName}${hallQP[${qindex}]}.yuv -FramesToBeEncoded=164
			elif [ "$seqName" == "foreman" ]; then
                            ./DvcEnc.exe -d encoder.cfg -InputFile=../test_sequences/${seqName}_qcif_15_150.yuv -WZBitstreamFile=${seqName}GOP${gop}Q${qindex}.wz -IntraBitstreamFile=${seqName}GOP${gop}Q${qindex}.intra -QIndex=${qindex} -GOP=${gop} -IntraQP=${foreman_IntraQP[${qindex}]} -KeyFrameSequence=../H264Enc_JM9.5/264_rec_${seqName}${foremanQP[${qindex}]}.yuv
			elif [ "$seqName" == "soccer" ]; then
                            ./DvcEnc.exe -d encoder.cfg -InputFile=../test_sequences/${seqName}_qcif_15_150.yuv -WZBitstreamFile=${seqName}GOP${gop}Q${qindex}.wz -IntraBitstreamFile=${seqName}GOP${gop}Q${qindex}.intra -QIndex=${qindex} -GOP=${gop} -IntraQP=${soccer_IntraQP[${qindex}]} -KeyFrameSequence=../H264Enc_JM9.5/264_rec_${seqName}${soccerQP[${qindex}]}.yuv
			elif [ "$seqName" == "coastguard" ]; then
				./DvcEnc.exe -d encoder.cfg -InputFile=../test_sequences/${seqName}_qcif_15_150.yuv -WZBitstreamFile=${seqName}GOP${gop}Q${qindex}.wz -IntraBitstreamFile=${seqName}GOP${gop}Q${qindex}.intra -QIndex=${qindex} -GOP=${gop} -IntraQP=${coastguard_IntraQP[${qindex}]} -KeyFrameSequence=../H264Enc_JM9.5/264_rec_${seqName}${coastguardQP[${qindex}]}.yuv
			fi
		done
	done
done
