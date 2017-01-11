#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "global.h"
#include "frame.h"
#include "transform.h"
#include "reconstruction.h"
#include "sideInfoCreation.h"
#include "noiseModel.h"
#ifdef CUDA
#include "noiseModel_kernel.h"
#include "updateSI_kernel.h"
#include "motionLearning_kernel.h"
#include "LDPCA_cuda.h"
#endif
#include "bitplane.h"
#include "LdpcaDecoder.h"		//¤p¤p¥Õ
#include "motionLearning.h"
#include "skipBlock.h"
#include "intra.h"
#include "deblock.h"

//decode frames in the same GOP by hierachical order
void decodeGOP(int begin, int end){
	int frameIdx = (begin + end) / 2;
	if(frameIdx > begin){
		//decode WZ frame

		#ifdef SHOW_TIME_INFO
			double overT_time=0.0, ldpca_time=0.0, cnm_time=0.0, condBit_time=0.0, updateSI_time=0.0, ML_time=0.0; 
			printf("decode WZ frame %d...\n", frameIdx);
		#endif

		TimeStamp st_time, end_time;
		timeStart(&st_time);

		decodedFrames[frameIdx].isWZFrame = 1;


		//1. decode intra block
		if(encodedFrames[frameIdx].intraMode){
            intraRecon(decodedFrames[frameIdx].reconsFrame, &(encodedFrames[frameIdx]));
			encodedFrames[frameIdx].intraBlockBits += IntraBS.frame_bitoffset - IntraBS.prev_frame_bitoffset;
            IntraBS.prev_frame_bitoffset = IntraBS.frame_bitoffset;
		}


		//2. side information creation
		TimeStamp side_st, side_end;
		timeStart(&side_st);

		createSideInfo(begin, end, frameIdx, &sideInfo);
		//copy intra block to side info
		copyIntraBlkToSideInfo(sideInfo.sideInfoFrame, decodedFrames[frameIdx].reconsFrame, encodedFrames[frameIdx].BlockModeFlag);
		//save side info to bitmap
		/*unsigned char *sideInfoFrame = (unsigned char*)malloc(sizeof(unsigned char)*FrameSize);
		for(int i=0; i<FrameSize; i++)
			sideInfoFrame[i] = (unsigned char)sideInfo.sideInfoFrame[i];
		char imgName[50];
		sprintf(imgName, "side_info_creation/sideinfo%d.bmp", frameIdx);
		saveToBmp(sideInfoFrame, codingOption.FrameWidth, codingOption.FrameHeight, imgName);
		free(sideInfoFrame);*/
		#ifdef SHOW_TIME_INFO
			double sideCreateTime = timeElapse(side_st,&side_end);				
			Total_sideCreate_time += sideCreateTime;
		#endif
		//getchar();
		//system("pause"); exit(-1);


		//3. convert residue frame to transform domain
		int *sideTransFrame = (int*)malloc(sizeof(int) * FrameSize);
		forwardTransform(sideInfo.residueFrame, sideTransFrame);


		//4. correlation noise parameter estimation
		noiseParameter(sideTransFrame, sideInfo.alpha, encodedFrames[frameIdx].BlockModeFlag);


		//5. ldpca decode and motion learning
		SIR sir;
		sir.searchInfo = (SearchInfo*)malloc(sizeof(SearchInfo) * Block44Num);
		initSIR(&sir, encodedFrames[frameIdx].BlockModeFlag);

		//initial statistic motion field
		SMF *smf = (SMF*)malloc(sizeof(SMF) * sir.searchCandidate);
		initSMF(smf, &sir);

		#ifdef CUDA
		if(codingOption.ParallelMode > 1){
			CNM_CUDA_Init_Variable(&sir, smf);
			ME_CUDA_Init_Variable(&sir, begin, end);
		}
		#endif

		//over-complete transform
		#ifdef SHOW_TIME_INFO
			TimeStamp overT_st, overT_end;
			timeStart(&overT_st);
		#endif
		int searchRangeBottom = codingOption.FrameHeight - 4;
		int searchRangeRight = codingOption.FrameWidth - 4;
		Trans *si_overcomplete_trans = (Trans*)malloc(sizeof(Trans) * (searchRangeBottom+1) * (searchRangeRight+1));
		switch(codingOption.ParallelMode){
			case 0:	//sequential
			case 2: //CUDA only
				overCompleteTransform(sideInfo.sideInfoFrame, si_overcomplete_trans, searchRangeBottom, searchRangeRight);
				break;
			case 1:	//OpenMP only
			case 3: //OpenMP + CUDA
				overCompleteTransform_OpenMP(sideInfo.sideInfoFrame, si_overcomplete_trans, searchRangeBottom, searchRangeRight);
				break;
			default:
				break;
		}
		#ifdef SHOW_TIME_INFO
			overT_time += timeElapse(overT_st,&overT_end);	
		#endif

		//initial motion learning side info
		init_si_ML(sideTransFrame, si_overcomplete_trans);

		//ldpca decode (zig-zag order)
		int SyndromeByteLength = Block44Num >> 3;	//fixed length source for LDPCA
		float *LR_intrinsic = (float*)malloc(sizeof(float) * Block44Num);	//store conditional bit probability
		unsigned char *decodedBitplane = (unsigned char*)malloc(sizeof(unsigned char) * Block44Num);
		unsigned char *deblockFlag = (unsigned char*)malloc(sizeof(unsigned char) * FrameSize);	//for deblocking filter
		//initial bit rate
		decodedFrames[frameIdx].bitRate = encodedFrames[frameIdx].skipBlockBits + encodedFrames[frameIdx].intraBlockBits;
		/*if(encodedFrames[frameIdx].skipMode == 1)
			decodedFrames[frameIdx].bitRate = Block88Num + 1;	//skip mode
		else
			decodedFrames[frameIdx].bitRate = 1;	//non skip mode*/


		//calculate quantization step size and coefficient range
		int stepSize[BLOCKSIZE44];
		int coeffRange[BLOCKSIZE44];
		stepSize[0] = 1024 / Qtable[0]; //DC range: [0, 1024)
		coeffRange[0] = 1024;
		//AC dynamic range
		for(int i=1; i<BLOCKSIZE44; i++){
			if(Qtable[i] > 0){				
				stepSize[i] = (int)ceil((double)( (encodedFrames[frameIdx].ACRange[i]+1)<<1 ) / (double)Qtable[i]); // ceil( 2*|Valmax| / level )
				decodedFrames[frameIdx].bitRate += 10; //use 10 bits to store AC dynamic range
				coeffRange[i] = (ACLevelShift[i] + 1) * stepSize[i] - 1;
			}
			else
				stepSize[i] = 0;
		}

		//copy skip block from reference frame
		if(encodedFrames[frameIdx].skipMode == 1){
			copySkipBlkToRecons(decodedFrames[frameIdx].reconsFrame, decodedFrames[begin].reconsFrame, encodedFrames[frameIdx].BlockModeFlag);
		}
		//decode bitplane in zig-zag order
		memset(sideInfo.reconsTransFrame, 0, sizeof(int) * FrameSize);	//initial for ldpca decode
		int bitplaneIdx = 0;	//bitplane index
		for(int band=0; band<BLOCKSIZE44; band++){
			//for each band
			int index = zigzag[band];	//index in block (zig-zag order)
			int Qbits = QtableBits[index];
			if(Qbits > 0){
				int range = coeffRange[index];

				//correlation noise modeling
				#ifdef SHOW_TIME_INFO
					TimeStamp cnm_st, cnm_end;
					timeStart(&cnm_st);
				#endif				
				float *noiseDist;
				switch(codingOption.ParallelMode){
					case 0:	//sequential
						noiseDist = noiseDistribution(range, index, sideInfo.alpha, smf, si_overcomplete_trans, &sir);
						break;
					case 1:	//OpenMP only
						noiseDist = noiseDistribution_OpenMP(range, index, sideInfo.alpha, smf, si_overcomplete_trans, &sir);
						break;
					#ifdef CUDA
					case 2: //CUDA only
					case 3: //OpenMP + CUDA
						init_noiseDst_CUDA(range, index, sir.searchCandidate);
						noiseDistribution_CUDA(range, index, sideInfo.alpha, smf, si_overcomplete_trans, &sir);		
						break;
					#endif
					default:
						break;
				}
				#ifdef SHOW_TIME_INFO
					cnm_time += timeElapse(cnm_st,&cnm_end);	
				#endif				

				//decode from MSB to LSB
				for(int bit=Qbits; bit>0; bit--){

					//conditional bit probability calculation
					#ifdef SHOW_TIME_INFO
						TimeStamp condBit_st, condBit_end;
						timeStart(&condBit_st);
					#endif					
					if(encodedFrames[frameIdx].skipMode == 1 || encodedFrames[frameIdx].intraMode == 1){
						//initial LR_intrinsic
						for(int i=0; i<Block44Num; i++)
							LR_intrinsic[i] = 0.9999f / 0.0001f;
					}
					switch(codingOption.ParallelMode){
						case 0:	//sequential
							condBitProb(LR_intrinsic, index, bit, stepSize[index], sideInfo.reconsTransFrame, range, noiseDist, &sir);
							break;
						case 1:	//OpenMP only
							condBitProb_OpenMP(LR_intrinsic, index, bit, stepSize[index], sideInfo.reconsTransFrame, range, noiseDist, &sir);
							break;
						#ifdef CUDA
						case 2: //CUDA only
						case 3: //OpenMP + CUDA
							condBitProb_CUDA(LR_intrinsic, index, bit, stepSize[index], sideInfo.reconsTransFrame, range, &sir);
							break;
						#endif
						default:
							break;
					}
					#ifdef SHOW_TIME_INFO
						condBit_time += timeElapse(condBit_st,&condBit_end);	
					#endif

					//ldpca decode
					#ifdef SHOW_TIME_INFO
						TimeStamp ldpca_st, ldpca_end;
						timeStart(&ldpca_st);
					#endif	
					int rate = 0;
					float entropy;
					//conditional entropy
					switch(codingOption.ParallelMode){
						case 0:	//sequential
							entropy = ComputeCE(LR_intrinsic);
							break;
						case 1:	//OpenMP only
							entropy = ComputeCE_OpenMP(LR_intrinsic);
							break;
						#ifdef CUDA
						case 2: //CUDA only
							#ifdef FERMI
								entropy = ComputeCE_FERMI();
							#else
								entropy = ComputeCE(LR_intrinsic);
							#endif							
							break;
						case 3: //OpenMP + CUDA
							#ifdef FERMI
								entropy = ComputeCE_FERMI();
							#else
								entropy = ComputeCE_OpenMP(LR_intrinsic);
							#endif
							break;
						#endif
						default:
							break;
					}					
					//printf("band:%d, bits: %d, entropy: %f, bits: %f\n", band, bit, entropy, entropy*(float)Block44Num);

					//trans accumulatedSyndrome Byte to bits
					unsigned char *accumulatedSyndrome = ByteToBit(encodedFrames[frameIdx].AccumulatedSyndrome[bitplaneIdx], SyndromeByteLength, Block44Num);
			
					LdpcaDecoder(LR_intrinsic, accumulatedSyndrome, decodedBitplane, &rate, encodedFrames[frameIdx].CRC[bitplaneIdx], entropy);
					//int request = LdpcaDecoder(LR_intrinsic, accumulatedSyndrome, decodedBitplane, &rate, encodedFrames[frameIdx].CRC[bitplaneIdx], entropy);
					decodedFrames[frameIdx].bitRate += rate;
					free(accumulatedSyndrome);

					//printf("final rate: %d\n", rate);
					
					//put decoded bitplane to coefficient band
					putBitplaneToCoeff(decodedBitplane, sideInfo.reconsTransFrame, index, encodedFrames[frameIdx].BlockModeFlag);
					bitplaneIdx++;
					#ifdef SHOW_TIME_INFO
						ldpca_time += timeElapse(ldpca_st,&ldpca_end);	
					#endif
				}

				//reconstruction
				//reconstruction(sideInfo.reconsTransFrame, sideTransFrame, band, sideInfo.alpha, stepSize, encodedFrames[frameIdx].skipBlockFlag);
				switch(codingOption.ParallelMode){
					case 0:	//sequential
					case 2: //CUDA only
						reconstruction(sideInfo.reconsTransFrame, sideTransFrame, index, sideInfo.alpha, stepSize, encodedFrames[frameIdx].BlockModeFlag, deblockFlag);
						break;
					case 1:	//OpenMP only
					case 3: //OpenMP + CUDA
						reconstruction_OpenMP(sideInfo.reconsTransFrame, sideTransFrame, index, sideInfo.alpha, stepSize, encodedFrames[frameIdx].BlockModeFlag, deblockFlag);
						break;
					default:
						break;
				}	

				//inverse transform				
				switch(codingOption.ParallelMode){
					case 0:	//sequential
					case 2: //CUDA only
						inverseTransform(sideTransFrame, decodedFrames[frameIdx].reconsFrame, &sir);
						break;
					case 1:	//OpenMP only
					case 3: //OpenMP + CUDA
						inverseTransform_OpenMP(sideTransFrame, decodedFrames[frameIdx].reconsFrame, &sir);
						break;
					default:
						break;
				}	

				//side info re-estimation
				#ifdef SHOW_TIME_INFO
					TimeStamp updateSI_st, updateSI_end;
					timeStart(&updateSI_st);
				#endif
				switch(codingOption.ParallelMode){
					case 0:	//sequential
						updateSI(&sideInfo, decodedFrames[frameIdx].reconsFrame, frameIdx, begin, end, &sir);
						free(noiseDist);
						break;
					case 1:	//OpenMP only
						updateSI_OpenMP(&sideInfo, decodedFrames[frameIdx].reconsFrame, frameIdx, begin, end, &sir);
						free(noiseDist);
						break;
					#ifdef CUDA
					case 2: //CUDA only
					case 3:	//OpenMP + CUDA
						updateSI_CUDA(&sideInfo, decodedFrames[frameIdx].reconsFrame, frameIdx, begin, end, &sir);
						free_noiseDst_CUDA();
						break;
					#endif
					default:
						break;
				}
				#ifdef SHOW_TIME_INFO
					updateSI_time += timeElapse(updateSI_st,&updateSI_end);	
				#endif

				//over complete transform
				#ifdef SHOW_TIME_INFO					
					timeStart(&overT_st);
				#endif
				switch(codingOption.ParallelMode){
					case 0:	//sequential
					case 2: //CUDA only
						overCompleteTransform(sideInfo.sideInfoFrame, si_overcomplete_trans, searchRangeBottom, searchRangeRight);
						break;
					case 1:	//OpenMP only
					case 3: //OpenMP + CUDA
						overCompleteTransform_OpenMP(sideInfo.sideInfoFrame, si_overcomplete_trans, searchRangeBottom, searchRangeRight);
						break;
					default:
						break;
				}				
				#ifdef SHOW_TIME_INFO
					overT_time += timeElapse(overT_st,&overT_end);	
				#endif

				//motion learning
				#ifdef SHOW_TIME_INFO
					TimeStamp ML_st, ML_end;
					timeStart(&ML_st);
				#endif
				switch(codingOption.ParallelMode){
				case 0:	//sequential
					//update statistic motion fields
					updateSMF(smf, decodedFrames[frameIdx].reconsFrame, sideInfo.sideInfoFrame, &sir);
					//motion learning side info re-estimation
					update_si_ML(sideTransFrame, si_overcomplete_trans, band+1, smf, &sir);
					break;
				case 1:	//OpenMP only
					//update statistic motion fields
					updateSMF_OpenMP(smf, decodedFrames[frameIdx].reconsFrame, sideInfo.sideInfoFrame, &sir);
					//motion learning side info re-estimation
					update_si_ML_OpenMP(sideTransFrame, si_overcomplete_trans, band+1, smf, &sir);
					break;
				#ifdef CUDA
				case 2: //CUDA only
					//update statistic motion fields
					updateSMF_CUDA(smf, decodedFrames[frameIdx].reconsFrame, sideInfo.sideInfoFrame, &sir);
					//motion learning side info re-estimation
					update_si_ML(sideTransFrame, si_overcomplete_trans, band+1, smf, &sir);
					break;
				case 3: //OpenMP + CUDA
					//update statistic motion fields
					//TimeStamp start_t, end_t;
					//timeStart(&start_t);
					updateSMF_CUDA(smf, decodedFrames[frameIdx].reconsFrame, sideInfo.sideInfoFrame, &sir);
					//printf("%lfms\n", timeElapse(start_t,&end_t));
					//timeStart(&start_t);
					//motion learning side info re-estimation
					update_si_ML_OpenMP(sideTransFrame, si_overcomplete_trans, band+1, smf, &sir);
					//printf("%lfms\n", timeElapse(start_t,&end_t));getchar();
					break;
				#endif
				default:
					break;
				}
				#ifdef SHOW_TIME_INFO
					ML_time += timeElapse(ML_st,&ML_end);	
				#endif
			}
			else{
				//reconstruction
				//reconstruction(sideInfo.reconsTransFrame, sideTransFrame, band-1, sideInfo.alpha, stepSize, encodedFrames[frameIdx].skipBlockFlag);
				break;
			}
		}
		

		//6. inverse transform
		switch(codingOption.ParallelMode){
			case 0:	//sequential
			case 2: //CUDA only
				inverseTransform(sideTransFrame, decodedFrames[frameIdx].reconsFrame, &sir);
				break;
			case 1:	//OpenMP only
			case 3: //OpenMP + CUDA
				inverseTransform_OpenMP(sideTransFrame, decodedFrames[frameIdx].reconsFrame, &sir);
				break;
			default:
				break;
		}	
		//save reconstruct frame to bitmap
		/*sprintf(imgName, "reconstruct_frame/reconstuct%d.bmp", frameIdx);
		saveToBmp(decodedFrames[frameIdx].reconsFrame, codingOption.FrameWidth, codingOption.FrameHeight, imgName);*/


		//7. deblocking filter
		if(codingOption.DeblockingMode){
            unsigned char *deblockframe = (unsigned char*)malloc(sizeof(unsigned char) * FrameSize);
            deblock(deblockframe,decodedFrames[frameIdx].reconsFrame, deblockFlag, encodedFrames[frameIdx].BlockModeFlag, begin);
            memcpy(decodedFrames[frameIdx].reconsFrame, deblockframe, sizeof(unsigned char) * FrameSize);
            free(deblockframe);
        }
		free(deblockFlag);

		//release resource
		free(sideTransFrame);
		free(LR_intrinsic);
		free(decodedBitplane);
		free(smf);
		free(sir.searchInfo);
		free(si_overcomplete_trans);

		//8. print Rate & PSNR & time
		#ifdef SHOW_TIME_INFO			
			printf("side info creation time: %.2lfms\
				  \novercomplete transform: %.2lfms\
				  \ncorrelation noise modeling: %.2lfms\
				  \ncond bit prob compute: %.2lfms\
				  \nldpca decode: %.2lfms\
				  \nupdate side info: %.2lfms\
				  \nmotion learning: %.2lfms\n", sideCreateTime, overT_time, cnm_time, condBit_time, ldpca_time, updateSI_time, ML_time);
			Total_overT_time += overT_time;
			Total_CNM_time += cnm_time;
			Total_ldpca_time += ldpca_time;
			Total_condBit_time += condBit_time;
			Total_updateSI_time += updateSI_time;
			Total_ML_time += ML_time;
		#endif
		double decodeTime = timeElapse(st_time,&end_time);	

		decodedFrames[frameIdx].decodeTime = decodeTime;
		decodedFrames[frameIdx].PSNR = psnr(decodedFrames[frameIdx].reconsFrame, oriFrames[frameIdx]);
		#ifdef SHOW_TIME_INFO
			printf("FrameNo: %d, Y_bits: %d bits, PSNR: %.2lf, Decoding time: %.2lf ms \n\n", frameIdx, decodedFrames[frameIdx].bitRate, decodedFrames[frameIdx].PSNR, decodeTime);
		#endif
		//system("pause");
		
		decodeGOP(begin, frameIdx);
		decodeGOP(frameIdx, end);
	}
}

//skip frames(intra block bitstream) in the same GOP by hierachical order
void skipGOP(int begin, int end){
	int frameIdx = (begin + end) / 2;
	if(frameIdx > begin){
		//decode WZ frame

		#ifdef SHOW_TIME_INFO
			double skip_time=0.0; 
			printf("skip WZ frame %d...\n", frameIdx);
		#endif

		#ifndef _WIN32
		struct timeval st_time, end_time;
		gettimeofday(&st_time, NULL);
		#else
		LARGE_INTEGER st_time={0}, end_time={0};
		QueryPerformanceCounter(&st_time);
		#endif

		decodedFrames[frameIdx].isWZFrame = 1;


		//1. skip intra block
		if(encodedFrames[frameIdx].intraMode){
            intraRecon(decodedFrames[frameIdx].reconsFrame, &(encodedFrames[frameIdx]));
			encodedFrames[frameIdx].intraBlockBits += IntraBS.frame_bitoffset - IntraBS.prev_frame_bitoffset;
            IntraBS.prev_frame_bitoffset = IntraBS.frame_bitoffset;
		}
		#ifdef SHOW_TIME_INFO
			#ifndef _WIN32			
			gettimeofday(&end_time, NULL);
				skip_time = (double)(end_time.tv_sec - st_time.tv_sec) + (double)(end_time.tv_usec - st_time.tv_usec)/1000000.0;  //second
			#else
			QueryPerformanceCounter(&end_time);
				skip_time = (double)(end_time.QuadPart - st_time.QuadPart)/CPUFreq;  //second
			#endif
		#endif
	
		skipGOP(begin, frameIdx);
		skipGOP(frameIdx, end);
	}
}