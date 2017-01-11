#include <stdlib.h>
#include <math.h>
#include "sideInfoCreation.h"
#include "filter.h"
#ifdef CUDA
#include "forwardME_kernel.h"
#endif


double costFunc(double mad, Point *mv){
	return mad * ( 1.0 + K * sqrt(mv->x * mv->x + mv->y * mv->y) );
}

int computeEndPoint(double stPoint, double vector, double scale, int lower, int upper){
	int endPoint = (int)((stPoint + vector * scale) + 0.5); //round to nearest integer
	if(endPoint < lower)
		return lower;
	else if(endPoint > upper)
		return upper;
	else
		return endPoint;
}

void init_1616ME(SearchInfo *searchInfo1616){
	int searchRangeBottom = codingOption.FrameHeight - 16;
	int searchRangeRight = codingOption.FrameWidth - 16;
	int blk1616Idx = 0;
	for(int i=0; i<Block1616Height; i++){
		int top = i<<4;
		int searchTop = top - SEARCHRANGE;
		int searchBottom = top + SEARCHRANGE;
		if(searchTop < 0)
			searchTop = 0;
		if(searchBottom > searchRangeBottom)
			searchBottom = searchRangeBottom;
		for(int j=0; j<Block1616Width; j++){
			//for each 16x16 block
			int left = j<<4;

			int searchLeft = left - SEARCHRANGE;
			int searchRight = left + SEARCHRANGE;
			if(searchLeft < 0)
				searchLeft = 0;
			if(searchRight > searchRangeRight)
				searchRight = searchRangeRight;

			SearchInfo searchInfo;
			searchInfo.searchTop = searchTop;
			searchInfo.searchBottom = searchBottom;
			searchInfo.searchLeft = searchLeft;
			searchInfo.searchRight = searchRight;
			searchInfo.top = top;
			searchInfo.left = left;
			searchInfo.searchWidth = searchRight - searchLeft + 1;
			searchInfo.searchRange = searchInfo.searchWidth * (searchBottom - searchTop + 1);
			searchInfo1616[blk1616Idx++] = searchInfo;
		}
	}
}

void forwardME(int future, int past, int current, SideInfo *si){
	int searchRangeBottom = codingOption.FrameHeight - 16;
	int searchRangeRight = codingOption.FrameWidth - 16;
	unsigned char *futureFrame = decodedFrames[future].lowPassFrame;
	unsigned char *pastFrame = decodedFrames[past].lowPassFrame;

	double intervFC = (double)(future-current) / (double)(future-past);
	double intervPC = (double)(current-past) / (double)(future-past);

	MV *mcroBlockMV = (MV*)malloc(sizeof(MV) * Block1616Num);

	for(int i=0; i<Block1616Num; i++){
		SearchInfo searchInfo = SearchInfo1616[i];
		int top = searchInfo.top, left = searchInfo.left;

		int fBlock[BLOCKSIZE1616];
		int blkIdx = 0;
		int fIdx = top * codingOption.FrameWidth + left;
		for(int m=0; m<16; m++){
			for(int n=0; n<16; n++){
				fBlock[blkIdx++] = futureFrame[fIdx + n];
			}
			fIdx += codingOption.FrameWidth;
		}

		//motion estimation
		double minCost = 10000000.0;
		MV bestMV;
		for(int py=searchInfo.searchTop; py<=searchInfo.searchBottom; py++){
			for(int px=searchInfo.searchLeft; px<=searchInfo.searchRight; px++){
				int sad = 0;	//mean abosolute difference
				blkIdx = 0;
				int pIdx = py * codingOption.FrameWidth + px;
				for(int m=0; m<16; m++){
					for(int n=0; n<16; n++){
						sad += abs(fBlock[blkIdx++] - pastFrame[pIdx + n]);
					}
					pIdx += codingOption.FrameWidth;
				}
				Point vector;
				vector.x = (double)(px - left);
				vector.y = (double)(py - top);
				double cost = costFunc((double)sad, &vector);
				if(cost < minCost){
					minCost = cost;
					//bestMV.Future.x = (double)left; bestMV.Future.y = (double)top;
					bestMV.Past.x = (double)px; bestMV.Past.y = (double)py;
					bestMV.vector = vector;
				}

			}
		}

		mcroBlockMV[i] = bestMV;	//motion vector of future frame
		//calculate holes on current frame witch are passed by motion vector of future frame
		si->Holes[i].x = (double)left * intervPC + (double)bestMV.Past.x * intervFC;
		si->Holes[i].y = (double)top * intervPC + (double)bestMV.Past.y * intervFC;
	}

	//find motion vector of each macroblock for WZ frame
	double maxDst = codingOption.FrameWidth * codingOption.FrameWidth + codingOption.FrameHeight * codingOption.FrameHeight;
	for(int i=0; i<Block1616Num; i++){
		int top = SearchInfo1616[i].top, left = SearchInfo1616[i].left;

		double minDst = maxDst;
		MV bestMV;
		for(int j=0; j<Block1616Num; j++){				
			double dst = (left - si->Holes[j].x)*(left - si->Holes[j].x) + (top - si->Holes[j].y)*(top - si->Holes[j].y);
			if(dst < minDst){
				minDst = dst;
				bestMV = mcroBlockMV[j];
			}
		}

		MV *mbMV = &(si->mcroBlockMV[i]);
		mbMV->Future.x = (double)computeEndPoint(left, -bestMV.vector.x, intervFC, 0, searchRangeRight);
		mbMV->Future.y = (double)computeEndPoint(top, -bestMV.vector.y, intervFC, 0, searchRangeBottom);
		mbMV->Past.x = (double)computeEndPoint(left, bestMV.vector.x, intervPC, 0, searchRangeRight);
		mbMV->Past.y = (double)computeEndPoint(top, bestMV.vector.y, intervPC, 0, searchRangeBottom);
		mbMV->vector.x = mbMV->Past.x - mbMV->Future.x;
		mbMV->vector.y = mbMV->Past.y - mbMV->Future.y;		
	}

	free(mcroBlockMV);
}

void forwardME_OpenMP(int future, int past, int current, SideInfo *si){
	int searchRangeBottom = codingOption.FrameHeight - 16;
	int searchRangeRight = codingOption.FrameWidth - 16;
	unsigned char *futureFrame = decodedFrames[future].lowPassFrame;
	unsigned char *pastFrame = decodedFrames[past].lowPassFrame;

	double intervFC = (double)(future-current) / (double)(future-past);
	double intervPC = (double)(current-past) / (double)(future-past);

	MV *mcroBlockMV = (MV*)malloc(sizeof(MV) * Block1616Num);
#pragma omp parallel for
	for(int i=0; i<Block1616Num; i++){
		SearchInfo searchInfo = SearchInfo1616[i];
		int top = searchInfo.top, left = searchInfo.left;

		int fBlock[BLOCKSIZE1616];
		int blkIdx = 0;
		int fIdx = top * codingOption.FrameWidth + left;
		for(int m=0; m<16; m++){
			for(int n=0; n<16; n++){
				fBlock[blkIdx++] = futureFrame[fIdx + n];
			}
			fIdx += codingOption.FrameWidth;
		}

		//motion estimation
		double minCost = 10000000.0;
		MV bestMV;
		for(int py=searchInfo.searchTop; py<=searchInfo.searchBottom; py++){
			for(int px=searchInfo.searchLeft; px<=searchInfo.searchRight; px++){
				int sad = 0;	//mean abosolute difference
				blkIdx = 0;
				int pIdx = py * codingOption.FrameWidth + px;
				for(int m=0; m<16; m++){
					for(int n=0; n<16; n++){
						sad += abs(fBlock[blkIdx++] - pastFrame[pIdx + n]);
					}
					pIdx += codingOption.FrameWidth;
				}
				Point vector;
				vector.x = (double)(px - left);
				vector.y = (double)(py - top);
				double cost = costFunc((double)sad, &vector);
				if(cost < minCost){
					minCost = cost;
					//bestMV.Future.x = (double)left; bestMV.Future.y = (double)top;
					bestMV.Past.x = (double)px; bestMV.Past.y = (double)py;
					bestMV.vector = vector;
				}

			}
		}

		mcroBlockMV[i] = bestMV;	//motion vector of future frame
		//calculate holes on current frame witch are passed by motion vector of future frame
		si->Holes[i].x = (double)left * intervPC + (double)bestMV.Past.x * intervFC;
		si->Holes[i].y = (double)top * intervPC + (double)bestMV.Past.y * intervFC;
	}

	//find motion vector of each macroblock for WZ frame
	double maxDst = codingOption.FrameWidth * codingOption.FrameWidth + codingOption.FrameHeight * codingOption.FrameHeight;
#pragma omp parallel for
	for(int i=0; i<Block1616Num; i++){
		int top = SearchInfo1616[i].top, left = SearchInfo1616[i].left;

		double minDst = maxDst;
		MV bestMV;
		for(int j=0; j<Block1616Num; j++){				
			double dst = (left - si->Holes[j].x)*(left - si->Holes[j].x) + (top - si->Holes[j].y)*(top - si->Holes[j].y);
			if(dst < minDst){
				minDst = dst;
				bestMV = mcroBlockMV[j];
			}
		}

		MV *mbMV = &(si->mcroBlockMV[i]);
		mbMV->Future.x = (double)computeEndPoint(left, -bestMV.vector.x, intervFC, 0, searchRangeRight);
		mbMV->Future.y = (double)computeEndPoint(top, -bestMV.vector.y, intervFC, 0, searchRangeBottom);
		mbMV->Past.x = (double)computeEndPoint(left, bestMV.vector.x, intervPC, 0, searchRangeRight);
		mbMV->Past.y = (double)computeEndPoint(top, bestMV.vector.y, intervPC, 0, searchRangeBottom);
		mbMV->vector.x = mbMV->Past.x - mbMV->Future.x;
		mbMV->vector.y = mbMV->Past.y - mbMV->Future.y;		
	}

	free(mcroBlockMV);
}


void bidirectME(int future, int past, int current, MV *mv, int blockLen){
	unsigned char *futureFrame = decodedFrames[future].interpolateFrame;
	unsigned char *pastFrame = decodedFrames[past].interpolateFrame;

	int blockHeight = codingOption.FrameHeight / blockLen;
	int blockWidth = codingOption.FrameWidth / blockLen;
	double intervFC = (double)(future-current) / (double)(future-past);
	double intervPC = (double)(current-past) / (double)(future-past);
	
	int width2 = codingOption.FrameWidth << 1;
	int height2 = codingOption.FrameHeight << 1;
	int blockLen2 = blockLen << 1;
	int searchRangeBottom = height2 - blockLen2;
	int searchRangeRight = width2 - blockLen2;
	int blkIdx = 0;	//block 1-D index
	for(int i=0; i<blockHeight; i++){
		int y2 = i * blockLen2;	//top point (half pixel precision)

		for(int j=0; j<blockWidth; j++){
			//for each block
			int x2 = j * blockLen2;	//left point (half pixel precision)

			//calculate adaptive search range
			int searchTop, searchBottom, searchLeft, searchRight;
			if(i-1 >= 0)
				searchTop = (int)mv[blkIdx - blockWidth].Future.y + blockLen2;
			else
				searchTop = (int)mv[blkIdx].Future.y;
			if(i+1 < blockHeight)
				searchBottom = (int)mv[blkIdx + blockWidth].Future.y - blockLen2;
			else
				searchBottom = (int)mv[blkIdx].Future.y;
			if(j-1 >= 0)
				searchLeft = (int)mv[blkIdx - 1].Future.x + blockLen2;
			else
				searchLeft = (int)mv[blkIdx].Future.x;
			if(j+1 < blockWidth)
				searchRight = (int)mv[blkIdx + 1].Future.x - blockLen2;
			else
				searchRight = (int)mv[blkIdx].Future.x;

			//bidirectional motion estimation (half pixel precision)
			double minCost = 10000000.0;
			MV bestMV = mv[blkIdx];
			double scale = intervPC / intervFC;
			for(int fy = searchTop; fy<=searchBottom; fy++){
				int py = computeEndPoint(y2, (double)(y2 - fy), scale, 0, searchRangeBottom);	//top point of past frame

				for(int fx = searchLeft; fx<=searchRight; fx++){					
					int px = computeEndPoint(x2, (double)(x2 - fx), scale, 0, searchRangeRight);	//left point of past frame
					
					int mad = 0;	//mean abosolute difference
					for(int m=0; m<blockLen2; m+=2){
						int pIdx = (py+m) * width2 + px;
						int fIdx = (fy+m) * width2 + fx;
						for(int n=0; n<blockLen2; n+=2)
							mad += abs((int)futureFrame[fIdx + n] - (int)pastFrame[pIdx + n]);
					}
					Point vector;
					vector.x = (double)(px - fx);
					vector.y = (double)(py - fy);
					double cost = costFunc((double)mad, &vector);
					if(cost < minCost){
						minCost = cost;
						bestMV.Future.x = (double)fx; bestMV.Future.y = (double)fy;
						bestMV.Past.x = (double)px; bestMV.Past.y = (double)py;
						bestMV.vector = vector;
					}
				}
			}

			mv[blkIdx++] = bestMV;
		}
	}

}

void motionFilterAndComp(int future, int past, int current, SideInfo *si){
	unsigned char *futureFrame = decodedFrames[future].interpolateFrame;
	unsigned char *pastFrame = decodedFrames[past].interpolateFrame;

	int width2 = codingOption.FrameWidth << 1;
	int height2 = codingOption.FrameHeight << 1;
	double intervFC = (double)(future-current) / (double)(future-past);
	double intervPC = (double)(current-past) / (double)(future-past);

	int searchRangeBottom = height2 - 16;
	int searchRangeRight = width2 - 16;

	//calculate MSE for spatial motion smoothing
	for(int i=0; i<Block88Num; i++){
		MV mv = si->BlockMV[i];
		int fy = (int)mv.Future.y;
		int fx = (int)mv.Future.x;
		int py = (int)mv.Past.y;
		int px = (int)mv.Past.x;
		
		double mse = 0.0;	//mean square error
		for(int m=0; m<16; m+=2){
			int pIdx = (py+m) * width2 + px;
			int fIdx = (fy+m) * width2 + fx;
			for(int n=0; n<16; n+=2){
				double dst = (double)futureFrame[fIdx + n] - (double)pastFrame[pIdx + n];
				mse += dst * dst;
			}
		}
		si->MSE[i] = mse / BLOCKSIZE88;
	}

	//weighted vector meadian filter
	MV *blockMV = (MV*)malloc(sizeof(MV) * Block88Num);
	int blkIdx = 0;
	for(int i=0; i<Block88Height; i++){
		int y2 = i << 4;
		for(int j=0; j<Block88Width; j++){
			//for each block
			int x2 = j << 4;
			
			//3x3 window
			int neighborIdx[9];
			int count = 0;
			for(int m=i-1; m<=i+1; m++){
				if(m<0 || m>=Block88Height)
					continue;
				for(int n=j-1;n<=j+1;n++){
					if(n<0 || n>=Block88Width)
						continue;
					neighborIdx[count++] = m * Block88Width + n;
				}
			}

			double minWVMF = 0.0;
			int bestBlkIdx = neighborIdx[0];
			for(int k=1; k<count; k++){
				int otherBlkIdx = neighborIdx[k];
				double dx = si->BlockMV[bestBlkIdx].vector.x - si->BlockMV[otherBlkIdx].vector.x;
				double dy = si->BlockMV[bestBlkIdx].vector.y - si->BlockMV[otherBlkIdx].vector.y;
				minWVMF += si->MSE[bestBlkIdx] / si->MSE[otherBlkIdx] * sqrt(dx*dx + dy*dy);
			}
			for(int c=1; c<count; c++){
				int candBlkIdx = neighborIdx[c];
				double wvmf = 0.0;
				for(int k=0; k<count; k++){
					if(c == k)
						continue;
					int otherBlkIdx = neighborIdx[k];
					double dx = si->BlockMV[candBlkIdx].vector.x - si->BlockMV[otherBlkIdx].vector.x;
					double dy = si->BlockMV[candBlkIdx].vector.y - si->BlockMV[otherBlkIdx].vector.y;
					wvmf += si->MSE[candBlkIdx] / si->MSE[otherBlkIdx] * sqrt(dx*dx + dy*dy);
				}
				if(wvmf < minWVMF){
					minWVMF = wvmf;
					bestBlkIdx = candBlkIdx;
				}
			}

			Point vector = si->BlockMV[bestBlkIdx].vector;
			MV mv;
			mv.Future.x = (double)computeEndPoint(x2, -vector.x, intervFC, 0, searchRangeRight);
			mv.Future.y = (double)computeEndPoint(y2, -vector.y, intervFC, 0, searchRangeBottom);
			mv.Past.x = (double)computeEndPoint(x2, vector.x, intervPC, 0, searchRangeRight);
			mv.Past.y = (double)computeEndPoint(y2, vector.y, intervPC, 0, searchRangeBottom);
			mv.vector.x = mv.Past.x - mv.Future.x;
			mv.vector.y = mv.Past.y - mv.Future.y;
			blockMV[blkIdx++] = mv;
			//si->BlockMV[blkIdx++] = mv;
		}
	}
	for(int i=0; i<Block88Num; i++)
		si->BlockMV[i] = blockMV[i];
	free(blockMV);

	//motion compensation
	blkIdx = 0;
	for(int i=0; i<Block88Height; i++){
		int y = i << 3;		//full pixel precision
		int y2 = y << 1;	//half pixel precision
		for(int j=0; j<Block88Width; j++){
			//for each block
			int x = j << 3;		//full pixel precision
			int x2 = x << 1;	//half pixel precision
			
			MV mv = si->BlockMV[blkIdx++];
			int fy = (int)mv.Future.y;
			int fx = (int)mv.Future.x;
			int py = (int)mv.Past.y;
			int px = (int)mv.Past.x;
			for(int m=0; m<16; m+=2){
				int pIdx = (py+m) * width2 + px;
				int fIdx = (fy+m) * width2 + fx;
				int cIdx = (y+(m>>1)) * codingOption.FrameWidth + x;
				//int testIdx = (y2+m) * width2 + x2;
				for(int n=0; n<16; n+=2){
					si->sideInfoFrame[cIdx + (n>>1)] = (int)((double)futureFrame[fIdx + n] * intervFC + (double)pastFrame[pIdx + n] * intervPC);
					//si->sideInfoFrame[cIdx + (n>>1)] = (int)((double)futureFrame[testIdx + n] * intervFC + (double)pastFrame[testIdx + n] * intervPC);
					si->residueFrame[cIdx + (n>>1)] = ((int)futureFrame[fIdx + n] - (int)pastFrame[pIdx + n]) >> 1;
				}
			}
		}
	}

}

void createSideInfo(int past, int future, int current, SideInfo *si){
	if(decodedFrames[past].hasReferenced == 0){
		//low pass filter (3x3 mean filter)
		decodedFrames[past].lowPassFrame = (unsigned char*)malloc(sizeof(unsigned char)*FrameSize);
		lowPassFilter(decodedFrames[past].reconsFrame, codingOption.FrameWidth, codingOption.FrameHeight, decodedFrames[past].lowPassFrame);
		
		//upsampling (FIR filter) (for half pixel motion estimation)
		decodedFrames[past].interpolateFrame = (unsigned char*)malloc(sizeof(unsigned char)*FrameSize*4);
		switch(codingOption.ParallelMode){
			case 0:	//sequential
			case 2: //CUDA only
				FIR_filter(decodedFrames[past].reconsFrame, codingOption.FrameWidth, codingOption.FrameHeight, decodedFrames[past].interpolateFrame);
				break;
			case 1:	//OpenMP only
			case 3: //OpenMP + CUDA
				FIR_filter_OpenMP(decodedFrames[past].reconsFrame, codingOption.FrameWidth, codingOption.FrameHeight, decodedFrames[past].interpolateFrame);
				break;
			default:
				break;
		}

		decodedFrames[past].hasReferenced = 1;
		//test
		/*char imgName[30];
		sprintf(imgName, "ReconsFrame%d.bmp", past);
		saveToBmp(decodedFrames[past].reconsFrame, codingOption.FrameWidth, codingOption.FrameHeight, imgName);
		sprintf(imgName, "lowPass%d.bmp", past);
		saveToBmp(decodedFrames[past].lowPassFrame, codingOption.FrameWidth, codingOption.FrameHeight, imgName);
		sprintf(imgName, "interpolate%d.bmp", past);
		saveToBmp(decodedFrames[past].interpolateFrame, codingOption.FrameWidth*2, codingOption.FrameHeight*2, imgName);
		system("pause");*/
	}
	if(decodedFrames[future].hasReferenced == 0){
		//low pass filter (3x3 mean filter)
		decodedFrames[future].lowPassFrame = (unsigned char*)malloc(sizeof(unsigned char)*FrameSize);
		lowPassFilter(decodedFrames[future].reconsFrame, codingOption.FrameWidth, codingOption.FrameHeight, decodedFrames[future].lowPassFrame);
		
		//upsampling (FIR filter) (for half pixel motion estimation)
		decodedFrames[future].interpolateFrame = (unsigned char*)malloc(sizeof(unsigned char)*FrameSize*4);
		switch(codingOption.ParallelMode){
			case 0:	//sequential
			case 2: //CUDA only
				FIR_filter(decodedFrames[future].reconsFrame, codingOption.FrameWidth, codingOption.FrameHeight, decodedFrames[future].interpolateFrame);
				break;
			case 1:	//OpenMP only
			case 3: //OpenMP + CUDA
				FIR_filter_OpenMP(decodedFrames[future].reconsFrame, codingOption.FrameWidth, codingOption.FrameHeight, decodedFrames[future].interpolateFrame);
				break;
			default:
				break;
		}

		decodedFrames[future].hasReferenced = 1;
	}
	//forward motion estimation
	switch(codingOption.ParallelMode){
		case 0:	//sequential
			forwardME(future, past, current, si);
			break;
		case 1:	//OpenMP only
			forwardME_OpenMP(future, past, current, si);
			break;
		#ifdef CUDA
		case 2: //CUDA only
		case 3: //OpenMP + CUDA
			forwardME_CUDA(future, past, current, si);
			break;
		#endif
		default:
			break;
	}

	//bidirectional motion estimation
	for(int i=0; i<Block1616Num; i++){	//for half pixel motion estimation
		//for each 16x16 block
		MV mv= si->mcroBlockMV[i];
		mv.Future.x *= 2;
		mv.Future.y *= 2;
		mv.Past.x *= 2;
		mv.Past.y *= 2;
		mv.vector.x = mv.Past.x - mv.Future.x;
		mv.vector.y = mv.Past.y - mv.Future.y;
		si->mcroBlockMV[i] = mv;
	}
	bidirectME(future, past, current, si->mcroBlockMV, 16);

	double intervFC = (double)(future-current) / (double)(future-past);
	double intervPC = (double)(current-past) / (double)(future-past);
	int searchRangeBottom = (codingOption.FrameHeight - 8) << 1;
	int searchRangeRight = (codingOption.FrameWidth - 8) << 1;
	int blkIdx = 0;
	for(int i=0; i<Block88Height; i++){
		int mbRow = (i>>1) * Block1616Width;	//for 16x16 block
		int y2 = i<<4;
		for(int j=0; j<Block88Width; j++){
			//for each 8x8 block
			int x2 = j<<4;
			Point vector = si->mcroBlockMV[mbRow + (j>>1)].vector;
			MV mv;
			mv.Future.x = (double)computeEndPoint(x2, -vector.x, intervFC, 0, searchRangeRight);
			mv.Future.y = (double)computeEndPoint(y2, -vector.y, intervFC, 0, searchRangeBottom);
			mv.Past.x = (double)computeEndPoint(x2, vector.x, intervPC, 0, searchRangeRight);
			mv.Past.y = (double)computeEndPoint(y2, vector.y, intervPC, 0, searchRangeBottom);
			mv.vector.x = mv.Past.x - mv.Future.x;
			mv.vector.y = mv.Past.y - mv.Future.y;
			si->BlockMV[blkIdx++] = mv; //initial motion vector of block 8x8
		}
	}
	bidirectME(future, past, current, si->BlockMV, 8);


	//spatial motion smoothing and motion compensation
	motionFilterAndComp(future, past, current, si);
}

#if 1	//search each integer pixel -> 8 half pixel around it
void initSIR(SIR *sir, unsigned char *BlockModeFlag){
	int searchRangeBottom = codingOption.FrameHeight - 4;
	int searchRangeRight = codingOption.FrameWidth - 4;
	sir->searchCandidate = 0;
	int blk44Idx = 0;
	for(int i=0; i<Block44Height; i++){
		int top = i<<2;
		int searchTop = top - SEARCHRANGE;
		int searchBottom = top + SEARCHRANGE;
		if(searchTop < 0)
			searchTop = 0;
		if(searchBottom > searchRangeBottom)
			searchBottom = searchRangeBottom;
		int blk88row = (i>>1) * Block88Width;
		for(int j=0; j<Block44Width; j++){
			//for each smf
			if(BlockModeFlag[blk88row + (j>>1)] == WZ_B){
				int left = j<<2;

				int searchLeft = left - SEARCHRANGE;
				int searchRight = left + SEARCHRANGE;
				if(searchLeft < 0)
					searchLeft = 0;
				if(searchRight > searchRangeRight)
					searchRight = searchRangeRight;

				SearchInfo searchInfo;
				searchInfo.searchTop = searchTop;
				searchInfo.searchBottom = searchBottom;
				searchInfo.searchLeft = searchLeft;
				searchInfo.searchRight = searchRight;
				searchInfo.top = top;
				searchInfo.left = left;
				searchInfo.searchWidth = searchRight - searchLeft + 1;
				searchInfo.searchRange = searchInfo.searchWidth * (searchBottom - searchTop + 1);
				searchInfo.blkIdx = blk44Idx;
				sir->searchInfo[sir->searchCandidate] = searchInfo;
				sir->searchCandidate++;
			}
			blk44Idx++;
		}
	}
}

void subPelRefine(double *block, unsigned char *referenceFrame, int *bestX, int *bestY, int width2, int height2){
	int searchRangeBottom = height2 - 8;
	int searchRangeRight = width2 - 8;
	int intX = (*bestX)<<1;
	int intY = (*bestY)<<1;
	double minCost = 10000000.0;
	for(int i=-1; i<=1; i++){
		int top = intY+i;
		if(top<0 || top>searchRangeBottom)
			continue;
		for(int j=-1; j<=1; j++){
			int left = intX+j;
			if(left<0 || left>searchRangeRight)
				continue;
			double cost = 0.0;
			int idxInBlk = 0;
			int frameRow = top * width2 + left;
			int width4 = width2<<1;
			for(int m=0;m<8;m+=2){
				//int frameRow = (top+m)*width2 + left;
				//int blkRow = (m>>1)<<2;
				//for(int n=0;n<8;n+=2){
				cost += abs(block[idxInBlk++] - (double)referenceFrame[frameRow]);
				cost += abs(block[idxInBlk++] - (double)referenceFrame[frameRow + 2]);
				cost += abs(block[idxInBlk++] - (double)referenceFrame[frameRow + 4]);
				cost += abs(block[idxInBlk++] - (double)referenceFrame[frameRow + 6]);
				//}
				frameRow += width4;
			}

			//Point mv;
			//mv.x = (double)(x-left);
			//mv.y = (double)(y-top);
			//cost = costFunc(cost,&mv);

			if(cost < minCost){
				minCost = cost;
				*bestX = left;
				*bestY = top;
			}
		}
	}
}

void updateSI(SideInfo *si, unsigned char *reconsFrame, int current, int past, int future, SIR *sir){
	unsigned char *futureFrame = decodedFrames[future].reconsFrame;
	unsigned char *pastFrame = decodedFrames[past].reconsFrame;
	unsigned char *futureFrame_half = decodedFrames[future].interpolateFrame;
	unsigned char *pastFrame_half = decodedFrames[past].interpolateFrame;
	int width = codingOption.FrameWidth;
	int height = codingOption.FrameHeight;
	int width2 = width << 1;
	int height2 = height << 1;
	double intervFC = (double)(future-current) / (double)(future-past);
	double intervPC = (double)(current-past) / (double)(future-past);

	for(int i=0; i<sir->searchCandidate; i++){

		SearchInfo searchInfo = sir->searchInfo[i];
		int top = searchInfo.top;
		int left = searchInfo.left;

		double block[BLOCKSIZE44];
		int idxInBlk = 0;
		for(int m=0; m<4; m++){
			int rowInFrame = (top+m) * codingOption.FrameWidth + left;
			//int rowInBlk = m << 2; //m*4
			for(int n=0; n<4; n++){
				int idxInFrame = rowInFrame + n;
				//int idxInBlk = rowInBlk + n;
				block[idxInBlk++] = (double)reconsFrame[idxInFrame];
			}
		}

		//bidirectional motion estimation
		int pBestX, pBestY, fBestX, fBestY;
		double pMinCost = 10000000.0;
		double fMinCost = 10000000.0;
		for(int y=searchInfo.searchTop; y<=searchInfo.searchBottom; y++){
			for(int x=searchInfo.searchLeft; x<=searchInfo.searchRight; x++){
				double pCost = 0.0;
				double fCost = 0.0;
				idxInBlk = 0;
				for(int m=0; m<4; m++){
					int rowInFrame = (y+m) * width + x;
					//int rowInBlk = m << 2;
					for(int n=0; n<4; n++){
						//mean absolute difference
						pCost += abs(block[idxInBlk] - (double)pastFrame[rowInFrame + n]);
						fCost += abs(block[idxInBlk] - (double)futureFrame[rowInFrame + n]);
						idxInBlk++;
						//double diff = block[rowInBlk + (n>>1)] - (double)pastFrame[rowInFrame + n];
						//pCost += diff * diff;
						//diff = block[rowInBlk + (n>>1)] - (double)futureFrame[rowInFrame + n];
						//fCost += diff * diff;
					}
				}
				Point mv;

				mv.x = (double)((x-left)<<1);
				mv.y = (double)((y-top)<<1);

				pCost = costFunc(pCost,&mv);
				fCost = costFunc(fCost,&mv);
				if(pCost < pMinCost){
					pBestX = x;
					pBestY = y;
					pMinCost = pCost;
				}
				if(fCost < fMinCost){
					fBestX = x;
					fBestY = y;
					fMinCost = fCost;
				}
			}
		}

		subPelRefine(block, pastFrame_half, &pBestX, &pBestY, width2, height2);
		subPelRefine(block, futureFrame_half, &fBestX, &fBestY, width2, height2);
		//subPelRefine(reconsFrame, past, left<<1, top<<1, &pBestX, &pBestY, pMinCost, block);
		//subPelRefine(reconsFrame, future, left<<1, top<<1, &fBestX, &fBestY, fMinCost, block);
	
		for(int m=0; m<8; m+=2){
			int rowInFrame = (top+(m>>1)) * width + left;
			int pRow = (pBestY+m) * width2 + pBestX;
			int fRow = (fBestY+m) * width2 + fBestX;
			for(int n=0; n<8; n+=2){
				//int idxInFrame = rowInFrame + (n>>1);
				si->sideInfoFrame[rowInFrame + (n>>1)] = (int)((double)pastFrame_half[pRow + n] * intervPC + (double)futureFrame_half[fRow + n] * intervFC);

				//si->residueFrame[idxInFrame] = si->sideInfoFrame[idxInFrame] - oriFrames[current][idxInFrame];
				//int newResidue = ((int)futureFrame[fRow + n] - (int)pastFrame[pRow + n]) >> 1;
				//if(abs(newResidue) > abs(si->residueFrame[idxInFrame]))
				//	si->residueFrame[idxInFrame] = newResidue;
			}
		}
	}

}

void updateSI_OpenMP(SideInfo *si, unsigned char *reconsFrame, int current, int past, int future, SIR *sir){
	unsigned char *futureFrame = decodedFrames[future].reconsFrame;
	unsigned char *pastFrame = decodedFrames[past].reconsFrame;
	unsigned char *futureFrame_half = decodedFrames[future].interpolateFrame;
	unsigned char *pastFrame_half = decodedFrames[past].interpolateFrame;
	int width = codingOption.FrameWidth;
	int height = codingOption.FrameHeight;
	int width2 = width << 1;
	int height2 = height << 1;
	double intervFC = (double)(future-current) / (double)(future-past);
	double intervPC = (double)(current-past) / (double)(future-past);
#pragma omp parallel for
	for(int i=0; i<sir->searchCandidate; i++){

		SearchInfo searchInfo = sir->searchInfo[i];
		int top = searchInfo.top;
		int left = searchInfo.left;

		double block[BLOCKSIZE44];
		int idxInBlk = 0;
		for(int m=0; m<4; m++){
			int rowInFrame = (top+m) * codingOption.FrameWidth + left;
			//int rowInBlk = m << 2; //m*4
			for(int n=0; n<4; n++){
				int idxInFrame = rowInFrame + n;
				//int idxInBlk = rowInBlk + n;
				block[idxInBlk++] = (double)reconsFrame[idxInFrame];
			}
		}

		//bidirectional motion estimation
		int pBestX, pBestY, fBestX, fBestY;
		double pMinCost = 10000000.0;
		double fMinCost = 10000000.0;
		for(int y=searchInfo.searchTop; y<=searchInfo.searchBottom; y++){
			for(int x=searchInfo.searchLeft; x<=searchInfo.searchRight; x++){
				double pCost = 0.0;
				double fCost = 0.0;
				idxInBlk = 0;
				for(int m=0; m<4; m++){
					int rowInFrame = (y+m) * width + x;
					//int rowInBlk = m << 2;
					for(int n=0; n<4; n++){
						//mean absolute difference
						pCost += abs(block[idxInBlk] - (double)pastFrame[rowInFrame + n]);
						fCost += abs(block[idxInBlk] - (double)futureFrame[rowInFrame + n]);
						idxInBlk++;
						//double diff = block[rowInBlk + (n>>1)] - (double)pastFrame[rowInFrame + n];
						//pCost += diff * diff;
						//diff = block[rowInBlk + (n>>1)] - (double)futureFrame[rowInFrame + n];
						//fCost += diff * diff;
					}
				}
				Point mv;

				mv.x = (double)((x-left)<<1);
				mv.y = (double)((y-top)<<1);

				pCost = costFunc(pCost,&mv);
				fCost = costFunc(fCost,&mv);
				if(pCost < pMinCost){
					pBestX = x;
					pBestY = y;
					pMinCost = pCost;
				}
				if(fCost < fMinCost){
					fBestX = x;
					fBestY = y;
					fMinCost = fCost;
				}
			}
		}

		subPelRefine(block, pastFrame_half, &pBestX, &pBestY, width2, height2);
		subPelRefine(block, futureFrame_half, &fBestX, &fBestY, width2, height2);
		//subPelRefine(reconsFrame, past, left<<1, top<<1, &pBestX, &pBestY, pMinCost, block);
		//subPelRefine(reconsFrame, future, left<<1, top<<1, &fBestX, &fBestY, fMinCost, block);
	
		for(int m=0; m<8; m+=2){
			int rowInFrame = (top+(m>>1)) * width + left;
			int pRow = (pBestY+m) * width2 + pBestX;
			int fRow = (fBestY+m) * width2 + fBestX;
			for(int n=0; n<8; n+=2){
				//int idxInFrame = rowInFrame + (n>>1);
				si->sideInfoFrame[rowInFrame + (n>>1)] = (int)((double)pastFrame_half[pRow + n] * intervPC + (double)futureFrame_half[fRow + n] * intervFC);

				//si->residueFrame[idxInFrame] = si->sideInfoFrame[idxInFrame] - oriFrames[current][idxInFrame];
				//int newResidue = ((int)futureFrame[fRow + n] - (int)pastFrame[pRow + n]) >> 1;
				//if(abs(newResidue) > abs(si->residueFrame[idxInFrame]))
				//	si->residueFrame[idxInFrame] = newResidue;
			}
		}
	}

}

#else	//search each half pixel
void initSIR(SIR *sir, unsigned char *BlockModeFlag){
	int width2 = codingOption.FrameWidth << 1;
	int height2 = codingOption.FrameHeight << 1;
	int searchRangeBottom = height2 - 8;
	int searchRangeRight = width2 - 8;
	int searchRange = SEARCHRANGE << 1;
	sir->searchCandidate = 0;
	for(int i=0; i<Block44Height; i++){
		int top = i<<2;
		int searchTop = (top<<1) - searchRange;
		int searchBottom = (top<<1) + searchRange;
		if(searchTop < 0)
			searchTop = 0;
		if(searchBottom > searchRangeBottom)
			searchBottom = searchRangeBottom;
		int blk88row = (i>>1) * Block88Width;
		for(int j=0; j<Block44Width; j++){
			//for each smf
			if(BlockModeFlag[blk88row + (j>>1)] == WZ_B){
				int left = j<<2;

				int searchLeft = (left<<1) - searchRange;
				int searchRight = (left<<1) + searchRange;
				if(searchLeft < 0)
					searchLeft = 0;
				if(searchRight > searchRangeRight)
					searchRight = searchRangeRight;

				SearchInfo searchInfo;
				searchInfo.searchTop = searchTop;
				searchInfo.searchBottom = searchBottom;
				searchInfo.searchLeft = searchLeft;
				searchInfo.searchRight = searchRight;
				searchInfo.top = top;
				searchInfo.left = left;
				sir->searchInfo[sir->searchCandidate] = searchInfo;
				sir->searchCandidate++;
			}
		}
	}
}


void updateSI(SideInfo *si, unsigned char *reconsFrame, int current, int past, int future, SIR *sir){
	unsigned char *futureFrame = decodedFrames[future].interpolateFrame;
	unsigned char *pastFrame = decodedFrames[past].interpolateFrame;
	int width2 = codingOption.FrameWidth << 1;
	int height2 = codingOption.FrameHeight << 1;
	int searchRangeBottom = height2 - 8;
	int searchRangeRight = width2 - 8;
	double intervFC = (double)(future-current) / (double)(future-past);
	double intervPC = (double)(current-past) / (double)(future-past);
//#pragma omp parallel for
	for(int i=0; i<sir->searchCandidate; i++){

		SearchInfo searchInfo = sir->searchInfo[i];
		int top = searchInfo.top;
		int left = searchInfo.left;

		double block[BLOCKSIZE44];
		for(int m=0; m<4; m++){
			int rowInFrame = (top+m) * codingOption.FrameWidth + left;
			int rowInBlk = m << 2; //m*4
			for(int n=0; n<4; n++){
				int idxInFrame = rowInFrame + n;
				int idxInBlk = rowInBlk + n;
				block[idxInBlk] = (double)reconsFrame[idxInFrame];
			}
		}

		//bidirectional motion estimation
		int pBestX, pBestY, fBestX, fBestY;
		double pMinCost = 10000000.0;
		double fMinCost = 10000000.0;
		for(int y=searchInfo.searchTop; y<=searchInfo.searchBottom; y++){
			for(int x=searchInfo.searchLeft; x<=searchInfo.searchRight; x++){
				double pCost = 0.0;
				double fCost = 0.0;
				for(int m=0; m<8; m+=2){
					int rowInFrame = (y+m) * width2 + x;
					int rowInBlk = (m>>1) << 2;
					for(int n=0; n<8; n+=2){
						//mean absolute difference
						pCost += abs(block[rowInBlk + (n>>1)] - (double)pastFrame[rowInFrame + n]);
						fCost += abs(block[rowInBlk + (n>>1)] - (double)futureFrame[rowInFrame + n]);
						/*double diff = block[rowInBlk + (n>>1)] - (double)pastFrame[rowInFrame + n];
						pCost += diff * diff;
						diff = block[rowInBlk + (n>>1)] - (double)futureFrame[rowInFrame + n];
						fCost += diff * diff;*/
					}
				}
				Point mv;
				mv.x = (double)(x-(left<<1));
				mv.y = (double)(y-(top<<1));
				pCost = costFunc(pCost,&mv);
				fCost = costFunc(fCost,&mv);
				if(pCost < pMinCost){
					pBestX = x;
					pBestY = y;
					pMinCost = pCost;
				}
				if(fCost < fMinCost){
					fBestX = x;
					fBestY = y;
					fMinCost = fCost;
				}
			}
		}
	
		for(int m=0; m<8; m+=2){
			int rowInFrame = (top+(m>>1)) * codingOption.FrameWidth + left;
			int pRow = (pBestY+m) * width2 + pBestX;
			int fRow = (fBestY+m) * width2 + fBestX;
			for(int n=0; n<8; n+=2){
				//int idxInFrame = rowInFrame + (n>>1);
				si->sideInfoFrame[rowInFrame + (n>>1)] = (int)((double)pastFrame[pRow + n] * intervPC + (double)futureFrame[fRow + n] * intervFC);

				//si->residueFrame[idxInFrame] = si->sideInfoFrame[idxInFrame] - oriFrames[current][idxInFrame];
				/*int newResidue = ((int)futureFrame[fRow + n] - (int)pastFrame[pRow + n]) >> 1;
				if(abs(newResidue) > abs(si->residueFrame[idxInFrame]))
					si->residueFrame[idxInFrame] = newResidue;*/
			}
		}
	}

}

void updateSI_OpenMP(SideInfo *si, unsigned char *reconsFrame, int current, int past, int future, SIR *sir){
	unsigned char *futureFrame = decodedFrames[future].interpolateFrame;
	unsigned char *pastFrame = decodedFrames[past].interpolateFrame;
	int width2 = codingOption.FrameWidth << 1;
	int height2 = codingOption.FrameHeight << 1;
	int searchRangeBottom = height2 - 8;
	int searchRangeRight = width2 - 8;
	double intervFC = (double)(future-current) / (double)(future-past);
	double intervPC = (double)(current-past) / (double)(future-past);
#pragma omp parallel for
	for(int i=0; i<sir->searchCandidate; i++){

		SearchInfo searchInfo = sir->searchInfo[i];
		int top = searchInfo.top;
		int left = searchInfo.left;

		double block[BLOCKSIZE44];
		for(int m=0; m<4; m++){
			int rowInFrame = (top+m) * codingOption.FrameWidth + left;
			int rowInBlk = m << 2; //m*4
			for(int n=0; n<4; n++){
				int idxInFrame = rowInFrame + n;
				int idxInBlk = rowInBlk + n;
				block[idxInBlk] = (double)reconsFrame[idxInFrame];
			}
		}

		//bidirectional motion estimation
		int pBestX, pBestY, fBestX, fBestY;
		double pMinCost = 10000000.0;
		double fMinCost = 10000000.0;
		for(int y=searchInfo.searchTop; y<=searchInfo.searchBottom; y++){
			for(int x=searchInfo.searchLeft; x<=searchInfo.searchRight; x++){
				double pCost = 0.0;
				double fCost = 0.0;
				for(int m=0; m<8; m+=2){
					int rowInFrame = (y+m) * width2 + x;
					int rowInBlk = (m>>1) << 2;
					for(int n=0; n<8; n+=2){
						//mean absolute difference
						pCost += abs(block[rowInBlk + (n>>1)] - (double)pastFrame[rowInFrame + n]);
						fCost += abs(block[rowInBlk + (n>>1)] - (double)futureFrame[rowInFrame + n]);
						/*double diff = block[rowInBlk + (n>>1)] - (double)pastFrame[rowInFrame + n];
						pCost += diff * diff;
						diff = block[rowInBlk + (n>>1)] - (double)futureFrame[rowInFrame + n];
						fCost += diff * diff;*/
					}
				}
				Point mv;
				mv.x = (double)(x-(left<<1));
				mv.y = (double)(y-(top<<1));
				pCost = costFunc(pCost,&mv);
				fCost = costFunc(fCost,&mv);
				if(pCost < pMinCost){
					pBestX = x;
					pBestY = y;
					pMinCost = pCost;
				}
				if(fCost < fMinCost){
					fBestX = x;
					fBestY = y;
					fMinCost = fCost;
				}
			}
		}
	
		for(int m=0; m<8; m+=2){
			int rowInFrame = (top+(m>>1)) * codingOption.FrameWidth + left;
			int pRow = (pBestY+m) * width2 + pBestX;
			int fRow = (fBestY+m) * width2 + fBestX;
			for(int n=0; n<8; n+=2){
				//int idxInFrame = rowInFrame + (n>>1);
				si->sideInfoFrame[rowInFrame + (n>>1)] = (int)((double)pastFrame[pRow + n] * intervPC + (double)futureFrame[fRow + n] * intervFC);

				//si->residueFrame[idxInFrame] = si->sideInfoFrame[idxInFrame] - oriFrames[current][idxInFrame];
				/*int newResidue = ((int)futureFrame[fRow + n] - (int)pastFrame[pRow + n]) >> 1;
				if(abs(newResidue) > abs(si->residueFrame[idxInFrame]))
					si->residueFrame[idxInFrame] = newResidue;*/
			}
		}
	}

}

#endif