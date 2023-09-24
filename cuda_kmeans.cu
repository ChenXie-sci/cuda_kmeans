/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         cuda_kmeans.cu  (CUDA version)                            */
/*   Description:  Implementation of simple k-means clustering algorithm     */
/*                 This program takes an array of N data objects, each with  */
/*                 M coordinates and performs a k-means clustering given a   */
/*                 user-provided value of the number of clusters (K). The    */
/*                 clustering results are saved in 2 arrays:                 */
/*                 1. a returned array of size [K][N] indicating the center  */
/*                    coordinates of K clusters                              */
/*                 2. membership[N] stores the cluster center ids, each      */
/*                    corresponding to the cluster a data object is assigned */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department, Northwestern University                        */
/*            email: wkliao@ece.northwestern.edu                             */
/*   Copyright, 2005, Wei-keng Liao                                          */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// Copyright (c) 2005 Wei-keng Liao
// Copyright (c) 2011 Serban Giuroiu
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// -----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>

#include "kmeans.h"

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

#define MAX_THREADS_PER_BLOCK 1024

static inline int nextPowerOfTwo(int n) {
    n--;

    n = n >>  1 | n;
    n = n >>  2 | n;
    n = n >>  4 | n;
    n = n >>  8 | n;
    n = n >> 16 | n;
//  n = n >> 32 | n;    //  For 64-bit ints

    return ++n;
}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__host__ __device__ inline static
float euclid_dist_2(int    numCoords,
                    int    numObjs,
                    int    numClusters,
                    float *objects,     // [numCoords][numObjs]
                    float *clusters,    // [numCoords][numClusters]
                    int    objectId,
                    int    clusterId)
{
    int i;
    float ans=0.0;

    for (i = 0; i < numCoords; i++) {
        ans += (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]) *
               (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]);
    }

    return(ans);
}

/*----< find_nearest_cluster() >---------------------------------------------*/
__global__ static
void find_nearest_cluster(int numCoords,
                          int numObjs,
                          int numClusters,
                          float *objects,           //  [numCoords][numObjs]
                          float *deviceClusters,    //  [numCoords][numClusters]
                          int *membership,          //  [numObjs]
                          int *intermediates)
{
    extern __shared__ char sharedMemory[];

    //  The type chosen for membershipChanged must be large enough to support
    //  reductions! There are blockDim.x elements, one for each thread in the
    //  block.
    unsigned char *membershipChanged = (unsigned char *)sharedMemory;
    float *clusters = (float *)(sharedMemory + blockDim.x);

    membershipChanged[threadIdx.x] = 0;

    //  BEWARE: We can overrun our shared memory here if there are too many
    //  clusters or too many coordinates!
    for (int i = threadIdx.x; i < numClusters; i += blockDim.x) {
        for (int j = 0; j < numCoords; j++) {
            clusters[numClusters * j + i] = deviceClusters[numClusters * j + i];
        }
    }
    __syncthreads();

    int objectId = blockDim.x * blockIdx.x + threadIdx.x;

    if (objectId < numObjs) {
        int   index, i;
        float dist, min_dist;

        /* find the cluster id that has min distance to object */
        index    = 0;
        min_dist = euclid_dist_2(numCoords, numObjs, numClusters,
                                 objects, clusters, objectId, 0);

        for (i=1; i<numClusters; i++) {
            dist = euclid_dist_2(numCoords, numObjs, numClusters,
                                 objects, clusters, objectId, i);
            /* no need square root */
            if (dist < min_dist) { /* find the min and its array index */
                min_dist = dist;
                index    = i;
            }
        }

        if (membership[objectId] != index) {
            membershipChanged[threadIdx.x] = 1;
        }

        /* assign the membership to object objectId */
        membership[objectId] = index;

        __syncthreads();    //  For membershipChanged[]

        //  blockDim.x *must* be a power of two!
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                membershipChanged[threadIdx.x] +=
                    membershipChanged[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            intermediates[blockIdx.x] = membershipChanged[0];
        }
    }
}

__global__ static
void compute_delta(int *deviceIntermediates,
                   int numIntermediates,    //  The actual number of intermediates
                   int numIntermediates2,   //  The next power of two
                   int *delta)              //  Array of delta for each thread block
{
    //  The number of elements in this array should be equal to
    //  numIntermediates2, the number of threads launched. It *must* be a power
    //  of two!
    extern __shared__ unsigned int intermediates[];

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    //  Copy global intermediate values into shared memory.
    intermediates[threadIdx.x] =
        (index < numIntermediates) ? deviceIntermediates[index] : 0;

    __syncthreads();

    //  numIntermediates2 *must* be a power of two!
    for (unsigned int s = numIntermediates2 / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            intermediates[threadIdx.x] += intermediates[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Update the appropriate position in the array of delta
    if (threadIdx.x == 0) {
        delta[blockIdx.x] = intermediates[0];
    }
}

__global__ static
void update_clusters_sum(int numCoords,
                     int numObjs,
                     int numClusters,
                     float *objects,                      //  [numCoords][numObjs]
                     unsigned int *newDeviceClustersSize, //  [numClusters]
                     float *newDeviceClusters,            //  [numCoords][numClusters]
                     int *membership)                     //  [numObjs]
{
    extern __shared__ float sharedMemory1[];
    float *sharedClusters = sharedMemory1; // numCoords*numClusters float
    unsigned int *sharedClustersSize =
        (unsigned int *)&sharedClusters[numCoords*numClusters]; // numClusters unsigned int

    // numClusters is much smaller than number of threads (1024)
    if(threadIdx.x < numClusters)
        sharedClustersSize[threadIdx.x] = 0;

    int i;
    for(i = threadIdx.x; i < numClusters*numCoords; i += blockDim.x)
        sharedClusters[i] = 0.0;

    __syncthreads();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < numObjs) {
        int index, j;

        /* find the array index of nearest cluster center */
        index = membership[tid];

        /* update new cluster centers : sum of objects located within */
        atomicAdd(&sharedClustersSize[index], 1);
        for(j = 0; j < numCoords; j++)
            atomicAdd(&sharedClusters[j*numClusters + index], objects[j*numObjs + tid]);
    }

    __syncthreads();

    // Aggregate local buffers to global arrays
    if(threadIdx.x < numClusters)
        atomicAdd(&newDeviceClustersSize[threadIdx.x], sharedClustersSize[threadIdx.x]);
    for(i = threadIdx.x; i < numClusters*numCoords; i += blockDim.x) {
        atomicAdd(&newDeviceClusters[i], sharedClusters[i]);
    }
}


__global__ static
void update_clusters_average(int numClusters,
                             unsigned int *newDeviceClustersSize, //  [numClusters]
                             float *newDeviceClusters,            //  [numCoords][numClusters]
                             float *deviceClusters)               //  [numCoords][numClusters]  
{
    int position = blockIdx.x + threadIdx.x*numClusters;
    if(newDeviceClustersSize[blockIdx.x]) {
        deviceClusters[position] = 
            newDeviceClusters[position]/newDeviceClustersSize[blockIdx.x];
    }
    newDeviceClusters[position] = 0.0;
}

/*----< cuda_kmeans() >-------------------------------------------------------*/
//
//  ----------------------------------------
//  DATA LAYOUT
//
//  objects                [numObjs][numCoords]
//  clusters               [numClusters][numCoords]
//  dimObjects             [numCoords][numObjs]
//  dimClusters            [numCoords][numClusters]
//  deviceObjects          [numCoords][numObjs]
//  deviceClusters         [numCoords][numClusters]
//  newDeviceClustersSize  [numClusters]
//  newDeviceClusters      [numCoords][numClusters]
//  ----------------------------------------
//
/* return an array of cluster centers of size [numClusters][numCoords]       */
float** cuda_kmeans(float **objects,      /* in: [numObjs][numCoords] */
                   int     numCoords,    /* no. features */
                   int     numObjs,      /* no. objects */
                   int     numClusters,  /* no. clusters */
                   float   threshold,    /* % objects change membership */
                   int    *membership,   /* out: [numObjs] */
                   int    *loop_iterations)
{
    int      i, j, index, loop=0;
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
    float    delta;          /* % of objects change their clusters */
    float  **dimObjects;
    float  **clusters;       /* out: [numClusters][numCoords] */
    float  **dimClusters;

    float *deviceObjects;
    float *deviceClusters;
    int *deviceMembership;
    int *deviceIntermediates;

    unsigned int *newDeviceClustersSize; /* [numClusters] */
    float        *newDeviceClusters;     /* [numCoords][numClusters] */

    //  Copy objects given in [numObjs][numCoords] layout to new
    //  [numCoords][numObjs] layout
    malloc2D(dimObjects, numCoords, numObjs, float);
    for (i = 0; i < numCoords; i++) {
        for (j = 0; j < numObjs; j++) {
            dimObjects[i][j] = objects[j][i];
        }
    }

    /* pick first numClusters elements of objects[] as initial cluster centers*/
    malloc2D(dimClusters, numCoords, numClusters, float);
    for (i = 0; i < numCoords; i++) {
        for (j = 0; j < numClusters; j++) {
            dimClusters[i][j] = dimObjects[i][j];
        }
    }

    /* initialize membership[] */
    for (i=0; i<numObjs; i++) membership[i] = -1;

    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);

    checkCuda(cudaMalloc(&newDeviceClustersSize, numClusters*sizeof(unsigned int)));
    cudaMemset(newDeviceClustersSize, 0, numClusters*sizeof(unsigned int));

    checkCuda(cudaMalloc(&newDeviceClusters, numCoords*numClusters*sizeof(float)));
    cudaMemset(newDeviceClusters, 0, numCoords*numClusters*sizeof(float));

    //  To support reduction, numThreadsPerClusterBlock *must* be a power of
    //  two, and it *must* be no larger than the number of bits that will
    //  fit into an unsigned char, the type used to keep track of membership
    //  changes in the kernel.
    const unsigned int numThreadsPerClusterBlock = 128;
    const unsigned int numClusterBlocks =
        (numObjs + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;
    const unsigned int clusterBlockSharedDataSize =
        numThreadsPerClusterBlock * sizeof(unsigned char) +
        numClusters * numCoords * sizeof(float);

    unsigned int numReductionThreads = nextPowerOfTwo(numClusterBlocks);
    // Limit the number of threads in a block
    unsigned int numReductionBlocks = numReductionThreads/MIN(numReductionThreads, MAX_THREADS_PER_BLOCK);
    numReductionThreads = MIN(numReductionThreads, MAX_THREADS_PER_BLOCK);
    const unsigned int reductionBlockSharedDataSize =
        numReductionThreads * sizeof(unsigned int);

    const unsigned int numUpdateThreads = MAX_THREADS_PER_BLOCK;
    unsigned int numUpdateBlocks = (numObjs + numUpdateThreads - 1) / numUpdateThreads;
    const unsigned int updateBlockSharedDataSize =
        nextPowerOfTwo(sizeof(unsigned int)*numClusters + sizeof(float)*numClusters*numCoords);

    // // For debugging
    // printf("\n\nHELLO\t%d*%d = %d, clusterBlockSharedDataSize = %d\n\n", numClusterBlocks, numThreadsPerClusterBlock, numClusterBlocks*numThreadsPerClusterBlock, clusterBlockSharedDataSize);
    // printf("\n\nHELLO\t%d*%d = %d, reductionBlockSharedDataSize = %d\n\n", numReductionBlocks, numReductionThreads, numReductionBlocks*numReductionThreads, reductionBlockSharedDataSize);
    // printf("\n\nHELLO\t%d*%d = %d, updateBlockSharedDataSize = %d\n\n", numUpdateBlocks, numUpdateThreads, numUpdateBlocks*numUpdateThreads, updateBlockSharedDataSize);

    checkCuda(cudaMalloc(&deviceObjects, numObjs*numCoords*sizeof(float)));
    checkCuda(cudaMalloc(&deviceClusters, numClusters*numCoords*sizeof(float)));
    checkCuda(cudaMalloc(&deviceMembership, numObjs*sizeof(int)));
    checkCuda(cudaMalloc(&deviceIntermediates, numReductionThreads*sizeof(unsigned int)));

    checkCuda(cudaMemcpy(deviceObjects, dimObjects[0],
              numObjs*numCoords*sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(deviceMembership, membership,
              numObjs*sizeof(int), cudaMemcpyHostToDevice));

    // Only need to initialize once
    checkCuda(cudaMemcpy(deviceClusters, dimClusters[0],
              numClusters*numCoords*sizeof(float), cudaMemcpyHostToDevice));

    int *deviceDelta;
    checkCuda(cudaMalloc(&deviceDelta, numReductionBlocks*sizeof(int)));
    cudaMemset(deviceDelta, 0, numReductionBlocks*sizeof(int));

    int *hostDelta = (int*) calloc(numReductionBlocks, sizeof(int));
    assert(hostDelta != NULL);

    do {
        find_nearest_cluster
            <<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
            (numCoords, numObjs, numClusters,
             deviceObjects, deviceClusters, deviceMembership, deviceIntermediates);

        cudaDeviceSynchronize(); checkLastCudaError();

        compute_delta <<< numReductionBlocks, numReductionThreads, reductionBlockSharedDataSize >>>
            (deviceIntermediates, numClusterBlocks, numReductionThreads, deviceDelta);
        
        cudaDeviceSynchronize(); checkLastCudaError();

        int d = 0;
        checkCuda(cudaMemcpy(hostDelta, deviceDelta,
                  numReductionBlocks*sizeof(int), cudaMemcpyDeviceToHost));
        for(i = 0; i < numReductionBlocks; i++)
            d += hostDelta[i];
        delta = (float)d/numObjs;

        // Update clusters on device instead of host
        update_clusters_sum <<< numUpdateBlocks, numUpdateThreads, updateBlockSharedDataSize >>>
            (numCoords, numObjs, numClusters,
             deviceObjects, newDeviceClustersSize, newDeviceClusters, deviceMembership);
        
        cudaDeviceSynchronize(); checkLastCudaError();

        /* average the sum and replace old cluster centers */
        update_clusters_average <<< numClusters, numCoords >>>
            (numClusters, newDeviceClustersSize, newDeviceClusters, deviceClusters);
        
        cudaThreadSynchronize(); checkLastCudaError();
        cudaMemset(newDeviceClustersSize, 0, numClusters*sizeof(unsigned int));

    } while (delta > threshold && loop++ < 500);

    *loop_iterations = loop + 1;

    // Copy results to host
    checkCuda(cudaMemcpy(membership, deviceMembership,
              numObjs*sizeof(int), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(newClusterSize, newDeviceClustersSize,
              numClusters*sizeof(int), cudaMemcpyDeviceToHost));

    // Calculate the final clusters
    malloc2D(clusters, numClusters, numCoords, float);
    memset(clusters[0], 0, numCoords*numClusters*sizeof(float));

    for (i=0; i<numObjs; i++) {
       /* find the array index of nestest cluster center */
        index = membership[i];

       /* update new cluster centers : sum of objects located within */
        for (j=0; j<numCoords; j++)
            // Save directly to clusters; newClusters is not used (deleted)
            clusters[index][j] += objects[i][j];
    }

    for (i = 0; i < numClusters; i++) {
        for (j = 0; j < numCoords; j++) {
            clusters[i][j] = dimClusters[j][i]/newClusterSize[i];
        }
    }

    checkCuda(cudaFree(deviceObjects));
    checkCuda(cudaFree(deviceClusters));
    checkCuda(cudaFree(deviceMembership));
    checkCuda(cudaFree(deviceIntermediates));

    checkCuda(cudaFree(newDeviceClustersSize));
    checkCuda(cudaFree(newDeviceClusters));

    free(dimObjects[0]);
    free(dimObjects);
    free(dimClusters[0]);
    free(dimClusters);
    free(newClusterSize);

    return clusters;
}

