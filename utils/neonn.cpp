#include<iostream>
#include<arm_neon.h>

#define BLOCK_SIZE 4
using namespace std;

void matrix_multiply_4x4_neon(uint32_t A[4][4], uint32_t B[4][4], uint32_t C[4][4]) {
        // these are the columns A
        uint32x4_t A0;
        uint32x4_t A1;
        uint32x4_t A2;
        uint32x4_t A3;
        
        // these are the columns B
        uint32x4_t B0;
        uint32x4_t B1;
        uint32x4_t B2;
        uint32x4_t B3;
        
        // these are the columns C
        uint32x4_t C0;
        uint32x4_t C1;
        uint32x4_t C2;
        uint32x4_t C3;
        
        A0 = vld1q_u32(A[0]);
        A1 = vld1q_u32(A[1]);
        A2 = vld1q_u32(A[2]);
        A3 = vld1q_u32(A[3]);
        
        // Zero accumulators for C values
        C0 = vmovq_n_u32(0);
        C1 = vmovq_n_u32(0);
        C2 = vmovq_n_u32(0);
        C3 = vmovq_n_u32(0);
        
        // Multiply accumulate in 4x1 blocks, i.e. each column in C
        B0 = vld1q_u32(B[0]);
        C0 = vmlaq_laneq_u32(C0, A0, B0, 0);
        C0 = vmlaq_laneq_u32(C0, A1, B0, 1);
        C0 = vmlaq_laneq_u32(C0, A2, B0, 2);
        C0 = vmlaq_laneq_u32(C0, A3, B0, 3);
        vst1q_u32(C[0], C0);
        
        B1 = vld1q_u32(B[1]);
        C1 = vmlaq_laneq_u32(C1, A0, B1, 0);
        C1 = vmlaq_laneq_u32(C1, A1, B1, 1);
        C1 = vmlaq_laneq_u32(C1, A2, B1, 2);
        C1 = vmlaq_laneq_u32(C1, A3, B1, 3);
        vst1q_u32(C[1], C1);
        
        B2 = vld1q_u32(B[2]);
        C2 = vmlaq_laneq_u32(C2, A0, B2, 0);
        C2 = vmlaq_laneq_u32(C2, A1, B2, 1);
        C2 = vmlaq_laneq_u32(C2, A2, B2, 2);
        C2 = vmlaq_laneq_u32(C2, A3, B2, 3);
        vst1q_u32(C[2], C2);
        
        B3 = vld1q_u32(B[3]);
        C3 = vmlaq_laneq_u32(C3, A0, B3, 0);
        C3 = vmlaq_laneq_u32(C3, A1, B3, 1);
        C3 = vmlaq_laneq_u32(C3, A2, B3, 2);
        C3 = vmlaq_laneq_u32(C3, A3, B3, 3);
        vst1q_u32(C[3], C3);
}


int main(){
	uint32_t A[4][4] = {{ 74 , 45 , 13 , 57 },
            { 16 , 12 , 48 , 75 },
            { 90 , 64 , 12 , 45 },
            { 35 , 12 , 90 , 95 }};
        uint32_t B[4][4] = {{ 74 , 45 , 13 , 57 },
            { 16 , 12 , 48 , 75 },
            { 90 , 64 , 12 , 45 },
            { 35 , 12 , 90 , 95 }};
        uint32_t C[4][4] = {{ 0 , 0 , 0 , 0 },
            { 0 , 0 , 0 , 0 },
            {0 , 0 , 0 , 0 },
            { 0, 0 , 0 , 0 }};
        uint32_t n = BLOCK_SIZE; // rows in A
        uint32_t m = BLOCK_SIZE; // cols in B
        uint32_t k = BLOCK_SIZE; // cols in a and rows in b
        
        matrix_multiply_4x4_neon(A,B,C);
        for ( int x = 0; x < 4; x++)
        {
            for ( int y = 0; y < 4; y++)
            {
                cout << "\t";
                cout << C[x][y];
            }
            cout << endl;
        }
	return 0;
}


