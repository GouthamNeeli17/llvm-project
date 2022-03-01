//Multiplication of 2 matrices resulting in a matrix of n*4;


#include<iostream>
#include<arm_sve.h>

#define BLOCK_SIZE 4
using namespace std;


void matrix_multiply_nx4_sve(const uint32_t *A, const uint32_t *B, uint32_t *C,
 uint32_t n) {
 // for columns in A
 svuint32_t A0;
 svuint32_t A1;
 svuint32_t A2;
 svuint32_t A3;
 // for columns in B
 svuint32_t B0;
 svuint32_t B1;
 svuint32_t B2;
 svuint32_t B3;
 // for columns in C
 svuint32_t C0;
 svuint32_t C1;
 svuint32_t C2;
 svuint32_t C3;
 svbool_t pred = svwhilelt_b32_u32(0, n); // This specifies a predicate vector register. svwhilelt => while(0<n) elements are active , here as n is greater than 0 all elements are active;
 					// _b32 specifies number of bits in each predicate element here 32 bits(this means each it will control 32 bit of data; 
 					//so a suffix of _b16 indicates that the predicate controls 16-bit data) ; _u32 specifies data type of the parameter.
 A0 = svld1_u32(pred, A);      // load unsigned 32bit int values starting from address of first value of A, until the pred values are active(1) that is till n values
 A1 = svld1_u32(pred, A+n);	// load next n values
 A2 = svld1_u32(pred, A+2*n); 
 A3 = svld1_u32(pred, A+3*n);  //and so on till all 4 columns are loaded into vector regs.
 // Accumulate zeroes for C
 C0 = svdup_n_u32(0); // zeroes to first column of C
 C1 = svdup_n_u32(0);
 C2 = svdup_n_u32(0);
 C3 = svdup_n_u32(0);
 // Multiply and accumulate in each column of C
 B0 = svld1rq_u32(svptrue_b32(), B); // loads 128 bit of data. That is 32bit of data from each column, here svptrue_b32 is a predicate vector where all elements are active and hence 4 elements of 						32bits loads to B0
 C0 = svmla_lane_u32(C0, A0, B0, 0); // multiplies each value A0 and B0 and adds to the 1st lane value of C0
 C0 = svmla_lane_u32(C0, A1, B0, 1);
 C0 = svmla_lane_u32(C0, A2, B0, 2); // multiplies each value of A2 and B0 and adds to the 3rd lane of of C0
 C0 = svmla_lane_u32(C0, A3, B0, 3);
 svst1_u32(pred, C, C0); // stores C0 in the first column of C
 // for column No.2
 B1 = svld1rq_u32(svptrue_b32(), B+4);
 C1 = svmla_lane_u32(C1, A0, B1, 0);
 C1 = svmla_lane_u32(C1, A1, B1, 1);
 C1 = svmla_lane_u32(C1, A2, B1, 2);
 C1 = svmla_lane_u32(C1, A3, B1, 3);
 svst1_u32(pred, C+n, C1);
 // for column No.3
 B2 = svld1rq_u32(svptrue_b32(), B+8);
 C2 = svmla_lane_u32(C2, A0, B2, 0);
 C2 = svmla_lane_u32(C2, A1, B2, 1);
 C2 = svmla_lane_u32(C2, A2, B2, 2);
 C2 = svmla_lane_u32(C2, A3, B2, 3);
 svst1_u32(pred, C+2*n, C2);
 // for column No.4
 B3 = svld1rq_u32(svptrue_b32(), B+12);
 C3 = svmla_lane_u32(C3, A0, B3, 0);
 C3 = svmla_lane_u32(C3, A1, B3, 1);
 C3 = svmla_lane_u32(C3, A2, B3, 2);
 C3 = svmla_lane_u32(C3, A3, B3, 3);
 svst1_u32(pred, C+3*n, C3);
}

void print_matrix(uint32_t *M, uint32_t cols, uint32_t rows) {
        for (int i=0; i<rows; i++) {
                for (int j=0; j<cols; j++) {
                       cout<<M[j*rows + i]<<" ";
                }
                cout<<endl;
        }
        cout<<endl;
}

int main(){

	uint32_t n = 2*BLOCK_SIZE; // rows in A
	uint32_t m = BLOCK_SIZE; // cols in B
	uint32_t k = BLOCK_SIZE; // cols in a and rows in b
        
	uint32_t A[n*k] = {1, 2, 3, 4,
	 5, 6, 7, 8,
	 9, 10, 11, 12,
	 13, 14, 15, 16, 
	 17, 18, 19, 20, 
	 21, 22, 23, 24, 
	 25, 26, 27, 28, 
	 29, 30, 31, 32};
        uint32_t B[k*m] = {33, 34, 35, 36, 
        37, 38, 39, 40, 
        41, 42, 43, 44, 
        45, 46, 47, 48, 
        49, 50, 51, 52, 
        53, 54, 55, 56, 
        57, 58, 59, 60, 
        61, 62, 63, 64};
        uint32_t C[n*m];     
        matrix_multiply_nx4_sve(A,B,C,n);
        print_matrix(C, n, m);
	return 0;
}

