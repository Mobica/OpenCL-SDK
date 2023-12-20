kernel void mulMatrix(
    const unsigned int M, //first matrix rows number
	const unsigned int N, //first matrix columns number
	const unsigned int P, //second matrix columns number
    int gpu,
    global int* A,
    global int* B,
    global int* C
)
{
    unsigned int k = 0;
    unsigned int element = get_global_id(0);
    unsigned int i = element + M/2 * gpu;
    unsigned int j = get_global_id(1);
    int tmp;
    {
        tmp = 0;
        for(k;k<N;k++)
        {
			tmp += A[element*N + k] * B[k*P + j];
        }
        C[i*P+j] = tmp;
    }
}
