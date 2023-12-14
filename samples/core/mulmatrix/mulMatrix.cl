kernel void mulMatrix(
    const int N, //first matrix rows number
	const int M, //first matrix columns number
	const int P, //second matrix columns number
    int gpu,
    global int* A,
    global int* B,
    global int* C
)
{
    int k = 0;
    int element = get_global_id(0);
    int i = element + N/2 * gpu;
    int j = get_global_id(1);
    int tmp;
    {
        tmp = 0;
        for(k;k<M;k++)
        {
			tmp += A[element*N+k] * B[k*P +j];
        }
        C[i*N+j] = tmp;
    }
}
