kernel void mulMatrix(
    const int N,
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
    if ( (i < N) && (j <N) )
    {
        tmp = 0;
        for(k;k<N;k++)
        {
            tmp += A[element*N+k] * B[k*N+j];
        }
        C[i*N+j] = tmp;
    }
}
