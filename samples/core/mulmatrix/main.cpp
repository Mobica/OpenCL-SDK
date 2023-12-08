#define __CL_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <iostream>
#include <fstream>


std::string kernelMulMatrix =
    "__kernel void mulMatrix(                 \n"
    "   const int N,                                                        \n"
    "   int gpu,                                                            \n"
    "   __global int* A,                                                    \n"
    "   __global int* B,                                                    \n"
    "   __global int* C)                                                    \n"
    "{                                                                      \n"
    "   int k = 0;                                                          \n"
    "   int element = get_global_id(0);                                     \n"
    "   int i = element + N/2 * gpu;                                        \n"
    "   int j = get_global_id(1);                                           \n"
    "   int tmp;                                                            \n"
    "   if ( (i < N) && (j <N) )                                            \n"
    "   {                                                                   \n"
    "       tmp = 0;                                                        \n"
    "       for(k;k<N;k++)                                                  \n"
    "       {                                                               \n"
    "           tmp += A[element*N+k] * B[k*N+j];                           \n"
    "       }                                                               \n"
    "       C[i*N+j] = tmp;                                                 \n"
    "   }                                                                   \n"
    "}                                                                      \n"
    "\n";

class MatrixMultiplication {
public:
    MatrixMultiplication(cl::Platform& platform);//, std::string& kernelSource);
    MatrixMultiplication(std::vector<cl::Platform>& platforms);//,
                         //std::string& kernelSource);
    ~MatrixMultiplication() = default;

    void Multiply();

private:
    void PrepareMatrices();
    void CreateContextsAndCommandQueues();
    void CreatePrograms();
    void MergeData();
    void CheckResults();

    bool LoadKernel();

    std::string mKernelSource;
    const int mMatrixDimension = 4 * 1024;
    std::vector<int> mMatrixA;
    std::vector<int> mMatrixB;
    std::vector<int> mMatrixC;
    std::vector<int> mMatrixD;

    std::vector<cl::Program> mPrograms;
    std::vector<cl::Context> mContexts;
    std::vector<cl::CommandQueue> mCommandQueues;
    std::vector<cl::Platform> mPlatforms;
};

MatrixMultiplication::MatrixMultiplication(cl::Platform& platform)//,
                                           //std::string& kernelSource)
//    : mKernelSource(kernelSource)
{
    if (!LoadKernel())
    {
        exit(-1);
    }
    mPlatforms.push_back(platform);
    CreateContextsAndCommandQueues();
    CreatePrograms();
    PrepareMatrices();
};

MatrixMultiplication::MatrixMultiplication(std::vector<cl::Platform>& platforms)//,
                                           //std::string& kernelSource)
    : mPlatforms(platforms)//, mKernelSource(kernelSource)
{
    if (!LoadKernel())
    {
        exit(-1);
    }
    CreateContextsAndCommandQueues();
    CreatePrograms();
    PrepareMatrices();
};

void MatrixMultiplication::CreateContextsAndCommandQueues()
{
    for (auto& platform : mPlatforms)
    {
        std::string name;
        platform.getInfo(CL_PLATFORM_NAME, &name);
        std::cout << "Platform: " << name << std::endl;

        std::vector<cl::Device> platformDevices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &platformDevices);
        if (platformDevices.size() > 0)
        {
            cl::Context context(platformDevices);
            mContexts.emplace_back(context);
            mCommandQueues.emplace_back(
                cl::CommandQueue(context, platformDevices[0]));
        }
    }
}

void MatrixMultiplication::CreatePrograms()
{
    for (auto& context : mContexts)
    {
        mPrograms.emplace_back(cl::Program(context, mKernelSource, true));
    }
};

void MatrixMultiplication::Multiply()
{
    std::mutex m;
    std::condition_variable cv;
    std::atomic<int> threads(0);

    auto start = std::chrono::system_clock::now();

    cl::NDRange matrixRange(mMatrixDimension / mContexts.size(),
                            mMatrixDimension);

    for (int gpu = 0; gpu < mContexts.size(); ++gpu)
    {
        auto half = (mMatrixDimension * mMatrixDimension) / mContexts.size();
        auto start = half * gpu;
        auto stop = start + half;

        threads++;
        std::thread work([&, gpu]() {
            cl::compatibility::make_kernel<int, int, cl::Buffer, cl::Buffer,
                                           cl::Buffer>
                kernelFunc(mPrograms[gpu], "mulMatrix");

            cl::Buffer aMatBuffer(mContexts[gpu], mMatrixA.begin() + start,
                                  mMatrixA.begin() + stop, true);
            cl::Buffer bMatBuffer(mContexts[gpu], mMatrixB.begin(),
                                  mMatrixB.end(), true);
            cl::Buffer cMatBuffer(mContexts[gpu], CL_MEM_WRITE_ONLY,
                                  mMatrixDimension * mMatrixDimension
                                      * sizeof(int));
            try
            {
                kernelFunc(cl::EnqueueArgs(mCommandQueues[gpu], matrixRange),
                           mMatrixDimension, gpu, aMatBuffer, bMatBuffer,
                           cMatBuffer);
            } catch (cl::Error err)
            {
                std::cout << "Error 1: " << (int)err.err() << " " << err.what()
                          << std::endl;
                exit(-1);
            }
            try
            {
                mCommandQueues[gpu].finish();
            } catch (cl::Error err)
            {
                std::cout << "Error 2: " << (int)err.err() << " " << err.what()
                          << std::endl;
                exit(-1);
            }


            if (gpu == 0) try
                {
                    cl::copy(mCommandQueues[gpu], cMatBuffer, mMatrixC.begin(),
                             mMatrixC.end());
                } catch (cl::Error err)
                {
                    std::cout << "Error 3: " << (int)err.err() << " "
                              << err.what() << std::endl;
                    exit(-1);
                }

            else
                try
                {
                    cl::copy(mCommandQueues[gpu], cMatBuffer, mMatrixD.begin(),
                             mMatrixD.end());
                } catch (cl::Error err)
                {
                    std::cout << "Error 4: " << (int)err.err() << " "
                              << err.what() << std::endl;
                    exit(-1);
                }



            std::lock_guard<std::mutex> lk(m);
            threads--;
            cv.notify_all();
        });
        work.detach();
    }

    std::unique_lock<std::mutex> lock(m);
    cv.wait(lock, [&]() { return threads == 0; });

    MergeData();

    auto end = std::chrono::system_clock::now();
    auto diff =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Calculation time: " << diff.count()
              << " ms"
              << std::endl;

    CheckResults();
};

void MatrixMultiplication::MergeData()
{
    if (mContexts.size() > 1)
    {
        auto start = std::chrono::system_clock::now();

        std::transform(mMatrixC.begin(), mMatrixC.end(), mMatrixD.begin(),
                       mMatrixC.begin(), std::plus<int>());

        auto end = std::chrono::system_clock::now();
        auto diff =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Merging data time: " << diff.count()
                  << " ms"
                  << std::endl;
    }
};

bool MatrixMultiplication::LoadKernel()
{
    std::string fileName("mulMatrix.cl");
    std::ifstream stream(fileName.c_str());
    if (!stream.is_open())
    {
        std::cout << "Cannot open file: " << fileName << std::endl;
        return false;
    }
    mKernelSource =
        std::move(std::string(std::istreambuf_iterator<char>(stream),
                              (std::istreambuf_iterator<char>())));

    return true;
}

void MatrixMultiplication::PrepareMatrices()
{
    auto size = mMatrixDimension * mMatrixDimension;
    mMatrixA = std::move(std::vector<int>(size, 5));
    mMatrixB = std::move(std::vector<int>(size, 6));
    mMatrixC = std::move(std::vector<int>(size, 0));
    mMatrixD = std::move(std::vector<int>(size, 0));
};

void MatrixMultiplication::CheckResults()
{
    int i, j;
    float cval, errsq, err;
    cval = (float)mMatrixDimension * 5 * 6;
    errsq = 0.0f;

    for (i = 0; i < mMatrixDimension; i++)
    {
        for (j = 0; j < mMatrixDimension; j++)
        {
            err = (mMatrixC[i * mMatrixDimension + j])
                - cval;
            errsq += err * err;
        }
    }
    if (errsq > 0.001)
    {
        std::cout << "Calculation Errors "
                  << " (" << errsq << ")" << std::endl;
    }
    std::cout << "-------------------------------------------------"
              << std::endl;
};

int main()
{
    try
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        if (!platforms.empty())
        {
            MatrixMultiplication nVidiaPlatformMul(platforms[0]);
            nVidiaPlatformMul.Multiply();
        }
        if (platforms.size() > 1)
        {

            MatrixMultiplication intelPlatformMul(platforms[1]);
            intelPlatformMul.Multiply();
            MatrixMultiplication twoPlatformMul(platforms);
            twoPlatformMul.Multiply();
        }

        char exit;
        std::cout << "Type any letter and press enter to exit." << std::endl;
        std::cin >> exit;
    } catch (cl::Error err)
    {
        std::cout << "Error: " << err.what() << std::endl;
        exit(-1);
    }

    return 0;
}