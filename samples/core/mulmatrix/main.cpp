#define __CL_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <iostream>
#include <fstream>
#include <random>


#include <CL/SDK/CLI.hpp>

struct MatricesDimensions
{
     unsigned int fstMtxRows;
     unsigned int fstMtxCols;
     unsigned int sndMtxRows;
     unsigned int sndMtxCols;
 };

struct Matrix
{
     unsigned int mRows;
     unsigned int mCols;
     std::vector<int> mData;

     Matrix(int rows, int cols)
         : mRows(rows), mCols(cols), mData(std::vector<int>(mRows*mCols,0))
     {
     };
     void GenerateData(int from, int to)
     {
         std::random_device randDev;
         std::mt19937 generator(randDev());
         std::uniform_int_distribution<int> distr(from, to);

         auto size = mRows * mCols;
         std::vector<int> data;
         data.reserve(size);
         for (unsigned int i = 0; i < size; ++i)
         {
             data.push_back(distr(generator));
         }
         mData = std::move(data);
     };
};

class MatrixMultiplication {
public:
     MatrixMultiplication(cl::Platform& platform, MatricesDimensions& matDims,
                          Matrix& a, Matrix& b);
    MatrixMultiplication(std::vector<cl::Platform>& platforms,
                          MatricesDimensions& matDims, Matrix& a, Matrix& b);
    ~MatrixMultiplication() = default;

    void Multiply();

private:
    //void PrepareMatrices();
    //std::vector<int> GenerateMatrix(unsigned int size);
    void CreateContextsAndCommandQueues();
    void CreatePrograms();
    void MergeData();
    //void CheckResults();

    bool LoadKernel();

    std::string mKernelSource;

    std::vector<cl::Program> mPrograms;
    std::vector<cl::Context> mContexts;
    std::vector<cl::CommandQueue> mCommandQueues;
    std::vector<cl::Platform> mPlatforms;

    MatricesDimensions mMatDims;
    Matrix& A;
    Matrix& B;
    Matrix result;
    Matrix helper;
};

MatrixMultiplication::MatrixMultiplication(cl::Platform& platform,
                                           MatricesDimensions& matDims,
                                           Matrix& a, Matrix& b)
    : mMatDims(matDims), A(a), B(b), result(Matrix(A.mRows, B.mCols)),
      helper(Matrix(0, 0))
{
    if (!LoadKernel())
    {
        exit(-1);
    }
    mPlatforms.push_back(platform);
    CreateContextsAndCommandQueues();
    CreatePrograms();
    //PrepareMatrices();
};

MatrixMultiplication::MatrixMultiplication(std::vector<cl::Platform>& platforms,
                                           MatricesDimensions& matDims,
                                           Matrix& a, Matrix& b)
    : mPlatforms(platforms), mMatDims(matDims), A(a), B(b), result(Matrix(A.mRows, B.mCols)), helper(Matrix(0, 0))
{
    if (!LoadKernel())
    {
        exit(-1);
    }
    CreateContextsAndCommandQueues();
    CreatePrograms();
    //PrepareMatrices();
};

void MatrixMultiplication::CreateContextsAndCommandQueues()
{
    for (auto& platform : mPlatforms)
    {
        std::string name;
        platform.getInfo(CL_PLATFORM_NAME, &name);
        std::cout << "Platform: " << name << std::endl;

        std::vector<cl::Device> platformDevices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &platformDevices);//CL_DEVICE_TYPE_ALL
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

    int GPUs = static_cast<int>(mContexts.size());
    if (GPUs > 2)
    {
        std::cout << "Maximum number of supported GPUs for usage is 2"
                  << std::endl;
        GPUs = 2;
    }
    if (mMatDims.fstMtxRows == 1 && GPUs > 1)
    {
        std::cout
            << "First matrix has only 1 row, nothing to divide on two GPUs"
            << std::endl;
        GPUs = 1;
    }
    auto start = std::chrono::system_clock::now();


    cl::NDRange matrixRange[2];
    matrixRange[0] = cl::NDRange(mMatDims.fstMtxRows / GPUs, mMatDims.sndMtxCols);
    matrixRange[1] = cl::NDRange(mMatDims.fstMtxRows / GPUs, mMatDims.sndMtxCols);
    if (mMatDims.fstMtxRows % GPUs == 1)
    {
        matrixRange[1] =
            cl::NDRange((mMatDims.fstMtxRows + 1) / GPUs, mMatDims.sndMtxCols);
    }
    
    int from[2];
    int to[2];
    from[0] = 0;
    to[0] = (mMatDims.fstMtxRows * mMatDims.fstMtxCols) / GPUs;
    if (mMatDims.fstMtxRows % GPUs == 1)
    {
        to[0] = ((mMatDims.fstMtxRows - 1) * mMatDims.fstMtxCols) / GPUs;
    }
    from[1] = to[0];
    to[1] = mMatDims.fstMtxRows * mMatDims.fstMtxCols;

    for (int gpu = 0; gpu < GPUs; ++gpu)
    {
        auto start = from[gpu];
        auto stop = to[gpu];

        threads++;
        std::thread work([&, gpu, start, stop]() 
        {
            cl::KernelFunctor<int, int, int, int, cl::Buffer, cl::Buffer, cl::Buffer>
                kernelFunc(mPrograms[gpu], "mulMatrix");

            cl::Buffer aMatBuffer(mContexts[gpu], A.mData.begin() + start,
                                  A.mData.begin() + stop, true);
            cl::Buffer bMatBuffer(mContexts[gpu], B.mData.begin(),
                                  B.mData.end(), true);
            cl::Buffer resultMatBuffer(mContexts[gpu], CL_MEM_WRITE_ONLY,
                                       mMatDims.fstMtxRows * mMatDims.sndMtxCols * sizeof(int));
            try
            {
                kernelFunc(cl::EnqueueArgs(mCommandQueues[gpu], matrixRange[gpu]),
                           mMatDims.fstMtxRows, mMatDims.fstMtxCols,
                           mMatDims.sndMtxCols, gpu, aMatBuffer, bMatBuffer,
                           resultMatBuffer);
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
                    cl::copy(mCommandQueues[gpu], resultMatBuffer,
                             result.mData.begin(),
                             result.mData.end());
                } catch (cl::Error err)
                {
                    std::cout << "Error 3: " << (int)err.err() << " "
                              << err.what() << std::endl;
                    exit(-1);
                }

            else
                try
                {
                    helper = std::move(Matrix(result.mRows, result.mCols));

                    cl::copy(mCommandQueues[gpu], resultMatBuffer,
                             helper.mData.begin(),
                             helper.mData.end());
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

    if (GPUs > 1)
    {
        MergeData();
    }

    auto end = std::chrono::system_clock::now();
    auto diff =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Calculation time: " << diff.count()
              << " ms"
              << std::endl;

    //CheckResults();
    std::cout << "-------------------------------------------------"
              << std::endl;
};

void MatrixMultiplication::MergeData()
{
    auto start = std::chrono::system_clock::now();

    std::transform(result.mData.begin(), result.mData.end(), helper.mData.begin(),
                   result.mData.begin(), std::plus<int>());

    auto end = std::chrono::system_clock::now();
    auto diff =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Merging data time: " << diff.count()
                << " ms"
                << std::endl;
};

bool MatrixMultiplication::LoadKernel()
{
    //std::string fileName("C:\\RND200\\OpenCL-SDK\\samples\\core\\mulmatrix\\mulMatrix.cl");
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

//std::vector<int> MatrixMultiplication::GenerateMatrix(unsigned int size)
//{
//    const int rangeFrom = -100;
//    const int rangeTo = 100;
//    std::random_device randDev;
//    std::mt19937 generator(randDev());
//    std::uniform_int_distribution<int> distr(rangeFrom, rangeTo);
//
//    std::vector<int> data;
//    data.reserve(size);
//    for (unsigned int i = 0; i < size; ++i)
//    {
//        data.push_back(distr(generator));
//    }
//
//    return data;
//};

//void MatrixMultiplication::PrepareMatrices()
//{
//    unsigned int size = mMatDims.fstMtxRows * mMatDims.fstMtxCols;
//    mMatrixA = std::move(GenerateMatrix(size)); //std::move(std::vector<int>(size, 5));
//    size = mMatDims.sndMtxRows * mMatDims.sndMtxCols;
//    mMatrixB = std::move(GenerateMatrix(size)); // std::move(std::vector<int>(size, 6));
//    size = mMatDims.fstMtxRows * mMatDims.sndMtxCols;
//    mMatrixC = std::move(std::vector<int>(size, 0));
//    mMatrixD = std::move(std::vector<int>(size, 0));
//};

//void MatrixMultiplication::CheckResults()
//{
//    unsigned int i, j;
//    float cval, errsq, err;
//    cval = (float)mMatDims.fstMtxCols * 5 * 6;
//    errsq = 0.0f;
//
//    for (i = 0; i < mMatDims.fstMtxRows; i++)
//    {
//        for (j = 0; j < mMatDims.sndMtxCols; j++)
//        {
//            err = (mMatrixC[i * mMatDims.sndMtxCols + j])
//                - cval;
//            errsq += err * err;
//        }
//    }
//    if (errsq > 0.001)
//    {
//        std::cout << "Calculation Errors "
//                  << " (" << errsq << ")" << std::endl;
//    }
//};

template <> auto cl::sdk::parse<MatricesDimensions>()
{
    return std::make_tuple(
        std::make_shared<TCLAP::ValueArg<unsigned int>>(
                               "p", "cols2", "Second matrix columns number (default 4096)", false,
           4096, "positive integral"),
        std::make_shared<TCLAP::ValueArg<unsigned int>>(
                               "n", "cols1", "First matrix columns number (default 4096)", false,
           4096, "positive integral"),
        std::make_shared<TCLAP::ValueArg<unsigned int>>(
                               "m", "rows1", "First matrix rows number (default 4096)", false,
           4096, "positive integral"));
}
template <>
MatricesDimensions cl::sdk::comprehend<MatricesDimensions>(
    std::shared_ptr<TCLAP::ValueArg<unsigned int>> sndMtxCols ,
    std::shared_ptr<TCLAP::ValueArg<unsigned int>> fstMtxCols,//2nd matrix rows must be the same as fst matrix column
    std::shared_ptr<TCLAP::ValueArg<unsigned int>> fstMtxRows)
{
    return MatricesDimensions{fstMtxRows->getValue(),
                          fstMtxCols->getValue(),
                          fstMtxCols->getValue(),//2nd matrix rows must be the same as fst matrix column
                          sndMtxCols->getValue() };
}

int main(int argc, char* argv[])
{
    auto opts = cl::sdk::parse_cli<MatricesDimensions>(argc, argv, "OpenCL SDK Matrices multiplication example");
    MatricesDimensions matDims = std::get<0>(opts);
    
    unsigned int rows1 = matDims.fstMtxRows;
    unsigned int cols1 = matDims.fstMtxCols;
    unsigned int rows2 = matDims.sndMtxRows;
    unsigned int cols2 = matDims.sndMtxCols;
    std::cout << "Matrices given for multiplication: A[" << rows1 << "x" << cols1 << "] x B["
              << rows2 << "x" << cols2 << "]" << std::endl;
    std::cout << "-------------------------------------------------"
              << std::endl;

    try
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        Matrix a(rows1, cols1);
        a.GenerateData(-100, 100);
        Matrix b(rows2, cols2);
        b.GenerateData(-100, 100);

        if (!platforms.empty())
        {
            MatrixMultiplication singlePlatformMul(platforms[0], matDims, a, b);
            singlePlatformMul.Multiply();
        }
        if (platforms.size() > 1)
        {

            MatrixMultiplication singlePlatformMul(platforms[1], matDims, a, b);
            singlePlatformMul.Multiply();
            MatrixMultiplication multiPlatformMul(platforms, matDims, a, b);
            multiPlatformMul.Multiply();
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