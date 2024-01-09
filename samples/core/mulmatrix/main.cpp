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

const unsigned int MAX_DIM = 4096;
const int FROM = -100;
const int TO = 100;

struct SampleOptions
{
     unsigned int fstMtxRows;
     unsigned int fstMtxCols;
     unsigned int sndMtxRows;
     unsigned int sndMtxCols;
     int from;
     int to;
 };

struct Matrix
{
     unsigned int mRows;
     unsigned int mCols;
     std::vector<int> mData;

     Matrix(int rows, int cols)
         : mRows(rows), mCols(cols),
           mData(std::vector<int>(mRows*mCols,0))
     {
     };
     Matrix(int rows, int cols, int from, int to)
         : mRows(rows), mCols(cols),
           mData(GenerateData(from, to))
     {
     };
     std::vector<int> GenerateData(int from, int to)
     {
         if (from >= to)
         {
             std::cout << "Passed range <" << from << "," << to
                       << "> to generate matrix data is invalid.";
             exit(-1);
         }
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
         return data;
     };
};

class MatrixMultiplication {
public:
    MatrixMultiplication(cl::Platform& platform,
                         Matrix& a, Matrix& b);
    MatrixMultiplication(std::vector<cl::Platform>& platforms,
                         Matrix& a, Matrix& b);
    ~MatrixMultiplication() = default;

    void Multiply();
    void ExportToCSV(const std::string& filename);

private:
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

    Matrix& A;
    Matrix& B;
    Matrix result;
    Matrix helper;
};

MatrixMultiplication::MatrixMultiplication(cl::Platform& platform,
                                           Matrix& a, Matrix& b)
    : A(a), B(b), result(Matrix(A.mRows, B.mCols)),
      helper(Matrix(0, 0))
{
    if (!LoadKernel())
    {
        exit(-1);
    }
    mPlatforms.push_back(platform);
    CreateContextsAndCommandQueues();
    CreatePrograms();
};

MatrixMultiplication::MatrixMultiplication(std::vector<cl::Platform>& platforms,
                                           Matrix& a, Matrix& b)
    : mPlatforms(platforms), A(a), B(b),
      result(Matrix(A.mRows, B.mCols)), helper(Matrix(0, 0))
{
    if (!LoadKernel())
    {
        exit(-1);
    }
    CreateContextsAndCommandQueues();
    CreatePrograms();
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
        try
        {
            cl::Program program(context, mKernelSource, true);
            program.build();
            mPrograms.push_back(program);
        }
        catch (cl::BuildError& error)
        {
            std::cout << "Build program error " << error.err() << " " << error.what() << std::endl;
            exit(-1);
        }
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
    if (A.mRows == 1 && GPUs > 1)
    {
        std::cout
            << "First matrix has only 1 row, nothing to divide on two GPUs"
            << std::endl;
        GPUs = 1;
    }
    auto start = std::chrono::system_clock::now();


    cl::NDRange matrixRange[2];

    matrixRange[0] = cl::NDRange(A.mRows / GPUs, B.mCols);
    matrixRange[1] = cl::NDRange(A.mRows / GPUs, B.mCols);
    if (A.mRows % GPUs == 1)
    {
        matrixRange[1] = cl::NDRange((A.mRows + 1) / GPUs, B.mCols);
    }
    
    int from[2];
    int to[2];
    from[0] = 0;
    to[0] = (A.mRows * A.mCols) / GPUs;
    if (A.mRows % GPUs == 1)
    {
        to[0] = ((A.mRows - 1) * A.mCols) / GPUs;
    }
    from[1] = to[0];
    to[1] = A.mRows * A.mCols;

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
            
            auto size = A.mRows * B.mCols * sizeof(int);
            cl::Buffer resultMatBuffer(mContexts[gpu], CL_MEM_WRITE_ONLY, size);
            mCommandQueues[gpu].enqueueFillBuffer(resultMatBuffer, 0, 0, size, nullptr);

            kernelFunc(cl::EnqueueArgs(mCommandQueues[gpu], matrixRange[gpu]),
                       A.mRows, A.mCols, B.mCols, gpu, aMatBuffer, bMatBuffer, resultMatBuffer);
            
            mCommandQueues[gpu].finish();
            
            if (gpu == 0)
            {
                cl::copy(mCommandQueues[gpu], resultMatBuffer,
                         result.mData.begin(), result.mData.end());
            }
            else if (gpu == 1)
            {
                helper = std::move(Matrix(result.mRows, result.mCols));

                cl::copy(mCommandQueues[gpu], resultMatBuffer,
                            helper.mData.begin(),
                            helper.mData.end());
            }

            std::lock_guard<std::mutex> lock(m);
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
    std::string fileName("C:\\RND200\\OpenCL-SDK\\samples\\core\\mulmatrix\\mulMatrix.cl");
    //std::string fileName("mulMatrix.cl");
    std::ifstream stream(fileName.c_str());
    if (!stream.is_open())
    {
        fileName = ("mulMatrix.cl");
        //std::cout << "Cannot open file: " << fileName << std::endl;
        //return false;
    }
    mKernelSource =
        std::move(std::string(std::istreambuf_iterator<char>(stream),
                              (std::istreambuf_iterator<char>())));

    return true;
};

void MatrixMultiplication::ExportToCSV(const std::string& filename)
{
    std::ofstream resultFile;
    resultFile.open(filename, std::ios::out | std::ios::trunc);
    if (resultFile.is_open())
    {
        std::cout << "Exporting results to CSV file " << filename << std::endl;
        std::cout << "-------------------------------------------------"
                  << std::endl << std::endl;

        resultFile << "Matrix A [" << A.mRows << " x " << A.mCols << "]\n\n ";

        auto rows = A.mRows;
        auto cols = A.mCols;
        for (unsigned int r = 0; r < rows; ++r)
        {
            for (unsigned int c = 0; c < cols; ++c)
            {
                resultFile << A.mData[r * cols + c] << ",";
            }
            resultFile << "\n";
        }

        resultFile << "\nMatrix B [" << B.mRows << " x " << B.mCols << "]\n\n";
        rows = B.mRows;
        cols = B.mCols;
        for (unsigned int r = 0; r < rows; ++r)
        {
            for (unsigned int c = 0; c < cols; ++c)
            {
                resultFile << B.mData[r * cols + c] << ",";
            }
            resultFile << "\n";
        }

        resultFile << "\nMultiplication of matrices A x B [" << result.mRows << " x " << result.mCols << "]\n\n";
        rows = result.mRows;
        cols = result.mCols;
        for (unsigned int r = 0; r < rows; ++r)
        {
            for (unsigned int c = 0; c < cols; ++c)
            {
                resultFile << result.mData[r * cols + c] << ",";
            }
            resultFile << "\n";
        }
        resultFile.close();
    }
};


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

template <> auto cl::sdk::parse<SampleOptions>()
{
    return std::make_tuple(
        std::make_shared<TCLAP::ValueArg<int>>(
            "t", "to",
            "Upper range to generate matrix data (maximum 100 default 100)", false, TO, "integral"),
        std::make_shared<TCLAP::ValueArg<int>>(
            "f", "from",
            "Bottom range to generate matrix data (minimum -100 default -100)", false, FROM, "integral"),
        std::make_shared<TCLAP::ValueArg<unsigned int>>(
            "p", "cols2", "Second matrix columns number (maximum 4096 default 4096)", false,
            MAX_DIM, "positive integral"),
        std::make_shared<TCLAP::ValueArg<unsigned int>>(
            "n", "cols1", "First matrix columns number (maximum 4096 default 4096)", false,
            MAX_DIM, "positive integral"),
        std::make_shared<TCLAP::ValueArg<unsigned int>>(
            "m", "rows1", "First matrix rows number (maximum 4096 default 4096)", false,
            MAX_DIM, "positive integral"));
}
template <>
SampleOptions cl::sdk::comprehend<SampleOptions>(
    std::shared_ptr<TCLAP::ValueArg<int>> to,
    std::shared_ptr<TCLAP::ValueArg<int>> from,
    std::shared_ptr<TCLAP::ValueArg<unsigned int>> sndMtxCols ,
    std::shared_ptr<TCLAP::ValueArg<unsigned int>> fstMtxCols,//second matrix rows number must be equal to first matrix column number
    std::shared_ptr<TCLAP::ValueArg<unsigned int>> fstMtxRows)
{
    auto rows1 = fstMtxRows->getValue();
    auto cols1 = fstMtxCols->getValue();
    auto cols2 = sndMtxCols->getValue();

    if (rows1 > MAX_DIM)
    {
        std::cout << rows1 << " is greater than maximum value. " << MAX_DIM << " will be used." << std::endl;
        rows1 = MAX_DIM;
    }
    if (cols1 > MAX_DIM)
    {
        std::cout << cols1 << " is greater than maximum value. " << MAX_DIM << " will be used." << std::endl;
        cols1 = MAX_DIM;
    }
    if (cols2 > MAX_DIM)
    {
        std::cout << cols2 << " is greater than maximum value. " << MAX_DIM << " will be used." << std::endl;
        cols2 = MAX_DIM;
    }

    auto bottom = from->getValue();
    auto upper = to->getValue();
    if (bottom < FROM)
    {
        std::cout << bottom << " is smaller than minimum value. " << FROM
                  << " will be used." << std::endl;
        bottom = FROM;
    }
    if (upper > TO)
    {
        std::cout << upper << " is greater than maximum value. " << TO
                  << " will be used." << std::endl;
        upper = TO;
    }

    return SampleOptions{
        rows1,
        cols1,
        cols1, // second matrix rows number must be equal to first matrix column number
        cols2,
        bottom,
        upper};
}

int main(int argc, char* argv[])
{
    auto cmdline = cl::sdk::parse_cli<SampleOptions>(
        argc, argv, "OpenCL SDK Matrices multiplication example");

    SampleOptions options = std::get<0>(cmdline);
    
    unsigned int rows1 = options.fstMtxRows;
    unsigned int cols1 = options.fstMtxCols;
    unsigned int rows2 = options.sndMtxRows;
    unsigned int cols2 = options.sndMtxCols;

    std::cout << "Matrices given for multiplication: A[" << rows1 << "x" << cols1 << "] x B["
              << rows2 << "x" << cols2 << "]" << std::endl;
    std::cout << "-------------------------------------------------"
              << std::endl;

    try
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        Matrix a(rows1, cols1, options.from, options.to);
        Matrix b(rows2, cols2, options.from, options.to);

        if (!platforms.empty())
        {
            MatrixMultiplication singlePlatformMul(platforms[0], a, b);
            singlePlatformMul.Multiply();
            singlePlatformMul.ExportToCSV("mulmatrix_first_platform.csv");
        }
        if (platforms.size() > 1)
        {

            MatrixMultiplication singlePlatformMul(platforms[1], a, b);
            singlePlatformMul.Multiply();
            singlePlatformMul.ExportToCSV("mulmatrix_second_platform.csv");
            MatrixMultiplication multiPlatformMul(platforms, a, b);
            multiPlatformMul.Multiply();
            multiPlatformMul.ExportToCSV("mulmatrix_two_platforms.csv");
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