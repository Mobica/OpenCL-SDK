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

template <typename... Args>
inline std::string str_sprintf(const std::string& str, Args... args)
{
    int str_size = std::snprintf(nullptr, 0, str.c_str(), args...) + 1;
    if (str_size <= 0) throw std::runtime_error("Formatting error.");
    size_t s = static_cast<size_t>(str_size);
    std::unique_ptr<char[]> buffer(new char[s]);
    std::snprintf(buffer.get(), s, str.c_str(), args...);
    return std::string(buffer.get(), buffer.get() + s - 1);
}

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
     unsigned int type;
 };

struct IMatrix
{
     IMatrix() = default;
     virtual ~IMatrix() = default;
};

template<class type>
struct Matrix : public IMatrix
{
     template <typename t>
     using uniform_distribution = typename std::conditional<
         std::is_floating_point<t>::value, std::uniform_real_distribution<t>,
         typename std::conditional<std::is_integral<t>::value,
                                   std::uniform_int_distribution<t>,
                                   void>::type>::type;

     unsigned int mRows;
     unsigned int mCols;
     std::vector<type> mData;

     Matrix(int rows, int cols)
         : mRows(rows), mCols(cols),
           mData(std::vector<type>(mRows*mCols,static_cast<type>(0)))
     {
     };
     Matrix(int rows, int cols, int from, int to)
         : mRows(rows), mCols(cols),
           mData(GenerateData(from, to))
     {
     };
     
     std::vector<type> GenerateData(int from, int to)
     {
         if (from >= to)
         {
             std::cout << "Passed range <" << from << "," << to
                       << "> to generate matrix data is invalid.";
             exit(-1);
         }
         std::random_device randDev;
         std::mt19937 generator(randDev());

         uniform_distribution<type> distr(from, to);

         auto size = mRows * mCols;
         std::vector<type> data;
         data.reserve(size);
         for (unsigned int i = 0; i < size; ++i)
         {
             data.push_back(distr(generator));
         }
         std::cout << "Matrix random data generated" << std::endl;
         return data;
     };
};

class IMatrixMultiplication
{
public:
     IMatrixMultiplication() = default;
     virtual ~IMatrixMultiplication() = default;

     virtual void Multiply() = 0;
     virtual void ExportToCSV(const std::string& filename) = 0;
};

template<typename type>
class MatrixMultiplication : public IMatrixMultiplication
{
public:
    MatrixMultiplication(cl::Platform& platform, std::shared_ptr<IMatrix> a,
                         std::shared_ptr<IMatrix> b);
    MatrixMultiplication(std::vector<cl::Platform>& platforms, std::shared_ptr<IMatrix> a,
                         std::shared_ptr<IMatrix> b);
    ~MatrixMultiplication() = default;

    void Multiply() override;
    void ExportToCSV(const std::string& filename) override;

private:
    void CreateContextsAndCommandQueues();
    void CreatePrograms();
    void CreateKernels();
    void MergeData();

    bool LoadKernel();

    std::string mKernelSource;
    const std::string mKernelName = "mulMatrix";

    std::vector<cl::Program> mPrograms;
    std::vector<cl::Context> mContexts;
    std::vector<cl::CommandQueue> mCommandQueues;
    std::vector<cl::Platform> mPlatforms;
    std::vector<cl::Kernel> mKernels;

    std::shared_ptr<IMatrix> A;
    std::shared_ptr<IMatrix> B;
    std::shared_ptr<IMatrix> result;
    std::shared_ptr<IMatrix> helper;
};

template <typename type>
MatrixMultiplication<type>::MatrixMultiplication(cl::Platform& platform,
                                                 std::shared_ptr<IMatrix> a,
                                                 std::shared_ptr<IMatrix> b)
    : A(a), B(b),
      result(new Matrix<type>(std::dynamic_pointer_cast<Matrix<type>>(A)->mRows,
                              std::dynamic_pointer_cast<Matrix<type>>(B)->mCols))
{
    if (!LoadKernel())
    {
        exit(-1);
    }
    mPlatforms.push_back(platform);
    CreateContextsAndCommandQueues();
    CreatePrograms();
    CreateKernels();
};

template <typename type>
MatrixMultiplication<type>::MatrixMultiplication(
    std::vector<cl::Platform>& platforms, std::shared_ptr<IMatrix> a, std::shared_ptr<IMatrix> b)
    : mPlatforms(platforms), A(a), B(b),
      result(new Matrix<type>(std::dynamic_pointer_cast<Matrix<type>>(A)->mRows,
                              std::dynamic_pointer_cast<Matrix<type>>(B)->mCols))
{
    if (!LoadKernel())
    {
        exit(-1);
    }
    CreateContextsAndCommandQueues();
    CreatePrograms();
    CreateKernels();
};

template <typename type>
void MatrixMultiplication<type>::CreateContextsAndCommandQueues()
{
    for (auto& platform : mPlatforms)
    {
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

template <typename type>
void MatrixMultiplication<type>::CreatePrograms()
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

template <typename type>
void MatrixMultiplication<type>::CreateKernels()
{
    for (auto& program : mPrograms)
    {
        try
        {
            cl::Kernel kernel(program, mKernelName.c_str());
            mKernels.push_back(kernel);
        }
        catch (cl::Error& error)
        {
            std::cout << "Create kernel error " << error.err() << " "
                      << error.what() << std::endl;
            exit(-1);
        }
    }
};

template <typename type>
void MatrixMultiplication<type>::Multiply()
{
    std::mutex m;
    std::condition_variable cv;
    std::atomic<int> threads(0);

    auto aMatrix = std::dynamic_pointer_cast<Matrix<type>>(A);
    auto bMatrix = std::dynamic_pointer_cast<Matrix<type>>(B);

    int GPUs = static_cast<int>(mContexts.size());
    if (GPUs > 2)
    {
        std::cout << "Maximum number of supported GPUs for usage is 2"
                  << std::endl;
        GPUs = 2;
    }
    if (aMatrix->mRows == 1 && GPUs > 1)
    {
        std::cout
            << "First matrix has only 1 row, nothing to divide on two GPUs"
            << std::endl;
        GPUs = 1;
    }

    auto start = std::chrono::system_clock::now();

    cl::NDRange matrixRange[2];

    matrixRange[0] = cl::NDRange(aMatrix->mRows / GPUs, bMatrix->mCols);
    matrixRange[1] = cl::NDRange(aMatrix->mRows / GPUs, bMatrix->mCols);
    if (aMatrix->mRows % GPUs == 1)
    {
        matrixRange[1] = cl::NDRange((aMatrix->mRows + 1) / GPUs, bMatrix->mCols);
    }

    int from[2]{ 0, 0 };
    int to[2]{ 0, 0 };

    to[0] = (aMatrix->mRows * aMatrix->mCols) / GPUs;
    if (aMatrix->mRows % GPUs == 1)
    {
        to[0] = ((aMatrix->mRows - 1) * aMatrix->mCols) / GPUs;
    }
    from[1] = to[0];
    to[1] = aMatrix->mRows * aMatrix->mCols;

    for (int gpu = 0; gpu < GPUs; ++gpu)
    {
        std::string name;
        mPlatforms[gpu].getInfo(CL_PLATFORM_NAME, &name);
        std::cout << "Calculation on platform: " << name << std::endl;

        auto start = from[gpu];
        auto stop = to[gpu];

        threads++;
        std::thread work([&, gpu, start, stop]() 
        {
            cl::Buffer aMatBuffer(mContexts[gpu], aMatrix->mData.begin() + start,
                                  aMatrix->mData.begin() + stop, true);
            cl::Buffer bMatBuffer(mContexts[gpu], bMatrix->mData.begin(),
                                  bMatrix->mData.end(), true);
            
            auto size = aMatrix->mRows * bMatrix->mCols * sizeof(type);
            cl::Buffer resultMatBuffer(mContexts[gpu], CL_MEM_WRITE_ONLY, size);

            mCommandQueues[gpu].enqueueFillBuffer(resultMatBuffer, static_cast<type>(0), 0, size, nullptr);

            cl::KernelFunctor<int, int, int, int, cl::Buffer, cl::Buffer,
                              cl::Buffer>
                kernelFunc(mKernels[gpu]);

            kernelFunc(cl::EnqueueArgs(mCommandQueues[gpu], matrixRange[gpu]),
                       aMatrix->mRows, aMatrix->mCols, bMatrix->mCols, gpu, aMatBuffer, bMatBuffer,
                       resultMatBuffer);

            mCommandQueues[gpu].finish();

            auto resultMatrix = std::dynamic_pointer_cast<Matrix<type>>(result);
            if (gpu == 0)
            {
                cl::copy(mCommandQueues[gpu], resultMatBuffer,
                         resultMatrix->mData.begin(), resultMatrix->mData.end());
            }
            else if (gpu == 1)
            {
                helper.reset(new Matrix<type>(resultMatrix->mRows, 
                                              resultMatrix->mCols));

                auto helperMatrix = std::dynamic_pointer_cast<Matrix<type>>(helper);
                cl::copy(mCommandQueues[gpu], resultMatBuffer,
                         helperMatrix->mData.begin(),
                         helperMatrix->mData.end());
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

    std::cout << "-------------------------------------------------"
              << std::endl;
};

template <typename type>
void MatrixMultiplication<type>::MergeData()
{
    auto resultMatrix = std::dynamic_pointer_cast<Matrix<type>>(result);
    auto helperMatrix = std::dynamic_pointer_cast<Matrix<type>>(helper);

    auto start = std::chrono::system_clock::now();

    std::transform(resultMatrix->mData.begin(), resultMatrix->mData.end(),
                   helperMatrix->mData.begin(), resultMatrix->mData.begin(),
                   std::plus<type>());

    auto end = std::chrono::system_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Merging data time: " << diff.count() << " ms" << std::endl;
};

template <typename type>
bool MatrixMultiplication<type>::LoadKernel()
{
    std::string fileName("mulMatrix.cl");
    std::ifstream stream(fileName.c_str());
    if (!stream.is_open())
    {
        //fileName = ("mulMatrix.cl");
        fileName = ("C:\\RND200\\OpenCL-SDK\\samples\\core\\mulmatrix\\mulMatrix.cl");
        stream = std::ifstream(fileName.c_str());
        //std::cout << "Cannot open file: " << fileName << std::endl;
        //return false;
    }
    mKernelSource =
        std::move(std::string(std::istreambuf_iterator<char>(stream),
                             (std::istreambuf_iterator<char>())));

    std::string templateType(typeid(type).name());

    mKernelSource = str_sprintf(mKernelSource, templateType.c_str(), templateType.c_str(),
                    templateType.c_str(), templateType.c_str());

    return true;
};

template <typename type>
void MatrixMultiplication<type>::ExportToCSV(const std::string& filename)
{
    std::ofstream resultFile;
    resultFile.open(filename, std::ios::out | std::ios::trunc);
    if (resultFile.is_open())
    {
        auto aMatrix = std::dynamic_pointer_cast<Matrix<type>>(A);
        auto bMatrix = std::dynamic_pointer_cast<Matrix<type>>(B);
        auto resultMatrix = std::dynamic_pointer_cast<Matrix<type>>(result);

        std::cout << "Exporting results to CSV file " << filename << std::endl;

        resultFile << "Matrix A [" << aMatrix->mRows << " x " << aMatrix->mCols << "]\n\n";

        auto rows = aMatrix->mRows;
        auto cols = aMatrix->mCols;
        for (unsigned int r = 0; r < rows; ++r)
        {
            for (unsigned int c = 0; c < cols; ++c)
            {
                resultFile << aMatrix->mData[r * cols + c] << ",";
            }
            resultFile << "\n";
        }

        resultFile << "\nMatrix B [" << bMatrix->mRows << " x " << bMatrix->mCols << "]\n\n";
        rows = bMatrix->mRows;
        cols = bMatrix->mCols;
        for (unsigned int r = 0; r < rows; ++r)
        {
            for (unsigned int c = 0; c < cols; ++c)
            {
                resultFile << bMatrix->mData[r * cols + c] << ",";
            }
            resultFile << "\n";
        }

        resultFile << "\nMultiplication of matrices A x B [" << resultMatrix->mRows
                   << " x " << resultMatrix->mCols << "]\n\n";
        rows = resultMatrix->mRows;
        cols = resultMatrix->mCols;
        for (unsigned int r = 0; r < rows; ++r)
        {
            for (unsigned int c = 0; c < cols; ++c)
            {
                resultFile << resultMatrix->mData[r * cols + c] << ",";
            }
            resultFile << "\n";
        }
        resultFile.close();
        std::cout << "-------------------------------------------------"
                  << std::endl
                  << std::endl;
    }
};


template <> auto cl::sdk::parse<SampleOptions>()
{
    return std::make_tuple(
        std::make_shared<TCLAP::ValueArg<unsigned int>>(
            "v", "type",
            "1: cl_short, 2: cl_int, 3: cl_long, 4: cl_half, 5: cl_float, 6: cl_double (default cl_int)",
            false, 2, "integral"),
        std::make_shared<TCLAP::ValueArg<int>>(
            "t", "to",
            "Upper range to generate matrix data (maximum 100 default 100)", false, TO, "integral"),
        std::make_shared<TCLAP::ValueArg<int>>(
            "f", "from",
            "Bottom range to generate matrix data (minimum -100 default -100)", false, FROM, "integral"),
        std::make_shared<TCLAP::ValueArg<unsigned int>>(
            "p", "cols2", "Second matrix columns number (maximum 4096 default 4096)", false,
            7, "positive integral"),
        std::make_shared<TCLAP::ValueArg<unsigned int>>(
            "n", "cols1", "First matrix columns number (maximum 4096 default 4096)", false,
            6, "positive integral"),
        std::make_shared<TCLAP::ValueArg<unsigned int>>(
            "m", "rows1", "First matrix rows number (maximum 4096 default 4096)", false,
            3, "positive integral"));
}
template <>
SampleOptions cl::sdk::comprehend<SampleOptions>(
    std::shared_ptr<TCLAP::ValueArg<unsigned int>> type,
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

    unsigned int t = type->getValue();
    if (t < 1 || t > 6)
    {
        std::cout << "Matrix values type must be from 1 - 6. Default 2 "
                     "(cl_int) will be used."
                  << std::endl;
        t = 2;
    }

    return SampleOptions{
        rows1,
        cols1,
        cols1, // second matrix rows number must be equal to first matrix column number
        cols2,
        bottom, upper,
        t};
}

void createMatrixObj(std::shared_ptr<IMatrix>& matrixPtr,
                     unsigned int type, int rows, int cols, int from, int to)
{
    switch (type)
    {
        case 1:
            matrixPtr.reset(new Matrix<cl_short>(rows, cols, from, to));
            break;
        case 2:
            matrixPtr.reset(new Matrix<cl_int>(rows, cols, from, to));
            break;
        case 3:
            matrixPtr.reset(new Matrix<cl_long>(rows, cols, from, to));
            break;
        case 4: {
            auto f = from < 0 ? 0 : from;
            auto t = to < 0 ? to * -1 : to;
            matrixPtr.reset(
                new Matrix<cl_half>(rows, cols, f, t));
        }
            break;
        case 5:
            matrixPtr.reset(new Matrix<cl_float>(rows, cols, from, to));
            break;
        case 6:
            matrixPtr.reset(new Matrix<cl_double>(rows, cols, from, to));
            break;
        default:
            matrixPtr.reset(new Matrix<cl_int>(rows, cols, from, to));
            break;
    }
}

template<typename T>
void createMatrixMultObj(
            std::shared_ptr<IMatrixMultiplication>& matrixPtr,
            unsigned int type,
            T platform,
            std::shared_ptr<IMatrix> a,
            std::shared_ptr<IMatrix> b)
{
    switch (type)
    {
        case 1:
            if (std::is_same<std::vector<cl::Platform>, T>::value)
            {
                matrixPtr.reset(
                    new MatrixMultiplication<cl_short>(platform, a, b));
            }
            else if (std::is_same<cl::Platform, T>::value)
            {
                matrixPtr.reset(
                    new MatrixMultiplication<cl_short>(platform, a, b));
            }
            break;
        case 2:
            if (std::is_same<std::vector<cl::Platform>, T>::value)
            {
                matrixPtr.reset(
                    new MatrixMultiplication<cl_int>(platform, a, b));
            }
            else if (std::is_same<cl::Platform, T>::value)
            {
                matrixPtr.reset(
                    new MatrixMultiplication<cl_int>(platform, a, b));
            }
            break;
        case 3:
            if (std::is_same<std::vector<cl::Platform>, T>::value)
            {
                matrixPtr.reset(
                    new MatrixMultiplication<cl_long>(platform, a, b));
            }
            else if (std::is_same<cl::Platform, T>::value)
            {
                matrixPtr.reset(
                    new MatrixMultiplication<cl_long>(platform, a, b));
            }
            break;
        case 4:
            if (std::is_same<std::vector<cl::Platform>, T>::value)
            {
                matrixPtr.reset(
                    new MatrixMultiplication<cl_half>(platform, a, b));
            }
            else if (std::is_same<cl::Platform, T>::value)
            {
                matrixPtr.reset(
                    new MatrixMultiplication<cl_half>(platform, a, b));
            }
            break;
        case 5:
            if (std::is_same<std::vector<cl::Platform>, T>::value)
            {
                matrixPtr.reset(
                    new MatrixMultiplication<cl_float>(platform, a, b));
            }
            else if (std::is_same<cl::Platform, T>::value)
            {
                matrixPtr.reset(
                    new MatrixMultiplication<cl_float>(platform, a, b));
            }
            break;
        case 6:
            if (std::is_same<std::vector<cl::Platform>, T>::value)
            {
                matrixPtr.reset(
                    new MatrixMultiplication<cl_double>(platform, a, b));
            }
            else if (std::is_same<cl::Platform, T>::value)
            {
                matrixPtr.reset(
                    new MatrixMultiplication<cl_double>(platform, a, b));
            }
            break;
        default:
            if (std::is_same<std::vector<cl::Platform>, T>::value)
            {
                matrixPtr.reset(
                    new MatrixMultiplication<cl_int>(platform, a, b));
            }
            else if (std::is_same<cl::Platform, T>::value)
            {
                matrixPtr.reset(
                    new MatrixMultiplication<cl_int>(platform, a, b));
            }
            break;
    }
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

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    std::shared_ptr<IMatrix> a;
    std::shared_ptr<IMatrix> b;

    createMatrixObj(a, options.type, rows1, cols1, options.from, options.to);
    createMatrixObj(b, options.type, rows2, cols2, options.from, options.to);

    if (!platforms.empty())
    {
        std::shared_ptr<IMatrixMultiplication> singlePlatformMul;
        createMatrixMultObj(singlePlatformMul, options.type, platforms[0], a, b);

        singlePlatformMul->Multiply();
        singlePlatformMul->ExportToCSV("mulmatrix_first_platform.csv");
    }
    if (platforms.size() > 1)
    {
        std::shared_ptr<IMatrixMultiplication> singlePlatformMul;
        std::shared_ptr<IMatrixMultiplication> multiPlatformMul;

        createMatrixMultObj(singlePlatformMul, options.type, platforms[1], a, b);
        createMatrixMultObj(multiPlatformMul, options.type, platforms, a, b);

        singlePlatformMul->Multiply();
        singlePlatformMul->ExportToCSV("mulmatrix_second_platform.csv");

        multiPlatformMul->Multiply();
        multiPlatformMul->ExportToCSV("mulmatrix_two_platforms.csv");
    }

    char exit;
    std::cout << "Type any letter and press enter to exit." << std::endl;
    std::cin >> exit;

    return 0;
}