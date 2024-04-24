/*
// Copyright (c) 2021-2022 Ben Ashbaugh
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
*/

// The code in this sample was derived from several samples in the Vulkan
// Tutorial: https://vulkan-tutorial.com
//
// The code samples in the Vulkan Tutorial are licensed as CC0 1.0 Universal.

#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <popl/popl.hpp>

#include "opencl.hpp"
#if !defined(cl_khr_external_memory)
#error cl_khr_external_memory not found, please update your OpenCL headers!
#endif
#if !defined(cl_khr_external_semaphore)
#error cl_khr_external_semaphore not found, please update your OpenCL headers!
#endif

#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "util.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <set>
#include <stdexcept>
#include <vector>

#include <math.h>

// GLM includes
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// ocean mesh patch size
#define NX 1024
#define NY 1024

#define WX 1.f
#define WY 1.f

#define DRAG_SPEED_FAC 0.05f


namespace {

std::int32_t current_array_layer = 0;

const char kernelString[] =

R"CLC(kernel void Julia( write_only image2d_t dst, float cr, float ci ))
{
    const float cMinX = -1.5f;
    const float cMaxX =  1.5f;
    const float cMinY = -1.5f;
    const float cMaxY =  1.5f;

    const int cWidth = get_global_size(0);
    const int cHeight = get_global_size(1);
    const int cIterations = 16;

    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);

    float a = x * ( cMaxX - cMinX ) / cWidth + cMinX;
    float b = y * ( cMaxY - cMinY ) / cHeight + cMinY;

    float result = 0.0f;
    const float thresholdSquared = cIterations * cIterations / 64.0f;

    for( int i = 0; i < cIterations; i++ ) {
        float aa = a * a;
        float bb = b * b;

        float magnitudeSquared = aa + bb;
        if( magnitudeSquared >= thresholdSquared ) {
            break;
        }

        result += 1.0f / cIterations;
        b = 2 * a * b + ci;
        a = aa - bb + cr;
    }

    result = max( result, 0.0f );
    result = min( result, 1.0f );

    // RGBA
    float4 color = (float4)(result + 0.6f, result, result * result, 1.0f);)
    write_imagef(dst, (int2)(x, y), color);
}
)CLC";


const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation",
    //"VK_LAYER_LUNARG_api_dump", // useful for debugging but adds a LOT of
    //output!
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pDebugMessenger)
{
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr)
    {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else
    {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks* pAllocator)
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr)
    {
        func(instance, debugMessenger, pAllocator);
    }
}

struct QueueFamilyIndices
{
    uint32_t graphicsFamily;
    uint32_t presentFamily;

    QueueFamilyIndices(): graphicsFamily(~0), presentFamily(~0) {}

    bool isComplete() { return graphicsFamily != ~0 && presentFamily != ~0; }
};

struct SwapChainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct UniformBufferObject
{
    alignas(4) glm::mat4    view_proj;
    alignas(4) glm::vec3    sun_dir;
    alignas(4) glm::vec3    cam_pos;
    alignas(4) std::int32_t ocean_size = 0;
    alignas(4) std::int32_t tex_size = 0;
};


struct Vertex {

    glm::vec3 pos;
    glm::vec2 tc;

    static VkVertexInputBindingDescription getBindingDescription() {
            VkVertexInputBindingDescription bindingDescription{};

            bindingDescription.binding = 0;
            bindingDescription.stride = sizeof(Vertex);
            bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

            return bindingDescription;
        }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, tc);

        return attributeDescriptions;
    }
};

struct Camera
{
    glm::vec3 eye = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 dir = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 rvec = glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec2 begin = glm::vec2(-1.0f, -1.0f);
    float yaw = -90.0f;
    float pitch = 0.0f;
    bool drag = false;
};

class OceanApplication {

public:
    void run(int argc, char** argv)
    {
        commandLine(argc, argv);
        initWindow();
        initOpenCL();
        initVulkan();
        initOpenCLMems();
        initOpenCLSemaphores();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window;

    Camera camera;

    bool animate = false;
    bool redraw = false;

    size_t gwx = 512;
    size_t gwy = 512;

    size_t lwx = 0;
    size_t lwy = 0;

    float cr = -0.123f;
    float ci = 0.745f;

    bool vsync = true;
    size_t startFrame = 0;
    size_t frame = 0;
    std::chrono::system_clock::time_point start =
        std::chrono::system_clock::now();

    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;

    VkQueue graphicsQueue;
    VkQueue presentQueue;

    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;

    VkRenderPass renderPass;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    VkCommandPool commandPool;

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    bool linearImages = false;
    bool deviceLocalImages = true;

    // Only last stage image must be shared between OCL and vulkan
    // those images will hold displacements and normals.

    enum InteropTexType
    {
        IOPT_DISPLACEMENT = 0,
        IOPT_NORMAL_MAP,
        IOPT_COUNT
    };

    struct TextureOCL
    {
        std::vector<VkImage> images;
        std::vector<VkDeviceMemory> imageMemories;
        std::vector<VkImageView> imageViews;
    };

    // 0 - displacement map, 1 - normal map
    std::array<TextureOCL, IOPT_COUNT> textureImages;

    // Ocean grid vertices and related buffers
    std::vector<Vertex> verts;
    std::vector<VkBuffer> vertexBuffers;
    std::vector<VkDeviceMemory> vertexBufferMemories;

    std::vector <std::uint16_t> inds;
    std::vector<VkBuffer> indexBuffers;
    std::vector<VkDeviceMemory> indexBufferMemories;

    std::array<VkSampler, IOPT_COUNT> textureSampler;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkSemaphore> openclFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> imagesInFlight;
    size_t currentFrame = 0;

#ifdef _WIN32
    PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHR = NULL;
    PFN_vkGetSemaphoreWin32HandleKHR vkGetSemaphoreWin32HandleKHR = NULL;
#elif defined(__linux__)
    PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR = NULL;
    PFN_vkGetSemaphoreFdKHR vkGetSemaphoreFdKHR = NULL;
#endif

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;

    // common stuff
    int platformIndex = 0;
    int deviceIndex = 0;

    bool deviceLocalBuffers = true;
    bool useExternalMemory = true;
    bool useExternalSemaphore = true;


    // OpenCL stuff
    cl_external_memory_handle_type_khr externalMemType = 0;

    cl::Context context;
    cl::CommandQueue commandQueue;
    cl::Kernel kernel;

    // IFFT intermediate computation storages
    std::unique_ptr<cl::Image> h0pk;
    std::unique_ptr<cl::Image> h0mk;
    std::unique_ptr<cl::Image> hkt_ping_pong[2];
    std::unique_ptr<cl::Image> phase_pingpong[2];

    size_t ocl_max_img2d_width;
    cl_ulong ocl_max_alloc_size, ocl_mem_size;

    // final computation result with displacements and normal map, needs to follow swap-chain scheme
    std::array<std::vector<std::unique_ptr<cl::Image>>, IOPT_COUNT> mems;
    std::vector<cl::Semaphore> signalSemaphores;

    clEnqueueAcquireExternalMemObjectsKHR_fn
        clEnqueueAcquireExternalMemObjectsKHR = NULL;
    clEnqueueReleaseExternalMemObjectsKHR_fn
        clEnqueueReleaseExternalMemObjectsKHR = NULL;

    clCreateSemaphoreWithPropertiesKHR_fn clCreateSemaphoreWithPropertiesKHR =
        NULL;
    clEnqueueSignalSemaphoresKHR_fn clEnqueueSignalSemaphoresKHR = NULL;
    clReleaseSemaphoreKHR_fn clReleaseSemaphoreKHR = NULL;

    void commandLine(int argc, char** argv)
    {
        bool hostCopy = false;
        bool hostSync = false;
        bool noDeviceLocal = false;
        bool immediate = false;
        bool paused = false;

        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index",
                                 platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex,
                                 &deviceIndex);
        op.add<popl::Switch>("", "hostcopy",
                             "Do not use cl_khr_external_memory", &hostCopy);
        op.add<popl::Switch>("", "hostsync",
                             "Do not use cl_khr_external_semaphore", &hostSync);
        op.add<popl::Switch>("", "linear", "Use linearly tiled images",
                             &linearImages);
        op.add<popl::Switch>("", "nodevicelocal",
                             "Do not use device local images", &noDeviceLocal);
        op.add<popl::Value<size_t>>(
            "", "gwx", "Global Work Size X AKA Image Width", gwx, &gwx);
        op.add<popl::Value<size_t>>(
            "", "gwy", "Global Work Size Y AKA Image Height", gwy, &gwy);
        op.add<popl::Value<size_t>>("", "lwx", "Local Work Size X", lwx, &lwx);
        op.add<popl::Value<size_t>>("", "lwy", "Local Work Size Y", lwy, &lwy);
        op.add<popl::Switch>("", "immediate",
                             "Prefer VK_PRESENT_MODE_IMMEDIATE_KHR (no vsync)",
                             &immediate);
        op.add<popl::Switch>("", "paused", "Start with Animation Paused",
                             &paused);

        bool printUsage = false;
        try
        {
            op.parse(argc, argv);
        } catch (std::exception& e)
        {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty()
            || !op.non_option_args().empty())
        {
            fprintf(stderr,
                    "Usage: juliavk [options]\n"
                    "%s",
                    op.help().c_str());
            throw std::runtime_error("exiting.");
        }

        deviceLocalImages = !noDeviceLocal;
        useExternalMemory = !hostCopy;
        useExternalSemaphore = !hostSync;
        vsync = !immediate;
        animate = !paused;
    }

    void initWindow()
    {
        if (!glfwInit())
        {
            throw std::runtime_error("failed to initialize glfw!");
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow((int)gwx, (int)gwy, "Julia Set with Vulkan",
                                  nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
    }

    void initOpenCL()
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        printf("Running on platform: %s\n",
               platforms[platformIndex].getInfo<CL_PLATFORM_NAME>().c_str());

        std::vector<cl::Device> devices;
        platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);

        printf("Running on device: %s\n",
               devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str());

        checkOpenCLExternalMemorySupport(devices[deviceIndex]);
        checkOpenCLExternalSemaphoreSupport(devices[deviceIndex]);

        if (useExternalMemory)
        {
            clEnqueueAcquireExternalMemObjectsKHR =
                (clEnqueueAcquireExternalMemObjectsKHR_fn)
                    clGetExtensionFunctionAddressForPlatform(
                        platforms[platformIndex](),
                        "clEnqueueAcquireExternalMemObjectsKHR");
            clEnqueueReleaseExternalMemObjectsKHR =
                (clEnqueueReleaseExternalMemObjectsKHR_fn)
                    clGetExtensionFunctionAddressForPlatform(
                        platforms[platformIndex](),
                        "clEnqueueReleaseExternalMemObjectsKHR");
            if (clEnqueueAcquireExternalMemObjectsKHR == NULL
                || clEnqueueReleaseExternalMemObjectsKHR == NULL)
            {
                throw std::runtime_error("couldn't get function pointers for "
                                         "cl_khr_external_memory");
            }
        }

        if (useExternalSemaphore)
        {
            clCreateSemaphoreWithPropertiesKHR =
                (clCreateSemaphoreWithPropertiesKHR_fn)
                    clGetExtensionFunctionAddressForPlatform(
                        platforms[platformIndex](),
                        "clCreateSemaphoreWithPropertiesKHR");
            clEnqueueSignalSemaphoresKHR = (clEnqueueSignalSemaphoresKHR_fn)
                clGetExtensionFunctionAddressForPlatform(
                    platforms[platformIndex](), "clEnqueueSignalSemaphoresKHR");
            clReleaseSemaphoreKHR = (clReleaseSemaphoreKHR_fn)
                clGetExtensionFunctionAddressForPlatform(
                    platforms[platformIndex](), "clReleaseSemaphoreKHR");
            if (clCreateSemaphoreWithPropertiesKHR == NULL
                || clEnqueueSignalSemaphoresKHR == NULL
                || clReleaseSemaphoreKHR == NULL)
            {
                throw std::runtime_error("couldn't get function pointers for "
                                         "cl_khr_external_semaphore");
            }
        }

        int error = CL_SUCCESS;
        error |= clGetDeviceInfo( devices[deviceIndex](), CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof( ocl_max_img2d_width ), &ocl_max_img2d_width, NULL );
        error |= clGetDeviceInfo( devices[deviceIndex](), CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof( ocl_max_alloc_size ), &ocl_max_alloc_size, NULL );
        error |= clGetDeviceInfo( devices[deviceIndex](), CL_DEVICE_GLOBAL_MEM_SIZE, sizeof( ocl_mem_size ), &ocl_mem_size, NULL );

        if (error!=CL_SUCCESS)
            printf("clGetDeviceInfo error: %d\n", error);

        context = cl::Context{ devices[deviceIndex] };
        commandQueue = cl::CommandQueue{ context, devices[deviceIndex] };

        cl::Program program{ context, kernelString };

        try
        {
            program.build();
        } catch (const cl::BuildError& e)
        {
            auto bl = e.getBuildLog();
            printf("Build OpenCL kernel error: \n");
            for (auto elem : bl) printf("%s\n", elem.second.c_str());
            exit(1);
        }

        kernel = cl::Kernel{ program, "Julia" };
    }

    void initOpenCLMems()
    {
        for (size_t target = 0; target < IOPT_COUNT; target++)
        {
            mems[target].resize(swapChainImages.size());

            for (size_t img_num = 0; img_num < textureImages.size(); img_num++)
            {
                for (size_t i = 0; i < swapChainImages.size(); i++)
                {
                    if (useExternalMemory)
                    {
#ifdef _WIN32
                        HANDLE handle = NULL;
                        VkMemoryGetWin32HandleInfoKHR getWin32HandleInfo{};
                        getWin32HandleInfo.sType =
                            VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
                        getWin32HandleInfo.memory =
                            textureImages[img_num].imageMemories[i];
                        getWin32HandleInfo.handleType =
                            VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
                        vkGetMemoryWin32HandleKHR(device, &getWin32HandleInfo,
                                                  &handle);

                        const cl_mem_properties props[] = {
                            externalMemType,
                            (cl_mem_properties)handle,
                            0,
                        };
#elif defined(__linux__)
                        int fd = 0;
                        VkMemoryGetFdInfoKHR getFdInfo{};
                        getFdInfo.sType =
                            VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
                        getFdInfo.memory =
                            textureImages[img_num]
                                .imageMemories[i]; // textureImageMemories[i];
                        getFdInfo.handleType = externalMemType
                                == CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR
                            ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
                            : VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
                        vkGetMemoryFdKHR(device, &getFdInfo, &fd);

                        const cl_mem_properties props[] = {
                            externalMemType,
                            (cl_mem_properties)fd,
                            0,
                        };
#else
                        const cl_mem_properties* props = NULL;
#endif

                        cl::vector<cl_mem_properties> vprops(
                            sizeof(props) / sizeof(props[0]));
                        std::memcpy(vprops.data(), props,
                                    sizeof(cl_mem_properties) * vprops.size());

                        mems[target][i].reset(new cl::Image2D(
                            context, vprops, CL_MEM_WRITE_ONLY,
                            cl::ImageFormat(CL_RGBA, CL_UNORM_INT8), gwx, gwy));
                    }
                    else
                    {
                        mems[target][i].reset(new cl::Image2D{
                            context, CL_MEM_WRITE_ONLY,
                            cl::ImageFormat{ CL_RGBA, CL_UNORM_INT8 }, gwx,
                            gwy });
                    }
                }
            }
        }
    }

    void initOpenCLSemaphores()
    {
        if (useExternalSemaphore)
        {
            signalSemaphores.resize(MAX_FRAMES_IN_FLIGHT);

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
            {
                createOpenCLSemaphoreFromVulkanSemaphore(
                    openclFinishedSemaphores[i], signalSemaphores[i]);
            }
        }
    }

    void initVulkan()
    {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createUniformBuffer();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();

        createVertexBuffers();
        createIndexBuffers();

        createTextureImages();
        createTextureImageViews();
        createTextureSampler();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

    void mainLoop()
    {
        glfwSetKeyCallback(window, keyboard);
        glfwSetMouseButtonCallback(window, mouse_event);
        glfwSetCursorPosCallback(window, mouse_pos);

        while (!glfwWindowShouldClose(window))
        {
            if (animate || redraw)
            {
                drawFrame();
            }
            glfwPollEvents();
        }

        vkDeviceWaitIdle(device);
    }

    void keyboard(int key, int scancode, int action, int mods)
    {
        if (action == GLFW_PRESS || action == GLFW_REPEAT)
        {
            redraw = true;

            switch (key)
            {
                case GLFW_KEY_ESCAPE:
                    glfwSetWindowShouldClose(window, GLFW_TRUE);
                    break;
                case GLFW_KEY_SPACE:
                    animate = !animate;
                    printf("animation is %s\n", animate ? "ON" : "OFF");
                    break;

                case GLFW_KEY_A: cr += 0.005f; break;
                case GLFW_KEY_Z: cr -= 0.005f; break;

                case GLFW_KEY_S: ci += 0.005f; break;
                case GLFW_KEY_X: ci -= 0.005f; break;
            }
        }
    }

    void mouse_event(int button, int action, int mods) {
        double x, y;
        glfwGetCursorPos(window, &x, &y);
        switch (action) {
            case 0:
                // Button Up
                if (button == 1) {
                    camera.drag = false;
                }
                break;
            case 1:
                // Button Down
                if (button == 1) {
                    camera.drag = true;
                    camera.begin = glm::vec2(x, y);
                }
                break;
            default:
                break;
        }
    }

    void mouse_pos(double pX, double pY)
    {
        if (!camera.drag)
            return;

        glm::vec2 off = camera.begin - glm::vec2(pX, pY);
        camera.begin = glm::vec2(pX, pY);

        camera.yaw -= off.x * DRAG_SPEED_FAC;
        camera.pitch += off.y * DRAG_SPEED_FAC;
        glm::vec3 dir(
            cos(glm::radians(camera.yaw)) * cos(glm::radians(camera.pitch)),
            sin(glm::radians(camera.pitch)),
            sin(glm::radians(camera.yaw)) * cos(glm::radians(camera.pitch)));

        camera.dir = glm::normalize(dir);
        camera.rvec = glm::normalize(glm::cross(camera.dir, glm::vec3(0, 1, 0)));
        camera.up = glm::normalize(glm::cross(camera.rvec, camera.dir));
    }

    void cleanup()
    {
        for (auto semaphore : signalSemaphores)
        {
            semaphore.release();
        }

        for (auto framebuffer : swapChainFramebuffers)
        {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        // vkFreeCommandBuffers?

        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);

        for (auto imageView : swapChainImageViews)
        {
            vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);

        for (size_t img_num = 0; img_num < textureImages.size(); img_num++ )
        {
            for (auto textureImageView : textureImages[img_num].imageViews)
            {
                vkDestroyImageView(device, textureImageView, nullptr);
            }
            for (auto textureImage : textureImages[img_num].images)
            {
                vkDestroyImage(device, textureImage, nullptr);
            }
            for (auto textureImageMemory : textureImages[img_num].imageMemories)
            {
                vkFreeMemory(device, textureImageMemory, nullptr);
            }
        }

        for (size_t sampler_num = 0; sampler_num < textureSampler.size(); sampler_num++ )
        {
            vkDestroySampler(device, textureSampler[sampler_num], nullptr);
        }

        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        // cleanup vertices buffers
        for (auto buffer : vertexBuffers) {
            vkDestroyBuffer(device, buffer, nullptr);
        }

        for (auto bufferMemory : vertexBufferMemories) {
            vkFreeMemory(device, bufferMemory, nullptr);
        }

        // cleanup indices buffers
        for (auto buffer : indexBuffers) {
            vkDestroyBuffer(device, buffer, nullptr);
        }

        for (auto bufferMemory : indexBufferMemories) {
            vkFreeMemory(device, bufferMemory, nullptr);
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            if (useExternalSemaphore)
            {
                vkDestroySemaphore(device, openclFinishedSemaphores[i],
                                   nullptr);
            }
            vkDestroyFence(device, inFlightFences[i], nullptr);

            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
        }
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);


        vkDestroyCommandPool(device, commandPool, nullptr);

        vkDestroyDevice(device, nullptr);

        if (enableValidationLayers)
        {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);

        glfwTerminate();
    }

    void createInstance()
    {
        if (enableValidationLayers && !checkValidationLayerSupport())
        {
            throw std::runtime_error(
                "validation layers requested, but not available!");
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Julia Set OpenCL+Vulkan Sample";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        if (useExternalMemory || useExternalSemaphore)
        {
            appInfo.apiVersion = VK_API_VERSION_1_1;
        }
        else
        {
            appInfo.apiVersion = VK_API_VERSION_1_0;
        }

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount =
            static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers)
        {
            createInfo.enabledLayerCount =
                static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext =
                (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
        }
        else
        {
            createInfo.enabledLayerCount = 0;

            createInfo.pNext = nullptr;
        }

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create instance!");
        }

#ifdef _WIN32
        if (useExternalMemory)
        {
            vkGetMemoryWin32HandleKHR =
                (PFN_vkGetMemoryWin32HandleKHR)vkGetInstanceProcAddr(
                    instance, "vkGetMemoryWin32HandleKHR");
            if (vkGetMemoryWin32HandleKHR == NULL)
            {
                throw std::runtime_error("couldn't get function pointer for "
                                         "vkGetMemoryWin32HandleKHR");
            }
        }
        if (useExternalSemaphore)
        {
            vkGetSemaphoreWin32HandleKHR =
                (PFN_vkGetSemaphoreWin32HandleKHR)vkGetInstanceProcAddr(
                    instance, "vkGetSemaphoreWin32HandleKHR");
            if (vkGetSemaphoreWin32HandleKHR == NULL)
            {
                throw std::runtime_error("couldn't get function pointer for "
                                         "vkGetSemaphoreWin32HandleKHR");
            }
        }
#elif defined(__linux__)
        if (useExternalMemory)
        {
            vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetInstanceProcAddr(
                instance, "vkGetMemoryFdKHR");
            if (vkGetMemoryFdKHR == NULL)
            {
                throw std::runtime_error(
                    "couldn't get function pointer for vkGetMemoryFdKHR");
            }
        }
        if (useExternalSemaphore)
        {
            vkGetSemaphoreFdKHR =
                (PFN_vkGetSemaphoreFdKHR)vkGetInstanceProcAddr(
                    instance, "vkGetSemaphoreFdKHR");
            if (vkGetSemaphoreFdKHR == NULL)
            {
                throw std::runtime_error(
                    "couldn't get function pointer for vkGetSemaphoreFdKHR");
            }
        }
#endif
    }

    void populateDebugMessengerCreateInfo(
        VkDebugUtilsMessengerCreateInfoEXT& createInfo)
    {
        createInfo = {};
        createInfo.sType =
            VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        // createInfo.messageSeverity |=
        // VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
        createInfo.pfnUserCallback = debugCallback;
    }

    void setupDebugMessenger()
    {
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr,
                                         &debugMessenger)
            != VK_SUCCESS)
        {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    void createSurface()
    {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface)
            != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    void pickPhysicalDevice()
    {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0)
        {
            throw std::runtime_error(
                "failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto& device : devices)
        {
            if (isDeviceSuitable(device))
            {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE)
        {
            throw std::runtime_error("failed to find a suitable GPU!");
        }

        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);

        printf("Running on Vulkan physical device: %s\n",
               properties.deviceName);
    }

    void createLogicalDevice()
    {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily,
                                                   indices.presentFamily };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies)
        {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures{};

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.queueCreateInfoCount =
            static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        createInfo.pEnabledFeatures = &deviceFeatures;

        auto extensions = getRequiredDeviceExtensions();
        createInfo.enabledExtensionCount =
            static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        if (enableValidationLayers)
        {
            createInfo.enabledLayerCount =
                static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else
        {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device)
            != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(device, indices.graphicsFamily, 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily, 0, &presentQueue);
    }

    void createSwapChain()
    {
        SwapChainSupportDetails swapChainSupport =
            querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat =
            chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode =
            chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0
            && imageCount > swapChainSupport.capabilities.maxImageCount)
        {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;

        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily,
                                          indices.presentFamily };

        if (indices.graphicsFamily != indices.presentFamily)
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform =
            swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;

        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain)
            != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create swap chain!");
        }

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount,
                                swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    void createImageViews()
    {
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++)
        {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapChainImageFormat;
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;

            if (vkCreateImageView(device, &createInfo, nullptr,
                                  &swapChainImageViews[i])
                != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create image views!");
            }
        }
    }

    void createRenderPass()
    {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass)
            != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    void createUniformBuffer()
    {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(swapChainImages.size());
        uniformBuffersMemory.resize(swapChainImages.size());

        for (size_t i = 0; i < uniformBuffers.size(); i++)
        {
            createBuffer(bufferSize,
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                             | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, uniformBuffers[i],
                         uniformBuffersMemory[i]);
        }
    }

    void createDescriptorSetLayout()
    {
        VkDescriptorSetLayoutBinding sampler0LayoutBinding{};
        sampler0LayoutBinding.binding = 0;
        sampler0LayoutBinding.descriptorCount = 1;
        sampler0LayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sampler0LayoutBinding.pImmutableSamplers = nullptr;
        sampler0LayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutBinding sampler1LayoutBinding{};
        sampler1LayoutBinding.binding = 1;
        sampler1LayoutBinding.descriptorCount = 1;
        sampler1LayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sampler1LayoutBinding.pImmutableSamplers = nullptr;
        sampler1LayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding uniformLayoutBinding{};
        uniformLayoutBinding.binding = 2;
        uniformLayoutBinding.descriptorCount = 1;
        uniformLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uniformLayoutBinding.pImmutableSamplers = nullptr;
        uniformLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        std::array<VkDescriptorSetLayoutBinding, 3> bindings = {
            sampler0LayoutBinding, sampler1LayoutBinding, uniformLayoutBinding
        };

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                                        &descriptorSetLayout)
            != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }

    void createGraphicsPipeline()
    {
        auto vertShaderCode = readFile("ocean.vert.spv");
        auto fragShaderCode = readFile("ocean.frag.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = {
            vertShaderStageInfo, fragShaderStageInfo
        };

        // vertex info
        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType =
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapChainExtent.width;
        viewport.height = (float)swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType =
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType =
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType =
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT
            | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT
            | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType =
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType =
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                                   &pipelineLayout)
            != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                      nullptr, &graphicsPipeline)
            != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    void createFramebuffers()
    {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++)
        {
            VkImageView attachments[] = { swapChainImageViews[i] };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr,
                                    &swapChainFramebuffers[i])
                != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    void createCommandPool()
    {
        QueueFamilyIndices queueFamilyIndices =
            findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily;

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool)
            != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create command pool!");
        }
    }


    void createVertexBuffers() {

        int iCXY = ( NX + 1 ) * ( NY + 1 );
        verts.resize(iCXY);

        // Initialize vertices and normals to default (row, column, 0) and (0,
        // 0, 1) This step is not really neccessary (its just in case something
        // went wrong) Verts and normals will be updated with wave height field
        // every frame
        cl_float dfY = -0.5 * (NY * WY), dfBaseX = -0.5 * (NX * WX);
        cl_float tx=0.f, ty=0.f, dtx = 1.f / NX, dty= 1.f / NY;
        for (int iBase = 0, iY = 0; iY <= NY; iY++, iBase += NX + 1)
        {
            double dfX = dfBaseX;
            for (int iX = 0; iX <= NX; iX++)
            {
                verts[iBase + iX].pos = glm::vec3(dfX, dfY, 0.0);
                verts[iBase + iX].tc = glm::vec2(tx, ty);
                tx += dtx;
                dfX += WX;
            }
            dfY += WY;
            ty += dty;
        }

        vertexBuffers.resize(swapChainImages.size());
        vertexBufferMemories.resize(swapChainImages.size());

#if 0
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = sizeof(verts[0]) * verts.size();
        bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkMemoryPropertyFlags properties =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        for (size_t i = 0; i < swapChainImages.size(); i++) {

            if (vkCreateBuffer(device, &bufferInfo, nullptr, &vertexBuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create vertex buffer!");
            }

            VkMemoryRequirements memRequirements;
            vkGetBufferMemoryRequirements(device, vertexBuffers[i], &memRequirements);

            VkMemoryAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

            if (vkAllocateMemory(device, &allocInfo, nullptr, &vertexBufferMemories[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to allocate vertex buffer memory!");
            }

            vkBindBufferMemory(device, vertexBuffers[i], vertexBufferMemories[i], 0);
        }

        for (size_t i = 0; i < swapChainImages.size(); i++)
        {
            void* data;
            vkMapMemory(device, vertexBufferMemories[i], 0, bufferInfo.size, 0, &data);
                memcpy(data, verts.data(), (size_t) bufferInfo.size);
            vkUnmapMemory(device, vertexBufferMemories[i]);
        }
#else


        VkDeviceSize bufferSize = sizeof(verts[0]) * verts.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
            memcpy(data, verts.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        for (size_t i = 0; i < swapChainImages.size(); i++) {

            // create local memory buffer
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         vertexBuffers[i], vertexBufferMemories[i]);

            copyBuffer(stagingBuffer, vertexBuffers[i], bufferSize);
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
#endif
    }



    void createIndexBuffers() {

        // Add Tri Strip primitve sets
        inds.resize(NX + NX + 2);
        // Each tri strip draws one row of NX quads
        for (int iBaseTo, iBaseFrom = 0, iY = 0; iY < NY;
             iY++, iBaseFrom = iBaseTo)
        {
            iBaseTo = iBaseFrom + NX + 1;
            for (int iX = 0; iX <= NX; iX++)
            {
                inds[iX + iX + 0] = iBaseFrom + iX;
                inds[iX + iX + 1] = iBaseTo + iX;
            }
        }

        indexBuffers.resize(swapChainImages.size());
        indexBufferMemories.resize(swapChainImages.size());

        VkDeviceSize bufferSize = sizeof(inds[0]) * inds.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
            memcpy(data, inds.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        for (size_t i = 0; i < swapChainImages.size(); i++) {

            // create local memory buffer
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         indexBuffers[i], indexBufferMemories[i]);

            copyBuffer(stagingBuffer, indexBuffers[i], bufferSize);
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createTextureImages()
    {
        VkImageTiling tiling =
            linearImages ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL;
        VkMemoryPropertyFlags properties =
            deviceLocalImages ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT : 0;

        uint32_t texWidth = static_cast<uint32_t>(gwx);
        uint32_t texHeight = static_cast<uint32_t>(gwy);

        VkDeviceSize imageSize = texWidth * texHeight * 4;

        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                         | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingBufferMemory);

        for (size_t img_num = 0; img_num < textureImages.size(); img_num++ )
        {
            textureImages[img_num].images.resize(swapChainImages.size());
            textureImages[img_num].imageMemories.resize(swapChainImages.size());

            for (size_t i = 0; i < swapChainImages.size(); i++)
            {
                createShareableImage(
                    texWidth, texHeight, VK_FORMAT_R32G32B32A32_SFLOAT, tiling,
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    properties, textureImages[img_num].images[i], textureImages[img_num].imageMemories[i]
                );
                if (useExternalMemory)
                {
                    transitionImageLayout(textureImages[img_num].images[i],
                                          VK_FORMAT_R32G32B32A32_SFLOAT,
                                          VK_IMAGE_LAYOUT_UNDEFINED,
                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                    );
                }
            }
        }
    }

    void createTextureImageViews()
    {
        for (size_t img_num = 0; img_num < textureImages.size(); img_num++ )
        {
            textureImages[img_num].imageViews.resize(swapChainImages.size());

            for (size_t i = 0; i < swapChainImages.size(); i++)
            {
                textureImages[img_num].imageViews[i] =
                    createImageView(textureImages[img_num].images[i], VK_FORMAT_R32G32B32A32_SFLOAT);
            }
        }
    }

    void createTextureSampler()
    {
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_NEAREST;
        samplerInfo.minFilter = VK_FILTER_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

        for (size_t sampler_num = 0; sampler_num < textureSampler.size(); sampler_num++ )
        {
            if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler[sampler_num])
                != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create texture sampler!");
            }
        }
    }

    VkImageView createImageView(VkImage image, VkFormat format,
                                VkImageViewType type = VK_IMAGE_VIEW_TYPE_2D,
                                uint32_t layers = 1)
    {
        VkImageViewCreateInfo viewInfo{
            VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO
        };
        viewInfo.pNext = nullptr;
        viewInfo.image = image;
        viewInfo.viewType = type;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1; // VK_REMAINING_MIP_LEVELS;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount =
            layers; // VK_REMAINING_ARRAY_LAYERS;

        VkImageView imageView;
        if (vkCreateImageView(device, &viewInfo, nullptr, &imageView)
            != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create texture image view!");
        }

        return imageView;
    }

    void createShareableImage(uint32_t width, uint32_t height, VkFormat format,
                              VkImageTiling tiling, VkImageUsageFlags usage,
                              VkMemoryPropertyFlags properties, VkImage& image,
                              VkDeviceMemory& imageMemory,
                              VkImageType type = VK_IMAGE_TYPE_2D)
    {
        VkExternalMemoryImageCreateInfo externalMemCreateInfo{};
        externalMemCreateInfo.sType =
            VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;

#ifdef _WIN32
        externalMemCreateInfo.handleTypes =
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#elif defined(__linux__)
        externalMemCreateInfo.handleTypes =
            externalMemType == CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR
            ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
            : VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
#endif

        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        if (useExternalMemory)
        {
            imageInfo.pNext = &externalMemCreateInfo;
        }

        imageInfo.imageType = type;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);

        VkExportMemoryAllocateInfo exportMemoryAllocInfo{};
        exportMemoryAllocInfo.sType =
            VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        exportMemoryAllocInfo.handleTypes = externalMemCreateInfo.handleTypes;

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        if (useExternalMemory)
        {
            allocInfo.pNext = &exportMemoryAllocInfo;
        }
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex =
            findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory)
            != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate image memory!");
        }

        vkBindImageMemory(device, image, imageMemory, 0);
    }

    void createOpenCLSemaphoreFromVulkanSemaphore(VkSemaphore srcSemaphore,
                                                  cl::Semaphore& semaphore)
    {
#ifdef _WIN32
        HANDLE handle = NULL;
        VkSemaphoreGetWin32HandleInfoKHR getWin32HandleInfo{};
        getWin32HandleInfo.sType =
            VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
        getWin32HandleInfo.semaphore = srcSemaphore;
        getWin32HandleInfo.handleType =
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        vkGetSemaphoreWin32HandleKHR(device, &getWin32HandleInfo, &handle);

        const cl_semaphore_properties_khr props[] = {
            CL_SEMAPHORE_TYPE_KHR,
            CL_SEMAPHORE_TYPE_BINARY_KHR,
            CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR,
            (cl_semaphore_properties_khr)handle,
            0,
        };
#elif defined(__linux__)
        int fd = 0;
        VkSemaphoreGetFdInfoKHR getFdInfo{};
        getFdInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
        getFdInfo.semaphore = srcSemaphore;
        getFdInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
        vkGetSemaphoreFdKHR(device, &getFdInfo, &fd);

        const cl_semaphore_properties_khr props[] = {
            CL_SEMAPHORE_TYPE_KHR,
            CL_SEMAPHORE_TYPE_BINARY_KHR,
            CL_SEMAPHORE_HANDLE_OPAQUE_FD_KHR,
            (cl_semaphore_properties_khr)fd,
            0,
        };
#else
            const cl_mem_properties* props = NULL;
#endif

        cl::vector<cl_semaphore_properties_khr> vprops(sizeof(props)
                                                       / sizeof(props[0]));
        std::memcpy(vprops.data(), props,
                    sizeof(cl_semaphore_properties_khr) * vprops.size());
        semaphore = cl::Semaphore(context, vprops);
    }

    void transitionImageLayout(VkImage image, VkFormat format,
                               VkImageLayout oldLayout, VkImageLayout newLayout,
                               uint32_t layers = 1)
    {

        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        //        vulkan spec: If the calling command’s VkImage parameter is of
        //        VkImageType VK_IMAGE_TYPE_3D, the baseArrayLayer and
        //        layerCount members of imageSubresource must be 0 and 1,
        //        respectively
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = layers;

        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
            && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                 && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
                 && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else
        {
            throw std::invalid_argument("unsupported layout transition!");
        }

        vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0,
                             nullptr, 0, nullptr, 1, &barrier);

        endSingleTimeCommands(commandBuffer);
    }

    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width,
                           uint32_t height)
    {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = { 0, 0, current_array_layer }; // array path
        region.imageExtent = { width, height, 1 };

        vkCmdCopyBufferToImage(commandBuffer, buffer, image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                               &region);
        endSingleTimeCommands(commandBuffer);
    }

    void transitionUniformLayout(VkBuffer buffer, VkAccessFlagBits src,
                                 VkAccessFlagBits dst)
    {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkDeviceSize bufferSize = sizeof(UniformBufferObject);
        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.pNext = nullptr;
        barrier.srcAccessMask = src; // VK_ACCESS_HOST_WRITE_BIT;
        barrier.dstAccessMask = dst; // VK_ACCESS_UNIFORM_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = buffer;
        barrier.offset = 0;
        barrier.size = bufferSize;

        VkPipelineStageFlags sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkPipelineStageFlags destinationStage =
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
            | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;

        if (src == VK_ACCESS_SHADER_READ_BIT)
        {
            sourceStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }

        vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0,
                             nullptr, 1, &barrier, 0, nullptr);

        endSingleTimeCommands(commandBuffer);
    }


    void createDescriptorPool()
    {
        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[0].descriptorCount =
            static_cast<uint32_t>(swapChainImages.size());

        poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[1].descriptorCount =
            static_cast<uint32_t>(swapChainImages.size());

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool)
            != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }

    void createDescriptorSets()
    {
        std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(),
                                                   descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount =
            static_cast<uint32_t>(swapChainImages.size());
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets.resize(swapChainImages.size());
        if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data())
            != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        for (size_t i = 0; i < swapChainImages.size(); i++)
        {
            VkDescriptorImageInfo imageInfo[(size_t)InteropTexType::IOPT_COUNT] = {0};
            imageInfo[IOPT_DISPLACEMENT].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo[IOPT_DISPLACEMENT].imageView = textureImages[IOPT_DISPLACEMENT].imageViews[i];
            imageInfo[IOPT_DISPLACEMENT].sampler = textureSampler[IOPT_DISPLACEMENT];

            imageInfo[IOPT_NORMAL_MAP].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo[IOPT_NORMAL_MAP].imageView = textureImages[IOPT_NORMAL_MAP].imageViews[i];
            imageInfo[IOPT_NORMAL_MAP].sampler = textureSampler[IOPT_NORMAL_MAP];

            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptorSets[i];
            descriptorWrites[0].dstBinding = 0u;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pImageInfo = &imageInfo[IOPT_DISPLACEMENT];

            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptorSets[i];
            descriptorWrites[0].dstBinding = 1u;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pImageInfo = &imageInfo[IOPT_NORMAL_MAP];

            descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[2].dstSet = descriptorSets[i];
            descriptorWrites[2].dstBinding = 2u;
            descriptorWrites[2].dstArrayElement = 0;
            descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[2].descriptorCount = 1;
            descriptorWrites[2].pBufferInfo = &bufferInfo;

            vkUpdateDescriptorSets(
                device, static_cast<uint32_t>(descriptorWrites.size()),
                descriptorWrites.data(), 0, nullptr);
        }
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties, VkBuffer& buffer,
                      VkDeviceMemory& bufferMemory)
    {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex =
            findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory)
            != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }


    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0; // Optional
        copyRegion.dstOffset = 0; // Optional
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    uint32_t findMemoryType(uint32_t typeFilter,
                            VkMemoryPropertyFlags properties)
    {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
        {
            if ((typeFilter & (1 << i))
                && (memProperties.memoryTypes[i].propertyFlags & properties)
                    == properties)
            {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    VkCommandBuffer beginSingleTimeCommands()
    {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer)
    {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    void createCommandBuffers()
    {
        commandBuffers.resize(swapChainFramebuffers.size());

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data())
            != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate command buffers!");
        }

        for (size_t i = 0; i < commandBuffers.size(); i++)
        {
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

            if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo)
                != VK_SUCCESS)
            {
                throw std::runtime_error(
                    "failed to begin recording command buffer!");
            }

            VkRenderPassBeginInfo renderPassInfo{};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassInfo.renderPass = renderPass;
            renderPassInfo.framebuffer = swapChainFramebuffers[i];
            renderPassInfo.renderArea.offset = { 0, 0 };
            renderPassInfo.renderArea.extent = swapChainExtent;

            VkClearValue clearColor = { { { 0.0f, 0.0f, 0.0f, 1.0f } } };
            renderPassInfo.clearValueCount = 1;
            renderPassInfo.pClearValues = &clearColor;

            vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo,
                                 VK_SUBPASS_CONTENTS_INLINE);

            vkCmdBindPipeline(commandBuffers[i],
                              VK_PIPELINE_BIND_POINT_GRAPHICS,
                              graphicsPipeline);

            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, &vertexBuffers[i], offsets);

            vkCmdBindIndexBuffer(commandBuffers[i], indexBuffers[i], 0, VK_INDEX_TYPE_UINT16);

            vkCmdBindDescriptorSets(
                commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);

            vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(inds.size()), 1, 0, 0, 0);

            vkCmdEndRenderPass(commandBuffers[i]);

            if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to record command buffer!");
            }
        }
    }

    void createSyncObjects()
    {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);

        VkExportSemaphoreCreateInfo exportSemaphoreCreateInfo{};
        exportSemaphoreCreateInfo.sType =
            VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;

#ifdef _WIN32
        exportSemaphoreCreateInfo.handleTypes =
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#elif defined(__linux__)
        exportSemaphoreCreateInfo.handleTypes =
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT; // VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
#endif

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr,
                                  &imageAvailableSemaphores[i])
                    != VK_SUCCESS
                || vkCreateSemaphore(device, &semaphoreInfo, nullptr,
                                     &renderFinishedSemaphores[i])
                    != VK_SUCCESS
                || vkCreateFence(device, &fenceInfo, nullptr,
                                 &inFlightFences[i])
                    != VK_SUCCESS)
            {
                throw std::runtime_error(
                    "failed to create synchronization objects for a frame!");
            }
        }

        if (useExternalSemaphore)
        {
            openclFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);

            semaphoreInfo.pNext = &exportSemaphoreCreateInfo;

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
            {
                if (vkCreateSemaphore(device, &semaphoreInfo, nullptr,
                                      &openclFinishedSemaphores[i])
                    != VK_SUCCESS)
                {
                    throw std::runtime_error("failed to create synchronization "
                                             "objects for interop!");
                }
            }
        }
    }

    void updateUniforms(uint32_t currentImage)
    {
        UniformBufferObject ubo = {};
        ubo.ocean_size = NX;

        // update camera related uniform
        glm::mat4 view_matrix = glm::lookAt(camera.eye, camera.eye + camera.dir, camera.up);

        float fov = glm::radians(60.0);
        float aspect = gwx / (float)gwy;
        glm::mat4 proj_matrix = glm::perspective(fov, aspect, 0.1f, 100.0f);
        proj_matrix[1][1] *= -1;

        ubo.view_proj = view_matrix * proj_matrix;

        transitionUniformLayout(uniformBuffers[currentImage],
                                VK_ACCESS_SHADER_READ_BIT,
                                VK_ACCESS_TRANSFER_WRITE_BIT);

        VkCommandBuffer commandBuffer = beginSingleTimeCommands();
        vkCmdUpdateBuffer(commandBuffer, uniformBuffers[currentImage], 0,
                          sizeof(UniformBufferObject), &ubo);
        endSingleTimeCommands(commandBuffer);

        transitionUniformLayout(uniformBuffers[currentImage],
                                VK_ACCESS_TRANSFER_WRITE_BIT,
                                VK_ACCESS_SHADER_READ_BIT);
    }

    void updateOcean(uint32_t currentImage)
    {
        updateUniforms(currentImage);

        int kernel_arg_ind=0;
        for (size_t target=0; target<IOPT_COUNT; target++)
        {
            if (useExternalMemory)
            {
                commandQueue.enqueueAcquireExternalMemObjects(
                    { (*mems[target][currentImage]) });
            }

            kernel.setArg(kernel_arg_ind, *mems[target][currentImage]);
            kernel_arg_ind++;
        }
        kernel.setArg(kernel_arg_ind, cr); kernel_arg_ind++;
        kernel.setArg(kernel_arg_ind, ci);

        cl::NDRange lws; // NullRange by default.
        if (lwx > 0 && lwy > 0)
        {
            lws = cl::NDRange{ lwx, lwy };
        }

        commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                          cl::NDRange{ gwx, gwy }, lws);

        if (useExternalMemory)
        {
            for (size_t target=0; target<IOPT_COUNT; target++)
            {
                commandQueue.enqueueReleaseExternalMemObjects(
                    { *mems[target][currentImage] });
                if (useExternalSemaphore)
                {
                    commandQueue.enqueueSignalSemaphores(
                        { signalSemaphores[currentFrame] }, {}, nullptr, nullptr);
                    commandQueue.flush();
                }
                else
                {
                    commandQueue.finish();
                }
            }
        }
        else
        {
            for (size_t target=0; target<IOPT_COUNT; target++)
            {
                size_t rowPitch = 0;
                void* pixels = commandQueue.enqueueMapImage(
                    *mems[target][currentImage], CL_TRUE, CL_MAP_READ, { 0, 0, 0 },
                    { gwx, gwy, 1 }, &rowPitch, nullptr);

                VkDeviceSize imageSize = gwx * gwy * 4;

                void* data;
                vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
                memcpy(data, pixels, static_cast<size_t>(imageSize));
                vkUnmapMemory(device, stagingBufferMemory);

                commandQueue.enqueueUnmapMemObject(*mems[target][currentImage], pixels);
                commandQueue.flush();

                transitionImageLayout(textureImages[target].images[currentImage],
                                      VK_FORMAT_R32G32B32A32_SFLOAT,
                                      VK_IMAGE_LAYOUT_UNDEFINED,
                                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                copyBufferToImage(stagingBuffer, textureImages[target].images[currentImage],
                                  static_cast<uint32_t>(gwx),
                                  static_cast<uint32_t>(gwy));
                transitionImageLayout(textureImages[target].images[currentImage],
                                      VK_FORMAT_R32G32B32A32_SFLOAT,
                                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }
        }
    }

    void drawFrame()
    {
        if (animate)
        {
            float fcr = (frame % 599) / 599.f * 2.0f * CL_M_PI_F;
            float fci = (frame % 773) / 773.f * 2.0f * CL_M_PI_F;
            cr = sinf(fcr);
            ci = sinf(fci);

            ++frame;

            auto end = std::chrono::system_clock::now();
            std::chrono::duration<float> delta = end - start;
            float elapsed_seconds = delta.count();
            if (elapsed_seconds > 2.0f)
            {
                printf("FPS: %.1f\n", (frame - startFrame) / elapsed_seconds);
                startFrame = frame;
                start = end;
            }
        }
        if (redraw)
        {
            redraw = false;
        }

        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE,
                        UINT64_MAX);

        uint32_t imageIndex;
        vkAcquireNextImageKHR(device, swapChain, UINT64_MAX,
                              imageAvailableSemaphores[currentFrame],
                              VK_NULL_HANDLE, &imageIndex);

        updateOcean(imageIndex);

        if (imagesInFlight[imageIndex] != VK_NULL_HANDLE)
        {
            vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE,
                            UINT64_MAX);
        }
        imagesInFlight[imageIndex] = inFlightFences[currentFrame];

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        std::vector<VkSemaphore> waitSemaphores;
        std::vector<VkPipelineStageFlags> waitStages;
        waitSemaphores.push_back(imageAvailableSemaphores[currentFrame]);
        waitStages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        if (useExternalMemory && useExternalSemaphore)
        {
            waitSemaphores.push_back(openclFinishedSemaphores[currentFrame]);
            waitStages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        }
        submitInfo.waitSemaphoreCount =
            static_cast<uint32_t>(waitSemaphores.size());
        submitInfo.pWaitSemaphores = waitSemaphores.data();
        submitInfo.pWaitDstStageMask = waitStages.data();

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &renderFinishedSemaphores[currentFrame];

        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo,
                          inFlightFences[currentFrame])
            != VK_SUCCESS)
        {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &renderFinishedSemaphores[currentFrame];

        VkSwapchainKHR swapChains[] = { swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;

        presentInfo.pImageIndices = &imageIndex;

        vkQueuePresentKHR(presentQueue, &presentInfo);

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void checkOpenCLExternalMemorySupport(cl::Device& device)
    {
        if (checkDeviceForExtension(device, "cl_khr_external_memory"))
        {
            printf("Device supports cl_khr_external_memory.\n");
            printf("Supported external memory handle types:\n");

            std::vector<cl::ExternalMemoryType> types = device.getInfo<
                CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR>();
            for (auto type : types)
            {
#define CASE_TO_STRING(_e)                                                     \
    case _e: printf("\t%s\n", #_e); break;
                switch (static_cast<
                        std::underlying_type<cl::ExternalMemoryType>::type>(
                    type))
                {

                    CASE_TO_STRING(CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR);
                    CASE_TO_STRING(CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR);
                    CASE_TO_STRING(
                        CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR);
                    CASE_TO_STRING(CL_EXTERNAL_MEMORY_HANDLE_D3D11_TEXTURE_KHR);
                    CASE_TO_STRING(
                        CL_EXTERNAL_MEMORY_HANDLE_D3D11_TEXTURE_KMT_KHR);
                    CASE_TO_STRING(CL_EXTERNAL_MEMORY_HANDLE_D3D12_HEAP_KHR);
                    CASE_TO_STRING(
                        CL_EXTERNAL_MEMORY_HANDLE_D3D12_RESOURCE_KHR);
                    CASE_TO_STRING(CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR);
                    default:
                        printf(
                            "Unknown cl_external_memory_handle_type_khr %04X\n",
                            (unsigned int)type);
                }
#undef CASE_TO_STRING
            }


#ifdef _WIN32
            if (std::find(types.begin(), types.end(),
                          CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR)
                != types.end())
            {
                externalMemType = CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR;
            }
            else
            {
                printf("Couldn't find a compatible external memory type "
                       "(sample supports OPAQUE_WIN32).\n");
                useExternalMemory = false;
            }
#elif defined(__linux__)
            if (std::find(types.begin(), types.end(),
                          cl::ExternalMemoryType(
                              CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR))
                != types.end())
            {
                externalMemType = CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR;
            }
            else if (std::find(types.begin(), types.end(),
                               cl::ExternalMemoryType(
                                   CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR))
                     != types.end())
            {
                externalMemType = CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR;
            }
            else
            {
                printf("Couldn't find a compatible external memory type "
                       "(sample supports DMA_BUF or OPAQUE_FD).\n");
                useExternalMemory = false;
            }
#endif
        }
        else
        {
            printf("Device does not support cl_khr_external_memory.\n");
            useExternalMemory = false;
        }
    }

    void checkOpenCLExternalSemaphoreSupport(cl::Device& device)
    {
        if (checkDeviceForExtension(device, "cl_khr_external_semaphore"))
        {
            printf("Device supports cl_khr_external_semaphore.\n");
            printf("Supported external semaphore import handle types:\n");
            std::vector<cl_external_semaphore_handle_type_khr> types =
                device.getInfo<CL_DEVICE_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR>();
            for (auto type : types)
            {
#define CASE_TO_STRING(_e)                                                     \
    case _e: printf("\t%s\n", #_e); break;
                switch (type)
                {
                    CASE_TO_STRING(CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR);
                    CASE_TO_STRING(CL_SEMAPHORE_HANDLE_OPAQUE_FD_KHR);
                    CASE_TO_STRING(CL_SEMAPHORE_HANDLE_SYNC_FD_KHR);
                    CASE_TO_STRING(CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR);
                    CASE_TO_STRING(CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KMT_KHR);
                    default:
                        printf("Unknown cl_external_semaphore_handle_type_khr "
                               "%04X\n",
                               type);
                }
#undef CASE_TO_STRING
            }
#ifdef _WIN32
            if (std::find(types.begin(), types.end(),
                          CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR)
                == types.end())
            {
                printf("Couldn't find a compatible external semaphore type "
                       "(sample supports OPAQUE_WIN32).\n");
                useExternalSemaphore = false;
            }
#elif defined(__linux__)
            if (std::find(types.begin(), types.end(),
                          CL_SEMAPHORE_HANDLE_OPAQUE_FD_KHR)
                == types.end())
            {
                printf("Couldn't find a compatible external semaphore type "
                       "(sample supports OPAQUE_FD).\n");
                useExternalSemaphore = false;
            }
#endif
        }
        else
        {
            printf("Device does not support cl_khr_external_semaphore.\n");
            useExternalSemaphore = false;
        }
    }

    VkShaderModule createShaderModule(const std::vector<char>& code)
    {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule)
            != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(
        const std::vector<VkSurfaceFormatKHR>& availableFormats)
    {
        for (const auto& availableFormat : availableFormats)
        {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM)
            {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    VkPresentModeKHR chooseSwapPresentMode(
        const std::vector<VkPresentModeKHR>& availablePresentModes)
    {
        for (const auto& availablePresentMode : availablePresentModes)
        {
            if (vsync)
            {
                if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
                {
                    return availablePresentMode;
                }
            }
            else
            {
                if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR)
                {
                    return availablePresentMode;
                }
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
    {
        if (capabilities.currentExtent.width != UINT32_MAX)
        {
            return capabilities.currentExtent;
        }
        else
        {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = { static_cast<uint32_t>(width),
                                        static_cast<uint32_t>(height) };

            actualExtent.width =
                std::max(capabilities.minImageExtent.width,
                         std::min(actualExtent.width,
                                  capabilities.maxImageExtent.width));
            actualExtent.height =
                std::max(capabilities.minImageExtent.height,
                         std::min(actualExtent.height,
                                  capabilities.maxImageExtent.height));

            return actualExtent;
        }
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
    {
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface,
                                                  &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount,
                                             nullptr);

        if (formatCount != 0)
        {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount,
                                                 details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface,
                                                  &presentModeCount, nullptr);

        if (presentModeCount != 0)
        {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(
                device, surface, &presentModeCount,
                details.presentModes.data());
        }

        return details;
    }

    bool isDeviceSuitable(VkPhysicalDevice device)
    {
        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionsSupported)
        {
            SwapChainSupportDetails swapChainSupport =
                querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty()
                && !swapChainSupport.presentModes.empty();
        }

        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device)
    {

        VkPhysicalDeviceProperties pProperties;
        vkGetPhysicalDeviceProperties(device, &pProperties);

        if (std::string(pProperties.deviceName).find("Intel")
            != std::string::npos)
            return false;

        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                             nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                             availableExtensions.data());

        auto extensions = getRequiredDeviceExtensions();
        std::set<std::string> requiredExtensions(extensions.begin(),
                                                 extensions.end());

        for (const auto& extension : availableExtensions)
        {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
    {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                                 nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                                 queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies)
        {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface,
                                                 &presentSupport);

            if (presentSupport)
            {
                indices.presentFamily = i;
            }

            if (indices.isComplete())
            {
                break;
            }

            i++;
        }

        return indices;
    }

    std::vector<const char*> getRequiredExtensions()
    {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(
            glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (useExternalMemory || useExternalSemaphore)
        {
            extensions.push_back(
                VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        }
        if (useExternalMemory)
        {
            extensions.push_back(
                VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
        }
        if (useExternalSemaphore)
        {
            extensions.push_back(
                VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
        }
        if (enableValidationLayers)
        {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    std::vector<const char*> getRequiredDeviceExtensions()
    {
        std::vector<const char*> extensions(deviceExtensions);

        if (useExternalMemory)
        {
            extensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
#ifdef _WIN32
            extensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
#elif defined(__linux__)
            extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
//            if (externalMemType == CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR) {
//                extensions.push_back(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);
//            }
#endif
        }
        if (useExternalSemaphore)
        {
            extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
#ifdef _WIN32
            extensions.push_back(
                VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#elif defined(__linux__)
            extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif
        }

        return extensions;
    }

    bool checkValidationLayerSupport()
    {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers)
        {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers)
            {
                if (strcmp(layerName, layerProperties.layerName) == 0)
                {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound)
            {
                return false;
            }
        }

        return true;
    }

    static std::vector<char> readFile(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open())
        {
            throw std::runtime_error("failed to open file!");
        }

        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL
    debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                  VkDebugUtilsMessageTypeFlagsEXT messageType,
                  const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                  void* pUserData)
    {
        fprintf(stderr, "validation layer: %s\n", pCallbackData->pMessage);

        return VK_FALSE;
    }

    static void keyboard(GLFWwindow* window, int key, int scancode, int action,
                         int mods)
    {
        auto app = (OceanApplication*)glfwGetWindowUserPointer(window);
        app->keyboard(key, scancode, action, mods);
    }

    static void mouse_event(GLFWwindow* window, int button, int action, int mods) {
        auto app = (OceanApplication*)glfwGetWindowUserPointer(window);
        app->mouse_event(button, action, mods);
    }

    static void mouse_pos(GLFWwindow* window, double pX, double pY)
    {
        auto app = (OceanApplication*)glfwGetWindowUserPointer(window);
        app->mouse_pos(pX, pY);
    }
};

} // anonymous namespace

int main(int argc, char** argv)
{
    OceanApplication app;

    try
    {
        app.run(argc, argv);
    } catch (const std::exception& e)
    {
        fprintf(stderr, "%s\n", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
