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
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include <math.h>

// GLM includes
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


#define DRAG_SPEED_FAC 0.2f
#define ROLL_SPEED_FAC 8.f


//Plan prac:
//    1) dodaj do projektu juliavk dla porównania - DONE
//    2) teksture julia nakladaj na siatke zeby widziec efekt obracania - DONE
//    3) rozpoczynamy prace nad symulacja w OpenCL
//    4) adaptacja shaderow do symulacji oceanu
//         -displacement
//         -normalne
//         -piana
//    5) dodanie trybu wireframe
//    6) dodanie oswietlenia do sceny
//    7) adaptacja do srodowiska SDK
//        -przeniesienie kerneli do plików .kr
//    8) readme, szlify, PR


namespace {

const char *IGetErrorString(int clErrorCode)
{
    switch (clErrorCode)
    {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
            return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case CL_COMPILE_PROGRAM_FAILURE: return "CL_COMPILE_PROGRAM_FAILURE";
        case CL_LINKER_NOT_AVAILABLE: return "CL_LINKER_NOT_AVAILABLE";
        case CL_LINK_PROGRAM_FAILURE: return "CL_LINK_PROGRAM_FAILURE";
        case CL_DEVICE_PARTITION_FAILED: return "CL_DEVICE_PARTITION_FAILED";
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
            return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:
            return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_INVALID_PROPERTY: return "CL_INVALID_PROPERTY";
        case CL_INVALID_IMAGE_DESCRIPTOR: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case CL_INVALID_COMPILER_OPTIONS: return "CL_INVALID_COMPILER_OPTIONS";
        case CL_INVALID_LINKER_OPTIONS: return "CL_INVALID_LINKER_OPTIONS";
        case CL_INVALID_DEVICE_PARTITION_COUNT:
            return "CL_INVALID_DEVICE_PARTITION_COUNT";
        case CL_INVALID_PIPE_SIZE: return "CL_INVALID_PIPE_SIZE";
        case CL_INVALID_DEVICE_QUEUE: return "CL_INVALID_DEVICE_QUEUE";
        case CL_INVALID_SPEC_ID: return "CL_INVALID_SPEC_ID";
        case CL_MAX_SIZE_RESTRICTION_EXCEEDED:
            return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
        default: return "(unknown)";
    }
}

#define print_error(errCode, msg)                                              \
    printf("ERROR: %s! (%s from %s:%d)\n", msg, IGetErrorString(errCode),   \
              __FILE__, __LINE__);

#define test_error(errCode, msg)                                               \
    {                                                                          \
        auto errCodeResult = errCode;                                          \
        if (errCodeResult != CL_SUCCESS)                                       \
        {                                                                      \
            print_error(errCodeResult, msg);                                   \
            return errCode;                                                    \
        }                                                                      \
    }

uint32_t reverse_bits(uint32_t n, uint32_t log_2_N) {
    uint32_t r = 0;
    for (int j = 0; j < log_2_N; j++)
	{
		r = (r << 1) + (n & 1);
		n >>= 1;
	}
    return r;
}

const char twiddle_kernel_str[] =
R"CLC(
constant float PI = 3.14159265359;

typedef float2 complex;

__kernel void generate( int resolution, global int * bit_reversed, write_only image2d_t dst )
{
    int2 uv = (int2)((int)get_global_id(0), (int)get_global_id(1));
    float k = fmod(uv.y * ((float)(resolution) / pow(2.f, (float)(uv.x+1))), resolution);
    complex twiddle = (complex)( cos(2.0*PI*k/(float)(resolution)), sin(2.0*PI*k/(float)(resolution)));

    int butterflyspan = (int)(pow(2.f, (float)(uv.x)));
    int butterflywing;

    if (fmod(uv.y, pow(2.f, (float)(uv.x + 1))) < pow(2.f, (float)(uv.x)))
        butterflywing = 1;
    else
        butterflywing = 0;

    // first stage, bit reversed indices
    if (uv.x == 0) {
        // top butterfly wing
        if (butterflywing == 1)
            write_imagef(dst, uv, (float4)(twiddle.x, twiddle.y, bit_reversed[(int)(uv.y)], bit_reversed[(int)(uv.y + 1)]));
        // bot butterfly wing
        else
            write_imagef(dst, uv, (float4)(twiddle.x, twiddle.y, bit_reversed[(int)(uv.y - 1)], bit_reversed[(int)(uv.y)]));
    }
    // second to log2(resolution) stage
    else {
        // top butterfly wing
        if (butterflywing == 1)
            write_imagef(dst, uv, (float4)(twiddle.x, twiddle.y, uv.y, uv.y + butterflyspan));
        // bot butterfly wing
        else
            write_imagef(dst, uv, (float4)(twiddle.x, twiddle.y, uv.y - butterflyspan, uv.y));
    }
}
)CLC";

const char init_spectrum_kernel_str[] =
R"CLC(
constant float PI = 3.14159265359f;
constant sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;
constant float GRAVITY = 9.81f;

float4 gaussRND(float4 rnd)
{
	float u0 = 2.0*PI*rnd.x;
	float v0 = sqrt(-2.0 * log(rnd.y));
	float u1 = 2.0*PI*rnd.z;
	float v1 = sqrt(-2.0 * log(rnd.w));

	float4 ret = (float4)(v0 * cos(u0), v0 * sin(u0), v1 * cos(u1), v1 * sin(u1));
	return ret;
}

float SuppressionFactor(float suppress_length, float k_magnitude_sq)
{
    return exp(-k_magnitude_sq * suppress_length * suppress_length);
}

float PhillipsSpectrum(float2 k, float k_magnitude_sq, float l_phillips, float4 params)
{
    return params.z
            * ((exp(-1.0 / (k_magnitude_sq * l_phillips * l_phillips))
            * pow(dot(normalize(k), params.xy), 2))
            * SuppressionFactor(params.w, k_magnitude_sq))
            / (k_magnitude_sq * k_magnitude_sq);
}

// patch_info.x - ocean patch size
// patch_info.y - ocean texture unified resolution
// params.x - wind x
// params.y - wind.y
// params.z - amplitude
// params.w - capillar supress factor

__kernel void init_spectrum( int2 patch_info, float4 params, read_only image2d_t noise, write_only image2d_t dst )
{
    int2 uv = (int2)((int)get_global_id(0), (int)get_global_id(1));
    int res = patch_info.y;

    float2 fuv = (float2)(get_global_id(0), get_global_id(1)) - (float)(res)/2.f;
    float2 k = (2.f * PI * fuv) / patch_info.x;
    float k_mag = length(k);

    if (k_mag < 0.00001) k_mag = 0.00001;

    float wind_speed = length((float2)(params.x, params.y));
    float4 params_n = params;
    params_n.xy = (float2)(params.x/wind_speed, params.y/wind_speed);
    float l_phillips = (wind_speed * wind_speed) / GRAVITY;
    float4 rnd = clamp(read_imagef(noise, sampler, uv), 0.001f, 1.f);

#if 1
    float magSq = k_mag * k_mag;
    float h0k = clamp(sqrt((params.z/(magSq*magSq)) * pow(dot(normalize(k), params_n.xy), 2.f) *
                exp(-(1.0/(magSq * l_phillips * l_phillips))) * exp(-magSq*pow(params.w, 2.f)))/ sqrt(2.0), -4000.0, 4000.0);
    float h0minusk = clamp(sqrt((params.z/(magSq*magSq)) * pow(dot(normalize(-k), params_n.xy), 2.f) *
                exp(-(1.0/(magSq * l_phillips * l_phillips))) * exp(-magSq*pow(params.w, 2.f)))/ sqrt(2.0), -4000.0, 4000.0);
    float4 gauss_random = gaussRND(rnd);
    write_imagef(dst, uv, (float4)(gauss_random.xy*h0k, gauss_random.zw*h0minusk));
#else
    // path for pre-generated Gaussian randoms
    float h0_k = clamp(sqrt(PhillipsSpectrum(k, k_mag * k_mag, l_phillips, params_n) / 2.0), -4000.0, 4000.0);
    float h0_minus_k = clamp(sqrt(PhillipsSpectrum(-k, k_mag * k_mag, l_phillips, params_n) / 2.0), -4000.0, 4000.0);
    write_imagef(dst, uv, (float4)(rnd.xy * h0_k, rnd.zw * h0_minus_k));
#endif
}
)CLC";

const char time_spectrum_kernel_str[] =
    R"CLC(

constant float PI = 3.14159265359;
constant float G = 9.81;
constant sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;

typedef float2 complex;

complex mul(complex c0, complex c1)
{
    complex c;
    c.x = c0.x * c1.x - c0.y * c1.y;
    c.y = c0.x * c1.y + c0.y * c1.x;
    return c;
}

complex add(complex c0, complex c1)
{
    complex c;
    c.x = c0.x + c1.x;
    c.y = c0.y + c1.y;
    return c;
}

complex conj(complex c)
{
    complex c_conj = (complex)(c.x, -c.y);
    return c_conj;
}

__kernel void spectrum( float dt, int2 patch_info,
    read_only image2d_t src, write_only image2d_t dst_x,
    write_only image2d_t dst_y, write_only image2d_t dst_z )
{
    int2 uv = (int2)((int)get_global_id(0), (int)get_global_id(1));
    int res = patch_info.y;
    float2 wave_vec = (float2)(uv.x - res / 2.f, uv.y - res / 2.f);
    float2 k = (2.f * PI * wave_vec) / patch_info.x;
    float k_mag = length(k);

    float w = sqrt(G * k_mag);

    float4 h0k = read_imagef(src, sampler, uv);
    complex fourier_amp = (complex)(h0k.x, h0k.y);
    complex fourier_amp_conj = conj((complex)(h0k.z, h0k.w));

    float cos_wt = cos(w*dt);
    float sin_wt = sin(w*dt);

    // euler formula
    complex exp_iwt = (complex)(cos_wt, sin_wt);
    complex exp_iwt_inv = (complex)(cos_wt, -sin_wt);

    // dy
    complex h_k_t_dy = add(mul(fourier_amp, exp_iwt), (mul(fourier_amp_conj, exp_iwt_inv)));

    // dx
    complex dx = (complex)(0.0,-k.x/k_mag);
    complex h_k_t_dx = mul(dx, h_k_t_dy);

    // dz
    complex dz = (complex)(0.0,-k.y/k_mag);
    complex h_k_t_dz = mul(dz, h_k_t_dy);

    // amplitude
    write_imagef(dst_y, uv, (float4)(h_k_t_dy.x, h_k_t_dy.y, 0, 1));

    // choppiness
    write_imagef(dst_x, uv, (float4)(h_k_t_dx.x, h_k_t_dx.y, 0, 1));
    write_imagef(dst_z, uv, (float4)(h_k_t_dz.x, h_k_t_dz.y, 0, 1));
}
)CLC";

const char fft_kernel_str[] =

R"CLC(
constant sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;

typedef float2 complex;

complex mul(complex c0, complex c1)
{
    complex c;
    c.x = c0.x * c1.x - c0.y * c1.y;
    c.y = c0.x * c1.y + c0.y * c1.x;
    return c;
}

complex add(complex c0, complex c1)
{
    complex c;
    c.x = c0.x + c1.x;
    c.y = c0.y + c1.y;
    return c;
}

// mode.x - 0-horizontal, 1-vertical
// mode.y - subsequent count

__kernel void fft_1D( int2 mode, int2 patch_info,
    read_only image2d_t twiddle, read_only image2d_t src, write_only image2d_t dst )
{
    int2 uv = (int2)((int)get_global_id(0), (int)get_global_id(1));

    int2 data_coords = (int2)(mode.y, uv.x * (1-mode.x) + uv.y * mode.x);
    float4 data = read_imagef(twiddle, sampler, data_coords);


    work_group_barrier(CLK_IMAGE_MEM_FENCE);


    int2 pp_coords0 = (int2)(data.z, uv.y) * (1-mode.x) + (int2)(uv.x, data.z) * mode.x;
    float2 p = read_imagef(src, sampler, pp_coords0).rg;

    int2 pp_coords1 = (int2)(data.w, uv.y) * (1-mode.x) + (int2)(uv.x, data.w) * mode.x;
    float2 q = read_imagef(src, sampler, pp_coords1).rg;

    float2 w = (float2)(data.x, data.y);

    //Butterfly operation
    complex H = add(p,mul(w,q));

    write_imagef(dst, uv, (float4)(H.x, H.y, 0, 1));
}
)CLC";


const char inversion_kernel_str[] =
    R"CLC(
constant sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;

__kernel void inversion( int2 patch_info, read_only image2d_t src0,
    read_only image2d_t src1, read_only image2d_t src2, write_only image2d_t dst )
{
    int2 uv = (int2)((int)get_global_id(0), (int)get_global_id(1));
    int res2 = patch_info.y * patch_info.y;

#if 0
    float perms[] = {1.0, -1.0};
    int index = (uv.x + uv.y) % 2;
    float perm = perms[index];

    float x = read_imagef(src0, sampler, uv).r;
    float y = read_imagef(src1, sampler, uv).r;
    float z = read_imagef(src2, sampler, uv).r;

    write_imagef(dst, uv, (float4)(perm*(x/res2),
                                   perm*(y/res2),
                                   perm*(z/res2), 1));
#else
    // for some reason permutation destroys the final image
    float x = read_imagef(src0, sampler, uv).r;
    float y = read_imagef(src1, sampler, uv).r;
    float z = read_imagef(src2, sampler, uv).r;

    write_imagef(dst, uv, (float4)(x/res2, y/res2, z/res2, 1));
#endif
}
)CLC";

const char normals_kernel_str[] =
    R"CLC(
constant sampler_t sampler = CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR | CLK_NORMALIZED_COORDS_TRUE;
// scale_fac.x - choppines
// scale_fac.y - altitude scale
__kernel void normals( int2 patch_info, float2 scale_fac,
     read_only image2d_t src, write_only image2d_t dst )
{
    int2 uv = (int2)((int)get_global_id(0), (int)get_global_id(1));
    float2 fuv = convert_float2(uv) / patch_info.y;

    float texel = 1.f / patch_info.y;
    float texel_size = patch_info.x * texel;

    float3 dxyz_c = read_imagef(src, sampler, fuv).rgb;
    float3 dxyz_r = read_imagef(src, sampler, (float2)(fuv.x + texel, fuv.y)).rgb;
    float3 dxyz_t = read_imagef(src, sampler, (float2)(fuv.x, fuv.y + texel)).rgb;

    float3 center = (float3)(dxyz_c.x*scale_fac.x,
                             dxyz_c.z*scale_fac.x,
                             dxyz_c.y*scale_fac.y);
    float3 right = (float3)(dxyz_r.x*scale_fac.x+texel_size,
                            dxyz_r.z*scale_fac.x,
                            dxyz_r.y*scale_fac.y) - center;
    float3 top = (float3)(dxyz_t.x*scale_fac.x,
                          dxyz_t.z*scale_fac.x+texel_size,
                          dxyz_t.y*scale_fac.y) - center;
    float3 cprod = cross(normalize(right), normalize(top));
    write_imagef(dst, uv, (float4)(normalize(cprod), 1.f));
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
    alignas(4) glm::mat4    view_mat;
    alignas(4) glm::mat4    proj_mat;
    alignas(4) glm::vec3    sun_dir=glm::normalize(glm::vec3(0.f, 1.f, 1.f));
    alignas(4) std::float_t choppiness=1.f;
    alignas(4) std::float_t alt_scale=1.f;
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
    glm::vec3 eye = glm::vec3(0.0f, 0.0f, 20.0f);
    glm::vec3 dir = glm::normalize(glm::vec3(0.0f, 1.0f, -0.5f));  // should be in sync with pitch, eg (0,1,-1)->45degrees
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 rvec = glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec2 begin = glm::vec2(-1.0f, -1.0f);
    float yaw = 0.0f;
    float pitch = 71.565f;
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

    // mesh patch size - assume uniform x/y
    size_t ocean_grid_size = 256;

    // mesh patch spacing
    float mesh_spacing = 2.f;

    // ocean texture size - assume uniform x/y
    size_t ocean_tex_size = 256;

    size_t group_size = 16;

    size_t window_width = 1024;
    size_t window_height = 1024;


    bool animate = false;
    bool redraw = false;

    // ocean parameters changed - rebuild initial spectrum resources
    bool    changed = true;
    bool    twiddle_factors_init = true;

    // ocean in-factors
    float   wind_magnitude = 160.f;
    float   wind_angle = 45.f;
    float   choppiness = 16.f;
    float   alt_scale = 8.f;

    float   amplitude=16.f;
    float   supress_factor=0.1f;

    // env factors
    int     sun_elevation = 0;
    int     sun_azimuth = 90;
    bool    wireframe_mode = false;

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
    VkPipeline wireframePipeline;

    VkCommandPool commandPool;

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    bool linearImages = false;
    bool deviceLocalImages = true;

    // Only displacement and normal map images must be shared between OCL and vulkan

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

    // vulkan-opencl interop resources
    std::array<TextureOCL, IOPT_COUNT> textureImages;

    // Ocean grid vertices and related buffers
    std::vector<Vertex> verts;
    std::vector<VkBuffer> vertexBuffers;
    std::vector<VkDeviceMemory> vertexBufferMemories;

    std::vector <std::uint32_t> inds;
    struct IndexBuffer {
        std::vector<VkBuffer> buffers;
        std::vector<VkDeviceMemory> bufferMemories;
    };
    std::vector<IndexBuffer> indexBuffers;

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

    bool useExternalMemory = true;
    bool useExternalSemaphore = true;

    // OpenCL resources
    cl_external_memory_handle_type_khr externalMemType = 0;

    cl::Context context;
    cl::CommandQueue commandQueue;



    // generates twiddle factors kernel
    cl::Kernel twiddle_kernel;

    // initial spectrum kernel
    cl::Kernel init_spectrum_kernel;

    // Fourier components image kernel
    cl::Kernel time_spectrum_kernel;

    // FFT kernel
    cl::Kernel fft_kernel;

    // inversion kernel
    cl::Kernel inversion_kernel;

    // building normals kernel
    cl::Kernel normals_kernel;


    // FFT intermediate computation storages without vulkan iteroperability
    std::unique_ptr<cl::Image2D> dxyz_coef_mem[3];
    std::unique_ptr<cl::Image2D> hkt_pong_mem;
    std::unique_ptr<cl::Image2D> twiddle_factors_mem;
    std::unique_ptr<cl::Image2D> h0k_mem;
    std::unique_ptr<cl::Image2D> noise_mem;

    size_t ocl_max_img2d_width;
    cl_ulong ocl_max_alloc_size, ocl_mem_size;

    // opencl-vulkan iteroperability resources
    // final computation result with displacements and normal map,
    // needs to follow swap-chain scheme
    std::array<std::vector<std::unique_ptr<cl::Image2D>>, IOPT_COUNT> mems;
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
            "", "window_width", "Window width", window_width, &window_width);
        op.add<popl::Value<size_t>>(
            "", "window_height", "Window height", window_height, &window_height);

        op.add<popl::Value<size_t>>(
            "", "ocean_tex_size", "Ocean patch size (uniform) AKA global work size", ocean_tex_size, &ocean_tex_size);

        op.add<popl::Value<size_t>>("", "group_size", "Local Work Size (uniform)", group_size, &group_size);
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

        window = glfwCreateWindow((int)window_width, (int)window_height, "Julia Set with Vulkan",
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

        auto build_opencl_kernel = [&](const char * source, cl::Kernel & kernel, const char * name)
        {
            cl::Program program{ context, source };

            try
            {
                program.build();
            }
            catch (const cl::BuildError& e)
            {
                auto bl = e.getBuildLog();
                printf("Build OpenCL %s kernel error: \n", name);
                for (auto elem : bl) printf("%s\n", elem.second.c_str());
                exit(1);
            }

            kernel = cl::Kernel{ program, name };
        };


        build_opencl_kernel(twiddle_kernel_str, twiddle_kernel, "generate");
        build_opencl_kernel(init_spectrum_kernel_str, init_spectrum_kernel, "init_spectrum");
        build_opencl_kernel(time_spectrum_kernel_str, time_spectrum_kernel, "spectrum");
        build_opencl_kernel(fft_kernel_str, fft_kernel, "fft_1D");
        build_opencl_kernel(inversion_kernel_str, inversion_kernel, "inversion");
        build_opencl_kernel(normals_kernel_str, normals_kernel, "normals");
    }

    void initOpenCLMems()
    {
        // init intermediate opencl resources
        try
        {
            {
                std::vector<cl_float4> phase_array(ocean_tex_size
                                                    * ocean_tex_size);
                std::random_device dev;
                std::mt19937 rng(dev());
                std::uniform_real_distribution<float> dist(0.f, 1.f);

                for (size_t i = 0; i < phase_array.size(); ++i)
                    phase_array[i] = { dist(rng), dist(rng), dist(rng), dist(rng) };

                noise_mem = std::make_unique<cl::Image2D>(
                    context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                    cl::ImageFormat(CL_RGBA, CL_FLOAT), ocean_tex_size, ocean_tex_size,
                    0, phase_array.data());
            }


            hkt_pong_mem = std::make_unique<cl::Image2D>(
                context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT),
                ocean_tex_size, ocean_tex_size);

            dxyz_coef_mem[0] = std::make_unique<cl::Image2D>(
                context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT),
                ocean_tex_size, ocean_tex_size);

            dxyz_coef_mem[1] = std::make_unique<cl::Image2D>(
                context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT),
                ocean_tex_size, ocean_tex_size);

            dxyz_coef_mem[2] = std::make_unique<cl::Image2D>(
                context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT),
                ocean_tex_size, ocean_tex_size);

            h0k_mem = std::make_unique<cl::Image2D>(
                context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT),
                ocean_tex_size, ocean_tex_size);

            int log_2_N = log((float)ocean_tex_size) / log(2.f);

            twiddle_factors_mem = std::make_unique<cl::Image2D>(
                        context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT),
                        log_2_N, ocean_tex_size);


            for (size_t target = 0; target < IOPT_COUNT; target++)
            {
                mems[target].resize(swapChainImages.size());

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
                            textureImages[target].imageMemories[i];
                        getWin32HandleInfo.handleType =
                            VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
                        vkGetMemoryWin32HandleKHR(
                            device, &getWin32HandleInfo, &handle);

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
                            textureImages[target].imageMemories
                                [i]; // textureImageMemories[i];
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
                                    sizeof(cl_mem_properties)
                                        * vprops.size());

                        mems[target][i].reset(new cl::Image2D(
                            context, vprops, CL_MEM_READ_WRITE,
                            cl::ImageFormat(CL_RGBA, CL_FLOAT),
                            ocean_tex_size, ocean_tex_size));
                    }
                    else
                    {
                        mems[target][i].reset(new cl::Image2D{
                            context, CL_MEM_READ_WRITE,
                            cl::ImageFormat{ CL_RGBA, CL_FLOAT },
                            ocean_tex_size, ocean_tex_size });
                    }
                }
            }
        } catch (const cl::Error& e)
        {
            printf("initOpenCLMems: OpenCL %s image error: %s\n", e.what(),
                   IGetErrorString(e.err()));
            exit(1);
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
        glfwSetScrollCallback(window, mouse_roll);

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

                case GLFW_KEY_A: wind_magnitude += 1.f; changed = true; break;
                case GLFW_KEY_Z: wind_magnitude -= 1.f; changed = true; break;

                case GLFW_KEY_S: wind_angle += 1.f; changed = true; break;
                case GLFW_KEY_X: wind_angle -= 1.f; changed = true; break;

                case GLFW_KEY_D: amplitude += 0.5f; changed = true; break;
                case GLFW_KEY_C: amplitude -= 0.5f; changed = true; break;

                case GLFW_KEY_F: choppiness += 0.5f; break;
                case GLFW_KEY_V: choppiness -= 0.5f; break;

                case GLFW_KEY_G: alt_scale += 0.5f; break;
                case GLFW_KEY_B: alt_scale -= 0.5f; break;

                case GLFW_KEY_W:
                    wireframe_mode = !wireframe_mode;
                    createCommandBuffers();
                    break;
            }
        }
    }

    void mouse_event(int button, int action, int mods) {
        double x, y;
        glfwGetCursorPos(window, &x, &y);
        switch (action) {
            case 0:
                // Button Up
                camera.drag = false;
                break;
            case 1:
                // Button Down
                camera.drag = true;
                camera.begin = glm::vec2(x, y);
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

        glm::quat yaw (glm::cos(glm::radians(camera.yaw/2)), glm::vec3(0, 0, 1) * glm::sin(glm::radians(camera.yaw/2)));
        glm::quat pitch (glm::cos(glm::radians(camera.pitch/2)), glm::vec3(1, 0, 0) * glm::sin(glm::radians(camera.pitch/2)));
        glm::mat3 rot_mat ( yaw * pitch );
        glm::vec3 dir = rot_mat * glm::vec3(0, 0, -1);

        camera.dir = glm::normalize(dir);
        camera.rvec = glm::normalize(glm::cross(camera.dir, glm::vec3(0, 0, 1)));
        camera.up = glm::normalize(glm::cross(camera.rvec, camera.dir));
    }

    void mouse_roll(double offset_x, double offset_y)
    {
        camera.eye += camera.dir * (float)offset_y * ROLL_SPEED_FAC;
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
        vkDestroyPipeline(device, wireframePipeline, nullptr);
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
        for (auto ind_buffer : indexBuffers)
        {
            for(auto buffer : ind_buffer.buffers)
            {
                vkDestroyBuffer(device, buffer, nullptr);
            }
        }

        for (auto ind_buffer : indexBuffers)
        {
            for (auto bufferMemory : ind_buffer.bufferMemories)
            {
                vkFreeMemory(device, bufferMemory, nullptr);
            }
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
        uniformLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT;

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

        rasterizer.polygonMode = VK_POLYGON_MODE_LINE;
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                      nullptr, &wireframePipeline)
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

        int iCXY = ( ocean_grid_size + 1 ) * ( ocean_grid_size + 1 );
        verts.resize(iCXY);

        // Initialize vertices and normals to default (row, column, 0) and (0,
        // 0, 1) This step is not really neccessary (its just in case something
        // went wrong) Verts and normals will be updated with wave height field
        // every frame
        cl_float dfY = -0.5 * (ocean_grid_size * mesh_spacing), dfBaseX = -0.5 * (ocean_grid_size * mesh_spacing);
        cl_float tx=0.f, ty=0.f, dtx = 1.f / ocean_grid_size, dty= 1.f / ocean_grid_size;
        for (int iBase = 0, iY = 0; iY <= ocean_grid_size; iY++, iBase += ocean_grid_size + 1)
        {
            tx=0.f;
            double dfX = dfBaseX;
            for (int iX = 0; iX <= ocean_grid_size; iX++)
            {
                verts[iBase + iX].pos = glm::vec3(dfX, dfY, 0.0);
                verts[iBase + iX].tc = glm::vec2(tx, ty);
                tx += dtx;
                dfX += mesh_spacing;
            }
            dfY += mesh_spacing;
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

    void createIndexBuffers()
    {
        indexBuffers.resize(ocean_grid_size);
        // Add Tri Strip primitve sets
        inds.resize((ocean_grid_size + 1) * 2);

        VkDeviceSize bufferSize = sizeof(inds[0]) * inds.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                         | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingBufferMemory);

        // Each tri strip draws one row of NX quads
        for (int iBaseTo, iBaseFrom = 0, iY = 0; iY < ocean_grid_size;
             iY++, iBaseFrom = iBaseTo)
        {
            iBaseTo = iBaseFrom + ocean_grid_size + 1;
            for (int iX = 0; iX <= ocean_grid_size; iX++)
            {
                inds[iX * 2 + 0] = iBaseFrom + iX;
                inds[iX * 2 + 1] = iBaseTo + iX;
            }

            indexBuffers[iY].buffers.resize(swapChainImages.size());
            indexBuffers[iY].bufferMemories.resize(swapChainImages.size());

            void* data;
            vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
            memcpy(data, inds.data(), (size_t)bufferSize);
            vkUnmapMemory(device, stagingBufferMemory);

            for (size_t i = 0; i < swapChainImages.size(); i++)
            {

                // create local memory buffer
                createBuffer(bufferSize,
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                 | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                             indexBuffers[iY].buffers[i],
                             indexBuffers[iY].bufferMemories[i]);

                copyBuffer(stagingBuffer, indexBuffers[iY].buffers[i],
                           bufferSize);
            }
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

        uint32_t texWidth = static_cast<uint32_t>(ocean_tex_size);
        uint32_t texHeight = static_cast<uint32_t>(ocean_tex_size);

        VkDeviceSize imageSize = texWidth * texHeight * 4 * sizeof(float);

        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                         | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingBufferMemory);

        for (size_t target = 0; target < textureImages.size(); target++ )
        {
            textureImages[target].images.resize(swapChainImages.size());
            textureImages[target].imageMemories.resize(swapChainImages.size());

            for (size_t i = 0; i < swapChainImages.size(); i++)
            {
                createShareableImage(
                    texWidth, texHeight, VK_FORMAT_R32G32B32A32_SFLOAT, tiling,
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    properties, textureImages[target].images[i], textureImages[target].imageMemories[i]
                );
                if (useExternalMemory)
                {
                    transitionImageLayout(textureImages[target].images[i],
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
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;





//        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
//        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
//        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;




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
            destinationStage =
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT /*|
                    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT*/;
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
        region.imageOffset = { 0, 0, 0 };
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
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT /*| VK_PIPELINE_STAGE_VERTEX_SHADER_BIT*/
            /*| VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR*/;

        if (src == VK_ACCESS_SHADER_READ_BIT)
        {
            sourceStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT /*| VK_PIPELINE_STAGE_VERTEX_SHADER_BIT*/
                /*| VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR*/;
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

            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            std::array<VkWriteDescriptorSet, IOPT_COUNT + 1> descriptorWrites{};

            for (cl_int target = 0; target < IOPT_COUNT; target++)
            {
                imageInfo[target].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                imageInfo[target].imageView = textureImages[target].imageViews[i];
                imageInfo[target].sampler = textureSampler[target];

                descriptorWrites[target].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites[target].dstSet = descriptorSets[i];
                descriptorWrites[target].dstBinding = target;
                descriptorWrites[target].dstArrayElement = 0;
                descriptorWrites[target].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                descriptorWrites[target].descriptorCount = 1;
                descriptorWrites[target].pImageInfo = &imageInfo[target];
            }

            descriptorWrites[IOPT_COUNT].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[IOPT_COUNT].dstSet = descriptorSets[i];
            descriptorWrites[IOPT_COUNT].dstBinding = IOPT_COUNT;
            descriptorWrites[IOPT_COUNT].dstArrayElement = 0;
            descriptorWrites[IOPT_COUNT].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[IOPT_COUNT].descriptorCount = 1;
            descriptorWrites[IOPT_COUNT].pBufferInfo = &bufferInfo;

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
                              wireframe_mode ? wireframePipeline : graphicsPipeline);

            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, &vertexBuffers[i], offsets);

            vkCmdBindDescriptorSets(
                commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);

            for ( auto ind_buffer : indexBuffers )
            {
                vkCmdBindIndexBuffer(commandBuffers[i], ind_buffer.buffers[i], 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(inds.size()), 1, 0, 0, 0);
            }

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
        ubo.choppiness = choppiness;
        ubo.alt_scale = alt_scale;

        // update camera related uniform
        glm::mat4 view_matrix = glm::lookAt(camera.eye, camera.eye + camera.dir, camera.up);

        float fov = glm::radians(60.0);
        float aspect = (float)window_width / window_height;
        glm::mat4 proj_matrix = glm::perspective(fov, aspect, 1.f, 2.f * ocean_grid_size * mesh_spacing);
        proj_matrix[1][1] *= -1;

        ubo.view_mat = view_matrix;
        ubo.proj_mat = proj_matrix;

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

    void updateSpectrum(uint32_t currentImage)
    {
        cl_int2 patch = cl_int2{(int)(ocean_grid_size * mesh_spacing), (int)ocean_tex_size};
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<float> delta = end - start;

        // time factor of ocean animation
        float elapsed = delta.count();

        cl::NDRange lws; // NullRange by default.
        if (group_size > 0)
        {
            lws = cl::NDRange{ group_size, group_size };
        }

        if (twiddle_factors_init)
        {
            try
            {
                size_t log_2_N = (size_t)(log(ocean_tex_size) / log(2.f));

                /// Prepare vector of values to extract results
                std::vector<cl_int> v(ocean_tex_size);
                for (int i = 0; i < ocean_tex_size; i++)
                {
                    int x = reverse_bits(i, log_2_N);
                    v[i] = x;
                }

                /// Initialize device-side storage
                cl::Buffer bit_reversed_inds_mem{
                    context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    sizeof(cl_int) * v.size(), v.data()
                };

                twiddle_kernel.setArg(0, cl_int(ocean_tex_size));
                twiddle_kernel.setArg(1, bit_reversed_inds_mem);
                twiddle_kernel.setArg(2, *twiddle_factors_mem);

                commandQueue.enqueueNDRangeKernel(
                    twiddle_kernel, cl::NullRange,
                    cl::NDRange{ log_2_N, ocean_tex_size },
                    cl::NDRange{ 1, 16 });
                twiddle_factors_init = false;
            } catch (const cl::Error& e)
            {
                printf("twiddle indices: OpenCL %s kernel error: %s\n", e.what(), IGetErrorString(e.err()));
                exit(1);
            }
        }

        // change of some ocean's parameters requires to rebuild initial spectrum image
        if (changed)
        {
            try
            {
                float wind_angle_rad = glm::radians(wind_angle);
                cl_float4 params = cl_float4 {
                        wind_magnitude * glm::cos(wind_angle_rad),
                        wind_magnitude * glm::sin(wind_angle_rad),
                        amplitude, supress_factor
                };
                init_spectrum_kernel.setArg(0, patch);
                init_spectrum_kernel.setArg(1, params);
                init_spectrum_kernel.setArg(2, *noise_mem);
                init_spectrum_kernel.setArg(3, *h0k_mem);

                commandQueue.enqueueNDRangeKernel(init_spectrum_kernel, cl::NullRange,
                                                  cl::NDRange{ ocean_tex_size, ocean_tex_size }, lws);
                changed = false;
            } catch (const cl::Error& e)
            {
                printf("initial spectrum: OpenCL %s kernel error: %s\n", e.what(), IGetErrorString(e.err()));
                exit(1);
            }
        }

        // ping-pong phase spectrum kernel launch
        try
        {
            time_spectrum_kernel.setArg(0, elapsed);
            time_spectrum_kernel.setArg(1, patch);
            time_spectrum_kernel.setArg(2, *h0k_mem);
            time_spectrum_kernel.setArg(3, *dxyz_coef_mem[0]);
            time_spectrum_kernel.setArg(4, *dxyz_coef_mem[1]);
            time_spectrum_kernel.setArg(5, *dxyz_coef_mem[2]);

            commandQueue.enqueueNDRangeKernel(
                time_spectrum_kernel, cl::NullRange,
                cl::NDRange{ ocean_tex_size, ocean_tex_size }, lws);
        }
        catch (const cl::Error& e)
        {
            printf("updateSpectrum: OpenCL %s kernel error: %s\n", e.what(), IGetErrorString(e.err()));
            exit(1);
        }


        // perform 1D FFT horizontal and vertical iterations
        size_t log_2_N = (size_t) (log(ocean_tex_size)/log(2.f));
        fft_kernel.setArg(1, patch);
        fft_kernel.setArg(2, *twiddle_factors_mem);
        for ( cl_int i=0; i<3; i++)
        {
            const cl::Image * displ_swap[] = {dxyz_coef_mem[i].get(), hkt_pong_mem.get()};
            cl_int2 mode = (cl_int2){0, 0};

            bool ifft_pingpong=false;
            for (int p = 0; p < log_2_N; p++)
            {
                if (ifft_pingpong)
                {
                    fft_kernel.setArg(3, *displ_swap[1]);
                    fft_kernel.setArg(4, *displ_swap[0]);
                }
                else
                {
                    fft_kernel.setArg(3, *displ_swap[0]);
                    fft_kernel.setArg(4, *displ_swap[1]);
                }

                mode.s[1] = p;
                fft_kernel.setArg(0, mode);

                commandQueue.enqueueNDRangeKernel(
                    fft_kernel, cl::NullRange,
                    cl::NDRange{ ocean_tex_size, ocean_tex_size }, lws);



                ifft_pingpong = !ifft_pingpong;
            }

            // Cols
            mode.s[0] = 1;
            for (int p = 0; p < log_2_N; p++)
            {
                if (ifft_pingpong)
                {
                    fft_kernel.setArg(3, *displ_swap[1]);
                    fft_kernel.setArg(4, *displ_swap[0]);
                }
                else
                {
                    fft_kernel.setArg(3, *displ_swap[0]);
                    fft_kernel.setArg(4, *displ_swap[1]);
                }

                mode.s[1] = p;
                fft_kernel.setArg(0, mode);

                commandQueue.enqueueNDRangeKernel(
                    fft_kernel, cl::NullRange,
                    cl::NDRange{ ocean_tex_size, ocean_tex_size }, lws);

                ifft_pingpong = !ifft_pingpong;
            }

            if (log_2_N%2)
            {
                // swap images if pingpong hold on temporary buffer
                std::array<size_t, 3> orig = {0,0,0}, region={ocean_tex_size, ocean_tex_size, 1};
                commandQueue.enqueueCopyImage(*displ_swap[0], *displ_swap[1], orig, orig, region);
            }
        }

        if (useExternalMemory)
        {
            for (size_t target=0; target<IOPT_COUNT; target++)
            {
                commandQueue.enqueueAcquireExternalMemObjects(
                    { *mems[target][currentImage] });
            }
        }

        // inversion
        {
            inversion_kernel.setArg(0, patch);
            inversion_kernel.setArg(1, *dxyz_coef_mem[0]);
            inversion_kernel.setArg(2, *dxyz_coef_mem[1]);
            inversion_kernel.setArg(3, *dxyz_coef_mem[2]);
            inversion_kernel.setArg(4, *mems[IOPT_DISPLACEMENT][currentImage]);

            commandQueue.enqueueNDRangeKernel(
                inversion_kernel, cl::NullRange,
                cl::NDRange{ ocean_tex_size, ocean_tex_size }, lws);
        }

        // normals computation
        {
            cl_float2 factors = cl_float2 { choppiness, alt_scale };

            normals_kernel.setArg(0, patch);
            normals_kernel.setArg(1, factors);
            normals_kernel.setArg(2, *mems[IOPT_DISPLACEMENT][currentImage]);
            normals_kernel.setArg(3, *mems[IOPT_NORMAL_MAP][currentImage]);

            commandQueue.enqueueNDRangeKernel(
                normals_kernel, cl::NullRange,
                cl::NDRange{ ocean_tex_size, ocean_tex_size }, lws);
        }
    }

    void updateOcean(uint32_t currentImage)
    {
        updateUniforms(currentImage);

        updateSpectrum(currentImage);

        if (useExternalMemory)
        {
            for (size_t target=0; target<IOPT_COUNT; target++)
            {
                commandQueue.enqueueReleaseExternalMemObjects(
                    { *mems[target][currentImage] });
            }

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
        else
        {
            for (size_t target=0; target<IOPT_COUNT; target++)
            {
                size_t rowPitch = 0;
                void* pixels = commandQueue.enqueueMapImage(
                    *mems[target][currentImage], CL_TRUE, CL_MAP_READ, { 0, 0, 0 },
                    { ocean_tex_size, ocean_tex_size, 1 }, &rowPitch, nullptr);

                VkDeviceSize imageSize = ocean_tex_size * ocean_tex_size * 4 * sizeof(float);

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
                                  static_cast<uint32_t>(ocean_tex_size),
                                  static_cast<uint32_t>(ocean_tex_size));
                transitionImageLayout(textureImages[target].images[currentImage],
                                      VK_FORMAT_R32G32B32A32_SFLOAT,
                                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }
        }
    }

    void drawFrame()
    {
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

    static void mouse_roll(GLFWwindow* window, double oX, double oY)
    {
        auto app = (OceanApplication*)glfwGetWindowUserPointer(window);
        app->mouse_roll(oX, oY);
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
