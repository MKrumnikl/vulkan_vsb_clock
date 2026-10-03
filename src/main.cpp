#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr uint32_t WIDTH = 1100;
constexpr uint32_t HEIGHT = 820;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;
constexpr float PI = 3.14159265358979323846f;

const glm::vec3 COL_FACE   {0.92f, 0.93f, 0.91f};
const glm::vec3 COL_RIM    {0.10f, 0.12f, 0.14f};
const glm::vec3 COL_DARK   {0.055f, 0.065f, 0.075f};
const glm::vec3 COL_MARK   {0.18f, 0.19f, 0.20f};
const glm::vec3 COL_TEAL   {0.05f, 0.66f, 0.57f};
const glm::vec3 COL_RED    {0.90f, 0.035f, 0.045f};
const glm::vec3 COL_METAL  {0.34f, 0.37f, 0.40f};

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec3 color;

    static VkVertexInputBindingDescription bindingDescription() {
        VkVertexInputBindingDescription b{};
        b.binding = 0;
        b.stride = sizeof(Vertex);
        b.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return b;
    }

    static std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 3> a{};
        a[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)};
        a[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)};
        a[2] = {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)};
        return a;
    }
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

struct DrawRange {
    uint32_t firstIndex{};
    uint32_t indexCount{};
};

struct CameraUBO {
    glm::mat4 viewProj{1.0f};
    glm::vec4 eyePos{0.0f};
    glm::vec4 lightPos{0.0f};
    glm::vec4 ambient{0.0f};
};

struct PushConstants {
    glm::mat4 model{1.0f};
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    bool complete() const { return graphicsFamily.has_value() && presentFamily.has_value(); }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter,
                        VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
        if ((typeFilter & (1u << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("No suitable Vulkan memory type found");
}

std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open shader: " + filename);
    const auto size = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(size);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(size));
    return buffer;
}

void appendQuad(Mesh& m, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    m.indices.insert(m.indices.end(), {a, b, c, a, c, d});
}

void addBox(Mesh& m, glm::vec3 center, glm::vec3 size, glm::vec3 color, float rotZ = 0.0f) {
    const glm::vec3 h = size * 0.5f;
    const float cs = std::cos(rotZ), sn = std::sin(rotZ);
    auto transform = [&](glm::vec3 p) {
        glm::vec3 q{cs * p.x - sn * p.y, sn * p.x + cs * p.y, p.z};
        return q + center;
    };
    auto nrot = [&](glm::vec3 n) {
        return glm::normalize(glm::vec3{cs * n.x - sn * n.y, sn * n.x + cs * n.y, n.z});
    };

    const std::array<glm::vec3, 8> p = {
        glm::vec3{-h.x,-h.y,-h.z}, glm::vec3{ h.x,-h.y,-h.z},
        glm::vec3{ h.x, h.y,-h.z}, glm::vec3{-h.x, h.y,-h.z},
        glm::vec3{-h.x,-h.y, h.z}, glm::vec3{ h.x,-h.y, h.z},
        glm::vec3{ h.x, h.y, h.z}, glm::vec3{-h.x, h.y, h.z}
    };

    struct Face { int a,b,c,d; glm::vec3 n; };
    const std::array<Face, 6> faces = {{
        {4,5,6,7,{ 0, 0, 1}}, {1,0,3,2,{ 0, 0,-1}},
        {0,4,7,3,{-1, 0, 0}}, {5,1,2,6,{ 1, 0, 0}},
        {3,7,6,2,{ 0, 1, 0}}, {0,1,5,4,{ 0,-1, 0}}
    }};

    for (const auto& f : faces) {
        const uint32_t base = static_cast<uint32_t>(m.vertices.size());
        const glm::vec3 normal = nrot(f.n);
        m.vertices.push_back({transform(p[f.a]), normal, color});
        m.vertices.push_back({transform(p[f.b]), normal, color});
        m.vertices.push_back({transform(p[f.c]), normal, color});
        m.vertices.push_back({transform(p[f.d]), normal, color});
        appendQuad(m, base, base+1, base+2, base+3);
    }
}

void addPrism(Mesh& m, const std::vector<glm::vec2>& poly, float zFront, float depth, glm::vec3 color) {
    const float zBack = zFront - depth;
    const uint32_t frontBase = static_cast<uint32_t>(m.vertices.size());
    for (auto p : poly) m.vertices.push_back({{p.x,p.y,zFront},{0,0,1},color});
    for (size_t i = 1; i + 1 < poly.size(); ++i)
        m.indices.insert(m.indices.end(), {frontBase, frontBase + static_cast<uint32_t>(i), frontBase + static_cast<uint32_t>(i+1)});

    const uint32_t backBase = static_cast<uint32_t>(m.vertices.size());
    for (auto p : poly) m.vertices.push_back({{p.x,p.y,zBack},{0,0,-1},color});
    for (size_t i = 1; i + 1 < poly.size(); ++i)
        m.indices.insert(m.indices.end(), {backBase, backBase + static_cast<uint32_t>(i+1), backBase + static_cast<uint32_t>(i)});

    for (size_t i = 0; i < poly.size(); ++i) {
        const size_t j = (i + 1) % poly.size();
        const glm::vec2 e = poly[j] - poly[i];
        const glm::vec3 n = glm::normalize(glm::vec3{e.y, -e.x, 0.0f});
        const uint32_t base = static_cast<uint32_t>(m.vertices.size());
        m.vertices.push_back({{poly[i].x,poly[i].y,zBack},n,color});
        m.vertices.push_back({{poly[j].x,poly[j].y,zBack},n,color});
        m.vertices.push_back({{poly[j].x,poly[j].y,zFront},n,color});
        m.vertices.push_back({{poly[i].x,poly[i].y,zFront},n,color});
        appendQuad(m, base, base+1, base+2, base+3);
    }
}

void addCylinder(Mesh& m, float radius, float zCenter, float depth, glm::vec3 frontColor,
                 glm::vec3 sideColor, int segments = 128) {
    const float zf = zCenter + depth * 0.5f;
    const float zb = zCenter - depth * 0.5f;

    uint32_t centerF = static_cast<uint32_t>(m.vertices.size());
    m.vertices.push_back({{0,0,zf},{0,0,1},frontColor});
    uint32_t ringF = static_cast<uint32_t>(m.vertices.size());
    for (int i=0;i<=segments;++i) {
        float a = 2.0f*PI*float(i)/float(segments);
        m.vertices.push_back({{radius*std::cos(a),radius*std::sin(a),zf},{0,0,1},frontColor});
    }
    for(int i=0;i<segments;++i)
        m.indices.insert(m.indices.end(), {centerF, ringF+static_cast<uint32_t>(i), ringF+static_cast<uint32_t>(i+1)});

    uint32_t centerB = static_cast<uint32_t>(m.vertices.size());
    m.vertices.push_back({{0,0,zb},{0,0,-1},sideColor});
    uint32_t ringB = static_cast<uint32_t>(m.vertices.size());
    for (int i=0;i<=segments;++i) {
        float a = 2.0f*PI*float(i)/float(segments);
        m.vertices.push_back({{radius*std::cos(a),radius*std::sin(a),zb},{0,0,-1},sideColor});
    }
    for(int i=0;i<segments;++i)
        m.indices.insert(m.indices.end(), {centerB, ringB+static_cast<uint32_t>(i+1), ringB+static_cast<uint32_t>(i)});

    for(int i=0;i<segments;++i) {
        float a0 = 2.0f*PI*float(i)/float(segments);
        float a1 = 2.0f*PI*float(i+1)/float(segments);
        glm::vec3 n0{std::cos(a0),std::sin(a0),0};
        glm::vec3 n1{std::cos(a1),std::sin(a1),0};
        uint32_t base = static_cast<uint32_t>(m.vertices.size());
        m.vertices.push_back({{radius*n0.x,radius*n0.y,zb},n0,sideColor});
        m.vertices.push_back({{radius*n1.x,radius*n1.y,zb},n1,sideColor});
        m.vertices.push_back({{radius*n1.x,radius*n1.y,zf},n1,sideColor});
        m.vertices.push_back({{radius*n0.x,radius*n0.y,zf},n0,sideColor});
        appendQuad(m, base, base+1, base+2, base+3);
    }
}

void addRing(Mesh& m, float outerR, float innerR, float zFront, float depth, glm::vec3 color, int segments=160) {
    const float zb = zFront - depth;
    for(int i=0;i<segments;++i) {
        float a0=2*PI*float(i)/segments, a1=2*PI*float(i+1)/segments;
        glm::vec2 o0{outerR*std::cos(a0),outerR*std::sin(a0)};
        glm::vec2 o1{outerR*std::cos(a1),outerR*std::sin(a1)};
        glm::vec2 i0{innerR*std::cos(a0),innerR*std::sin(a0)};
        glm::vec2 i1{innerR*std::cos(a1),innerR*std::sin(a1)};

        uint32_t base = static_cast<uint32_t>(m.vertices.size());
        m.vertices.push_back({{i0.x,i0.y,zFront},{0,0,1},color});
        m.vertices.push_back({{o0.x,o0.y,zFront},{0,0,1},color});
        m.vertices.push_back({{o1.x,o1.y,zFront},{0,0,1},color});
        m.vertices.push_back({{i1.x,i1.y,zFront},{0,0,1},color});
        appendQuad(m,base,base+1,base+2,base+3);

        glm::vec3 no0{std::cos(a0),std::sin(a0),0}, no1{std::cos(a1),std::sin(a1),0};
        base = static_cast<uint32_t>(m.vertices.size());
        m.vertices.push_back({{o0.x,o0.y,zb},no0,color});
        m.vertices.push_back({{o1.x,o1.y,zb},no1,color});
        m.vertices.push_back({{o1.x,o1.y,zFront},no1,color});
        m.vertices.push_back({{o0.x,o0.y,zFront},no0,color});
        appendQuad(m,base,base+1,base+2,base+3);
    }
}

const bool SEGMENTS[10][7] = {
    {1,1,1,1,1,1,0}, {0,1,1,0,0,0,0}, {1,1,0,1,1,0,1}, {1,1,1,1,0,0,1},
    {0,1,1,0,0,1,1}, {1,0,1,1,0,1,1}, {1,0,1,1,1,1,1}, {1,1,1,0,0,0,0},
    {1,1,1,1,1,1,1}, {1,1,1,1,0,1,1}
};

void addSevenSegDigit(Mesh& m, int digit, glm::vec2 center, float height, float z, glm::vec3 color) {
    float w = height * 0.52f;
    float t = height * 0.075f;
    float half = height * 0.23f;
    float zh = 0.014f;
    auto hseg = [&](float y){ addBox(m,{center.x,center.y+y,z},{w-t,t,zh},color); };
    auto vseg = [&](float x,float y){ addBox(m,{center.x+x,center.y+y,z},{t,half-t*0.6f,zh},color); };
    const bool* s = SEGMENTS[digit];
    if(s[0]) hseg(height*0.43f);
    if(s[1]) vseg(w*0.47f, height*0.22f);
    if(s[2]) vseg(w*0.47f,-height*0.22f);
    if(s[3]) hseg(-height*0.43f);
    if(s[4]) vseg(-w*0.47f,-height*0.22f);
    if(s[5]) vseg(-w*0.47f, height*0.22f);
    if(s[6]) hseg(0.0f);
}

void addNumber(Mesh& m, int n, glm::vec2 center, float height, float z, glm::vec3 color) {
    if (n < 10) {
        addSevenSegDigit(m,n,center,height,z,color);
    } else {
        float spacing = height * 0.33f;
        addSevenSegDigit(m,n/10,{center.x-spacing,center.y},height,z,color);
        addSevenSegDigit(m,n%10,{center.x+spacing,center.y},height,z,color);
    }
}

void addBeam2D(Mesh& m, glm::vec2 a, glm::vec2 b, float width, float z, float depth, glm::vec3 color) {
    glm::vec2 d = b-a;
    float len = glm::length(d);
    float ang = std::atan2(d.y,d.x) - PI*0.5f;
    glm::vec2 c=(a+b)*0.5f;
    addBox(m,{c.x,c.y,z},{width,len,depth},color,ang);
}

void addVsbLogo(Mesh& m) {
    const float z = 0.168f;
    const float d = 0.022f;
    const float y = 0.105f;
    const float x0 = -0.25f;
    const float h = 0.19f;
    const float w = 0.13f;
    const float stroke = 0.026f;

    // V
    addBeam2D(m,{x0-w*0.45f,y+h*0.5f},{x0,y-h*0.5f},stroke,z,d,COL_TEAL);
    addBeam2D(m,{x0+w*0.45f,y+h*0.5f},{x0,y-h*0.5f},stroke,z,d,COL_TEAL);

    // Š, deliberately geometric/modern.
    const float sx = 0.0f;
    addBox(m,{sx,y+h*0.43f,z},{0.12f,stroke,d},COL_TEAL);
    addBox(m,{sx-0.045f,y+h*0.22f,z},{stroke,0.075f,d},COL_TEAL);
    addBox(m,{sx,y,z},{0.12f,stroke,d},COL_TEAL);
    addBox(m,{sx+0.045f,y-h*0.22f,z},{stroke,0.075f,d},COL_TEAL);
    addBox(m,{sx,y-h*0.43f,z},{0.12f,stroke,d},COL_TEAL);
    addBeam2D(m,{sx-0.035f,y+h*0.67f},{sx,y+h*0.59f},0.017f,z,d,COL_TEAL);
    addBeam2D(m,{sx+0.035f,y+h*0.67f},{sx,y+h*0.59f},0.017f,z,d,COL_TEAL);

    // B
    const float bx = 0.25f;
    addBox(m,{bx-w*0.43f,y,z},{stroke,h,d},COL_TEAL);
    addBox(m,{bx,y+h*0.43f,z},{w*0.72f,stroke,d},COL_TEAL);
    addBox(m,{bx,y,z},{w*0.72f,stroke,d},COL_TEAL);
    addBox(m,{bx,y-h*0.43f,z},{w*0.72f,stroke,d},COL_TEAL);
    addBox(m,{bx+w*0.29f,y+h*0.22f,z},{stroke,0.065f,d},COL_TEAL);
    addBox(m,{bx+w*0.29f,y-h*0.22f,z},{stroke,0.065f,d},COL_TEAL);

    // Five bars from the supplied clock reference.
    const std::array<float,5> bh{0.15f,0.24f,0.34f,0.24f,0.15f};
    for(size_t i=0;i<bh.size();++i) {
        float x = -0.20f + float(i)*0.10f;
        float cy = -0.23f - bh[i]*0.5f;
        addBox(m,{x,cy,z},{0.026f,bh[i],d},COL_TEAL);
    }
}

Mesh buildClockMesh(DrawRange& staticRange, DrawRange& hourRange,
                    DrawRange& minuteRange, DrawRange& secondRange) {
    Mesh m;
    staticRange.firstIndex = static_cast<uint32_t>(m.indices.size());

    // Back body and metallic bevel.
    addCylinder(m,1.075f,-0.015f,0.17f,COL_RIM,COL_DARK,192);
    addCylinder(m,1.018f, 0.055f,0.105f,COL_FACE,COL_METAL,192);
    addRing(m,1.033f,0.995f,0.126f,0.030f,COL_METAL,192);

    // Ticks.
    for(int i=0;i<60;++i) {
        const float theta = 2.0f*PI*float(i)/60.0f;
        const bool major = (i%5)==0;
        const float r = major ? 0.885f : 0.902f;
        const float len = major ? 0.072f : 0.031f;
        const float wid = major ? 0.014f : 0.0065f;
        const float x = r*std::sin(theta);
        const float y = r*std::cos(theta);
        addBox(m,{x,y,0.139f},{wid,len,0.014f},COL_MARK,-theta);
    }

    // Numerals.
    for(int n=1;n<=12;++n) {
        const float theta = 2.0f*PI*float(n)/12.0f;
        const float r = 0.745f;
        glm::vec2 p{r*std::sin(theta),r*std::cos(theta)};
        float h = (n>=10) ? 0.135f : 0.155f;
        addNumber(m,n,p,h,0.148f,COL_MARK);
    }

    addVsbLogo(m);

    // Center hub and a subtle recessed center ring.
    addCylinder(m,0.064f,0.205f,0.060f,COL_DARK,COL_METAL,64);
    addRing(m,0.081f,0.066f,0.181f,0.018f,COL_METAL,64);

    staticRange.indexCount = static_cast<uint32_t>(m.indices.size()) - staticRange.firstIndex;

    // Hour hand, local +Y points toward twelve. Small tail adds a mechanical feel.
    hourRange.firstIndex = static_cast<uint32_t>(m.indices.size());
    addPrism(m,{{-0.044f,-0.115f},{0.044f,-0.115f},{0.038f,0.45f},{0.0f,0.54f},{-0.038f,0.45f}},
             0.228f,0.035f,{0.025f,0.030f,0.034f});
    hourRange.indexCount = static_cast<uint32_t>(m.indices.size()) - hourRange.firstIndex;

    minuteRange.firstIndex = static_cast<uint32_t>(m.indices.size());
    addPrism(m,{{-0.027f,-0.14f},{0.027f,-0.14f},{0.023f,0.69f},{0.0f,0.79f},{-0.023f,0.69f}},
             0.252f,0.027f,{0.015f,0.018f,0.021f});
    minuteRange.indexCount = static_cast<uint32_t>(m.indices.size()) - minuteRange.firstIndex;

    secondRange.firstIndex = static_cast<uint32_t>(m.indices.size());
    addPrism(m,{{-0.010f,-0.20f},{0.010f,-0.20f},{0.007f,0.82f},{0.0f,0.91f},{-0.007f,0.82f}},
             0.280f,0.017f,COL_RED);
    addCylinder(m,0.031f,0.286f,0.019f,COL_RED,COL_RED,48);
    secondRange.indexCount = static_cast<uint32_t>(m.indices.size()) - secondRange.firstIndex;

    return m;
}

class App {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue presentQueue = VK_NULL_HANDLE;

    VkSwapchainKHR swapChain = VK_NULL_HANDLE;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat{};
    VkExtent2D swapChainExtent{};
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;

    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;

    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers;

    VkImage depthImage = VK_NULL_HANDLE;
    VkDeviceMemory depthImageMemory = VK_NULL_HANDLE;
    VkImageView depthImageView = VK_NULL_HANDLE;
    VkFormat depthFormat{};

    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indexBufferMemory = VK_NULL_HANDLE;
    uint32_t indexCount = 0;
    DrawRange staticRange{}, hourRange{}, minuteRange{}, secondRange{};

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    uint32_t currentFrame = 0;
    bool framebufferResized = false;

    float yaw = -0.14f;
    float pitch = 0.12f;
    float distance = 3.05f;
    bool dragging = false;
    bool autoOrbit = true;
    double lastX = 0.0, lastY = 0.0;
    bool fullscreen = false;
    int savedX=0,savedY=0,savedW=WIDTH,savedH=HEIGHT;
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

    static void framebufferResizeCallback(GLFWwindow* w, int, int) {
        auto app = reinterpret_cast<App*>(glfwGetWindowUserPointer(w));
        app->framebufferResized = true;
    }

    static void mouseButtonCallback(GLFWwindow* w, int button, int action, int) {
        auto app = reinterpret_cast<App*>(glfwGetWindowUserPointer(w));
        if(button == GLFW_MOUSE_BUTTON_LEFT) {
            app->dragging = action == GLFW_PRESS;
            glfwGetCursorPos(w,&app->lastX,&app->lastY);
        }
    }

    static void cursorCallback(GLFWwindow* w, double x, double y) {
        auto app = reinterpret_cast<App*>(glfwGetWindowUserPointer(w));
        if(!app->dragging) return;
        double dx=x-app->lastX, dy=y-app->lastY;
        app->lastX=x; app->lastY=y;
        app->yaw += static_cast<float>(dx)*0.004f;
        app->pitch += static_cast<float>(dy)*0.004f;
        app->pitch = std::clamp(app->pitch,-0.72f,0.72f);
        app->autoOrbit = false;
    }

    static void scrollCallback(GLFWwindow* w, double, double yoff) {
        auto app = reinterpret_cast<App*>(glfwGetWindowUserPointer(w));
        app->distance = std::clamp(app->distance-static_cast<float>(yoff)*0.16f,2.15f,5.0f);
    }

    static void keyCallback(GLFWwindow* w,int key,int,int action,int) {
        if(action != GLFW_PRESS) return;
        auto app = reinterpret_cast<App*>(glfwGetWindowUserPointer(w));
        if(key==GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(w,GLFW_TRUE);
        else if(key==GLFW_KEY_SPACE) app->autoOrbit=!app->autoOrbit;
        else if(key==GLFW_KEY_R) { app->yaw=-0.14f; app->pitch=0.12f; app->distance=3.05f; app->autoOrbit=true; }
        else if(key==GLFW_KEY_F) app->toggleFullscreen();
    }

    void toggleFullscreen() {
        fullscreen = !fullscreen;
        if(fullscreen) {
            glfwGetWindowPos(window,&savedX,&savedY);
            glfwGetWindowSize(window,&savedW,&savedH);
            GLFWmonitor* mon=glfwGetPrimaryMonitor();
            const GLFWvidmode* mode=glfwGetVideoMode(mon);
            glfwSetWindowMonitor(window,mon,0,0,mode->width,mode->height,mode->refreshRate);
        } else {
            glfwSetWindowMonitor(window,nullptr,savedX,savedY,savedW,savedH,0);
        }
    }

    void initWindow() {
        if(!glfwInit()) throw std::runtime_error("GLFW initialization failed");
        if(!glfwVulkanSupported()) throw std::runtime_error("GLFW reports no Vulkan support/loader");
        glfwWindowHint(GLFW_CLIENT_API,GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE,GLFW_TRUE);
        window=glfwCreateWindow(WIDTH,HEIGHT,"VSB 3D Vulkan Clock",nullptr,nullptr);
        if(!window) throw std::runtime_error("Cannot create GLFW window");
        glfwSetWindowUserPointer(window,this);
        glfwSetFramebufferSizeCallback(window,framebufferResizeCallback);
        glfwSetMouseButtonCallback(window,mouseButtonCallback);
        glfwSetCursorPosCallback(window,cursorCallback);
        glfwSetScrollCallback(window,scrollCallback);
        glfwSetKeyCallback(window,keyCallback);
    }

    void initVulkan() {
        createInstance();
        if(glfwCreateWindowSurface(instance,window,nullptr,&surface)!=VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan window surface");
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createCommandPool();
        createDepthResources();
        createFramebuffers();
        createMeshBuffers();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

    void createInstance() {
        VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        appInfo.pApplicationName="VSB Vulkan Clock";
        appInfo.applicationVersion=VK_MAKE_VERSION(1,0,0);
        appInfo.pEngineName="none";
        appInfo.engineVersion=VK_MAKE_VERSION(1,0,0);
        appInfo.apiVersion=VK_API_VERSION_1_0;

        uint32_t count=0;
        const char** exts=glfwGetRequiredInstanceExtensions(&count);
        if(!exts) throw std::runtime_error("GLFW cannot provide Vulkan instance extensions");
        VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ci.pApplicationInfo=&appInfo;
        ci.enabledExtensionCount=count;
        ci.ppEnabledExtensionNames=exts;
        if(vkCreateInstance(&ci,nullptr,&instance)!=VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan instance");
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice dev) {
        QueueFamilyIndices out;
        uint32_t count=0; vkGetPhysicalDeviceQueueFamilyProperties(dev,&count,nullptr);
        std::vector<VkQueueFamilyProperties> props(count);
        vkGetPhysicalDeviceQueueFamilyProperties(dev,&count,props.data());
        for(uint32_t i=0;i<count;++i) {
            if(props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) out.graphicsFamily=i;
            VkBool32 present=false; vkGetPhysicalDeviceSurfaceSupportKHR(dev,i,surface,&present);
            if(present) out.presentFamily=i;
            if(out.complete()) break;
        }
        return out;
    }

    bool deviceSupportsSwapchain(VkPhysicalDevice dev) {
        uint32_t count=0; vkEnumerateDeviceExtensionProperties(dev,nullptr,&count,nullptr);
        std::vector<VkExtensionProperties> props(count);
        vkEnumerateDeviceExtensionProperties(dev,nullptr,&count,props.data());
        for(const auto& p:props) if(std::strcmp(p.extensionName,VK_KHR_SWAPCHAIN_EXTENSION_NAME)==0) return true;
        return false;
    }

    SwapChainSupportDetails querySwapChain(VkPhysicalDevice dev) {
        SwapChainSupportDetails d;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev,surface,&d.capabilities);
        uint32_t count=0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(dev,surface,&count,nullptr);
        d.formats.resize(count); if(count) vkGetPhysicalDeviceSurfaceFormatsKHR(dev,surface,&count,d.formats.data());
        count=0; vkGetPhysicalDeviceSurfacePresentModesKHR(dev,surface,&count,nullptr);
        d.presentModes.resize(count); if(count) vkGetPhysicalDeviceSurfacePresentModesKHR(dev,surface,&count,d.presentModes.data());
        return d;
    }

    bool suitable(VkPhysicalDevice dev) {
        auto q=findQueueFamilies(dev);
        if(!q.complete() || !deviceSupportsSwapchain(dev)) return false;
        auto s=querySwapChain(dev);
        return !s.formats.empty() && !s.presentModes.empty();
    }

    void pickPhysicalDevice() {
        uint32_t count=0; vkEnumeratePhysicalDevices(instance,&count,nullptr);
        if(!count) throw std::runtime_error("No Vulkan-capable GPU found");
        std::vector<VkPhysicalDevice> devices(count); vkEnumeratePhysicalDevices(instance,&count,devices.data());
        // Prefer discrete GPU but accept any suitable device.
        VkPhysicalDevice fallback=VK_NULL_HANDLE;
        for(auto d:devices) {
            if(!suitable(d)) continue;
            VkPhysicalDeviceProperties p{}; vkGetPhysicalDeviceProperties(d,&p);
            if(fallback==VK_NULL_HANDLE) fallback=d;
            if(p.deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { physicalDevice=d; break; }
        }
        if(physicalDevice==VK_NULL_HANDLE) physicalDevice=fallback;
        if(physicalDevice==VK_NULL_HANDLE) throw std::runtime_error("No suitable Vulkan GPU found");
        VkPhysicalDeviceProperties p{}; vkGetPhysicalDeviceProperties(physicalDevice,&p);
        std::cout << "Vulkan GPU: " << p.deviceName << "\n";
    }

    void createLogicalDevice() {
        auto idx=findQueueFamilies(physicalDevice);
        std::set<uint32_t> unique{idx.graphicsFamily.value(),idx.presentFamily.value()};
        float priority=1.0f;
        std::vector<VkDeviceQueueCreateInfo> qcis;
        for(uint32_t q:unique) {
            VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            qi.queueFamilyIndex=q; qi.queueCount=1; qi.pQueuePriorities=&priority; qcis.push_back(qi);
        }
        const char* exts[]={VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        VkPhysicalDeviceFeatures features{};
        VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        ci.queueCreateInfoCount=static_cast<uint32_t>(qcis.size()); ci.pQueueCreateInfos=qcis.data();
        ci.pEnabledFeatures=&features; ci.enabledExtensionCount=1; ci.ppEnabledExtensionNames=exts;
        if(vkCreateDevice(physicalDevice,&ci,nullptr,&device)!=VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan logical device");
        vkGetDeviceQueue(device,idx.graphicsFamily.value(),0,&graphicsQueue);
        vkGetDeviceQueue(device,idx.presentFamily.value(),0,&presentQueue);
    }

    VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& f) {
        for(auto x:f) if(x.format==VK_FORMAT_B8G8R8A8_SRGB && x.colorSpace==VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) return x;
        return f.front();
    }
    VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& modes) {
        for(auto m:modes) if(m==VK_PRESENT_MODE_MAILBOX_KHR) return m;
        return VK_PRESENT_MODE_FIFO_KHR;
    }
    VkExtent2D chooseExtent(const VkSurfaceCapabilitiesKHR& c) {
        if(c.currentExtent.width != std::numeric_limits<uint32_t>::max()) return c.currentExtent;
        int w,h; glfwGetFramebufferSize(window,&w,&h);
        VkExtent2D e{static_cast<uint32_t>(w),static_cast<uint32_t>(h)};
        e.width=std::clamp(e.width,c.minImageExtent.width,c.maxImageExtent.width);
        e.height=std::clamp(e.height,c.minImageExtent.height,c.maxImageExtent.height);
        return e;
    }

    void createSwapChain() {
        auto s=querySwapChain(physicalDevice);
        auto fmt=chooseSurfaceFormat(s.formats); auto mode=choosePresentMode(s.presentModes); auto extent=chooseExtent(s.capabilities);
        uint32_t imageCount=s.capabilities.minImageCount+1;
        if(s.capabilities.maxImageCount>0 && imageCount>s.capabilities.maxImageCount) imageCount=s.capabilities.maxImageCount;
        VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        ci.surface=surface; ci.minImageCount=imageCount; ci.imageFormat=fmt.format; ci.imageColorSpace=fmt.colorSpace;
        ci.imageExtent=extent; ci.imageArrayLayers=1; ci.imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        auto idx=findQueueFamilies(physicalDevice);
        uint32_t q[]{idx.graphicsFamily.value(),idx.presentFamily.value()};
        if(q[0]!=q[1]) { ci.imageSharingMode=VK_SHARING_MODE_CONCURRENT; ci.queueFamilyIndexCount=2; ci.pQueueFamilyIndices=q; }
        else ci.imageSharingMode=VK_SHARING_MODE_EXCLUSIVE;
        ci.preTransform=s.capabilities.currentTransform; ci.compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        ci.presentMode=mode; ci.clipped=VK_TRUE;
        if(vkCreateSwapchainKHR(device,&ci,nullptr,&swapChain)!=VK_SUCCESS) throw std::runtime_error("Failed to create swapchain");
        vkGetSwapchainImagesKHR(device,swapChain,&imageCount,nullptr); swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device,swapChain,&imageCount,swapChainImages.data());
        swapChainImageFormat=fmt.format; swapChainExtent=extent;
    }

    VkImageView createImageView(VkImage image,VkFormat format,VkImageAspectFlags aspect) {
        VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        ci.image=image; ci.viewType=VK_IMAGE_VIEW_TYPE_2D; ci.format=format;
        ci.subresourceRange.aspectMask=aspect; ci.subresourceRange.baseMipLevel=0; ci.subresourceRange.levelCount=1;
        ci.subresourceRange.baseArrayLayer=0; ci.subresourceRange.layerCount=1;
        VkImageView v; if(vkCreateImageView(device,&ci,nullptr,&v)!=VK_SUCCESS) throw std::runtime_error("Failed to create image view"); return v;
    }

    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());
        for(size_t i=0;i<swapChainImages.size();++i) swapChainImageViews[i]=createImageView(swapChainImages[i],swapChainImageFormat,VK_IMAGE_ASPECT_COLOR_BIT);
    }

    VkFormat findDepthFormat() {
        for(VkFormat f:{VK_FORMAT_D32_SFLOAT,VK_FORMAT_D32_SFLOAT_S8_UINT,VK_FORMAT_D24_UNORM_S8_UINT}) {
            VkFormatProperties p{}; vkGetPhysicalDeviceFormatProperties(physicalDevice,f,&p);
            if(p.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) return f;
        }
        throw std::runtime_error("No depth format available");
    }

    void createRenderPass() {
        depthFormat=findDepthFormat();
        VkAttachmentDescription color{}; color.format=swapChainImageFormat; color.samples=VK_SAMPLE_COUNT_1_BIT;
        color.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR; color.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
        color.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE; color.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED; color.finalLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentDescription depth{}; depth.format=depthFormat; depth.samples=VK_SAMPLE_COUNT_1_BIT;
        depth.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR; depth.storeOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE; depth.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED; depth.finalLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference cref{0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference dref{1,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sub{}; sub.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS; sub.colorAttachmentCount=1; sub.pColorAttachments=&cref; sub.pDepthStencilAttachment=&dref;
        VkSubpassDependency dep{}; dep.srcSubpass=VK_SUBPASS_EXTERNAL; dep.dstSubpass=0;
        dep.srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.dstStageMask=dep.srcStageMask; dep.dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        std::array<VkAttachmentDescription,2> at{color,depth};
        VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO}; ci.attachmentCount=2; ci.pAttachments=at.data(); ci.subpassCount=1; ci.pSubpasses=&sub; ci.dependencyCount=1; ci.pDependencies=&dep;
        if(vkCreateRenderPass(device,&ci,nullptr,&renderPass)!=VK_SUCCESS) throw std::runtime_error("Failed to create render pass");
    }

    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding b{}; b.binding=0; b.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; b.descriptorCount=1; b.stageFlags=VK_SHADER_STAGE_VERTEX_BIT;
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; ci.bindingCount=1; ci.pBindings=&b;
        if(vkCreateDescriptorSetLayout(device,&ci,nullptr,&descriptorSetLayout)!=VK_SUCCESS) throw std::runtime_error("Failed to create descriptor layout");
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; ci.codeSize=code.size(); ci.pCode=reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule m; if(vkCreateShaderModule(device,&ci,nullptr,&m)!=VK_SUCCESS) throw std::runtime_error("Failed to create shader module"); return m;
    }

    void createGraphicsPipeline() {
        const std::string base=CLOCK_SHADER_DIR;
        auto vertCode=readFile(base+"/clock.vert.spv"), fragCode=readFile(base+"/clock.frag.spv");
        VkShaderModule vert=createShaderModule(vertCode), frag=createShaderModule(fragCode);
        VkPipelineShaderStageCreateInfo vs{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; vs.stage=VK_SHADER_STAGE_VERTEX_BIT; vs.module=vert; vs.pName="main";
        VkPipelineShaderStageCreateInfo fs{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; fs.stage=VK_SHADER_STAGE_FRAGMENT_BIT; fs.module=frag; fs.pName="main";
        VkPipelineShaderStageCreateInfo stages[]{vs,fs};
        auto bind=Vertex::bindingDescription(); auto attrs=Vertex::attributeDescriptions();
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO}; vi.vertexBindingDescriptionCount=1; vi.pVertexBindingDescriptions=&bind; vi.vertexAttributeDescriptionCount=static_cast<uint32_t>(attrs.size()); vi.pVertexAttributeDescriptions=attrs.data();
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO}; ia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO}; vp.viewportCount=1; vp.scissorCount=1;
        std::array<VkDynamicState,2> dyn{VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO}; ds.dynamicStateCount=2; ds.pDynamicStates=dyn.data();
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO}; rs.polygonMode=VK_POLYGON_MODE_FILL; rs.lineWidth=1; rs.cullMode=VK_CULL_MODE_BACK_BIT; rs.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO}; ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo dz{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO}; dz.depthTestEnable=VK_TRUE; dz.depthWriteEnable=VK_TRUE; dz.depthCompareOp=VK_COMPARE_OP_LESS;
        VkPipelineColorBlendAttachmentState cba{}; cba.colorWriteMask=VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO}; cb.attachmentCount=1; cb.pAttachments=&cba;
        VkPushConstantRange pcr{}; pcr.stageFlags=VK_SHADER_STAGE_VERTEX_BIT; pcr.offset=0; pcr.size=sizeof(PushConstants);
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pli.setLayoutCount=1; pli.pSetLayouts=&descriptorSetLayout; pli.pushConstantRangeCount=1; pli.pPushConstantRanges=&pcr;
        if(vkCreatePipelineLayout(device,&pli,nullptr,&pipelineLayout)!=VK_SUCCESS) throw std::runtime_error("Failed to create pipeline layout");
        VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO}; ci.stageCount=2; ci.pStages=stages; ci.pVertexInputState=&vi; ci.pInputAssemblyState=&ia; ci.pViewportState=&vp; ci.pRasterizationState=&rs; ci.pMultisampleState=&ms; ci.pDepthStencilState=&dz; ci.pColorBlendState=&cb; ci.pDynamicState=&ds; ci.layout=pipelineLayout; ci.renderPass=renderPass; ci.subpass=0;
        if(vkCreateGraphicsPipelines(device,VK_NULL_HANDLE,1,&ci,nullptr,&graphicsPipeline)!=VK_SUCCESS) throw std::runtime_error("Failed to create graphics pipeline");
        vkDestroyShaderModule(device,frag,nullptr); vkDestroyShaderModule(device,vert,nullptr);
    }

    void createCommandPool() {
        auto q=findQueueFamilies(physicalDevice);
        VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; ci.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; ci.queueFamilyIndex=q.graphicsFamily.value();
        if(vkCreateCommandPool(device,&ci,nullptr,&commandPool)!=VK_SUCCESS) throw std::runtime_error("Failed to create command pool");
    }

    void createImage(uint32_t w,uint32_t h,VkFormat format,VkImageUsageFlags usage,VkImage& image,VkDeviceMemory& memory) {
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO}; ci.imageType=VK_IMAGE_TYPE_2D; ci.extent={w,h,1}; ci.mipLevels=1; ci.arrayLayers=1; ci.format=format; ci.tiling=VK_IMAGE_TILING_OPTIMAL; ci.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED; ci.usage=usage; ci.samples=VK_SAMPLE_COUNT_1_BIT; ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        if(vkCreateImage(device,&ci,nullptr,&image)!=VK_SUCCESS) throw std::runtime_error("Failed to create image");
        VkMemoryRequirements req{}; vkGetImageMemoryRequirements(device,image,&req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize=req.size; ai.memoryTypeIndex=findMemoryType(physicalDevice,req.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if(vkAllocateMemory(device,&ai,nullptr,&memory)!=VK_SUCCESS) throw std::runtime_error("Failed to allocate image memory");
        vkBindImageMemory(device,image,memory,0);
    }

    void createDepthResources() {
        createImage(swapChainExtent.width,swapChainExtent.height,depthFormat,VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,depthImage,depthImageMemory);
        depthImageView=createImageView(depthImage,depthFormat,VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());
        for(size_t i=0;i<swapChainImageViews.size();++i) {
            std::array<VkImageView,2> at{swapChainImageViews[i],depthImageView};
            VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO}; ci.renderPass=renderPass; ci.attachmentCount=2; ci.pAttachments=at.data(); ci.width=swapChainExtent.width; ci.height=swapChainExtent.height; ci.layers=1;
            if(vkCreateFramebuffer(device,&ci,nullptr,&swapChainFramebuffers[i])!=VK_SUCCESS) throw std::runtime_error("Failed to create framebuffer");
        }
    }

    void createBuffer(VkDeviceSize size,VkBufferUsageFlags usage,VkMemoryPropertyFlags props,VkBuffer& buffer,VkDeviceMemory& memory) {
        VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; ci.size=size; ci.usage=usage; ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        if(vkCreateBuffer(device,&ci,nullptr,&buffer)!=VK_SUCCESS) throw std::runtime_error("Failed to create buffer");
        VkMemoryRequirements req{}; vkGetBufferMemoryRequirements(device,buffer,&req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize=req.size; ai.memoryTypeIndex=findMemoryType(physicalDevice,req.memoryTypeBits,props);
        if(vkAllocateMemory(device,&ai,nullptr,&memory)!=VK_SUCCESS) throw std::runtime_error("Failed to allocate buffer memory");
        vkBindBufferMemory(device,buffer,memory,0);
    }

    void copyBuffer(VkBuffer src,VkBuffer dst,VkDeviceSize size) {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandPool=commandPool; ai.commandBufferCount=1;
        VkCommandBuffer cmd; vkAllocateCommandBuffers(device,&ai,&cmd);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; vkBeginCommandBuffer(cmd,&bi);
        VkBufferCopy region{0,0,size}; vkCmdCopyBuffer(cmd,src,dst,1,&region); vkEndCommandBuffer(cmd);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cmd; vkQueueSubmit(graphicsQueue,1,&si,VK_NULL_HANDLE); vkQueueWaitIdle(graphicsQueue);
        vkFreeCommandBuffers(device,commandPool,1,&cmd);
    }

    void createMeshBuffers() {
        Mesh mesh=buildClockMesh(staticRange,hourRange,minuteRange,secondRange); indexCount=static_cast<uint32_t>(mesh.indices.size());
        VkDeviceSize vsize=sizeof(Vertex)*mesh.vertices.size(), isize=sizeof(uint32_t)*mesh.indices.size();
        VkBuffer staging; VkDeviceMemory stagingMem;
        createBuffer(vsize,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,staging,stagingMem);
        void* data; vkMapMemory(device,stagingMem,0,vsize,0,&data); std::memcpy(data,mesh.vertices.data(),static_cast<size_t>(vsize)); vkUnmapMemory(device,stagingMem);
        createBuffer(vsize,VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,vertexBuffer,vertexBufferMemory); copyBuffer(staging,vertexBuffer,vsize);
        vkDestroyBuffer(device,staging,nullptr); vkFreeMemory(device,stagingMem,nullptr);
        createBuffer(isize,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,staging,stagingMem);
        vkMapMemory(device,stagingMem,0,isize,0,&data); std::memcpy(data,mesh.indices.data(),static_cast<size_t>(isize)); vkUnmapMemory(device,stagingMem);
        createBuffer(isize,VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_INDEX_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,indexBuffer,indexBufferMemory); copyBuffer(staging,indexBuffer,isize);
        vkDestroyBuffer(device,staging,nullptr); vkFreeMemory(device,stagingMem,nullptr);
        std::cout << "Geometry: " << mesh.vertices.size() << " vertices, " << mesh.indices.size()/3 << " triangles\n";
    }

    void createUniformBuffers() {
        VkDeviceSize size=sizeof(CameraUBO); uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT); uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT); uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
        for(int i=0;i<MAX_FRAMES_IN_FLIGHT;++i) { createBuffer(size,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,uniformBuffers[i],uniformBuffersMemory[i]); vkMapMemory(device,uniformBuffersMemory[i],0,size,0,&uniformBuffersMapped[i]); }
    }

    void createDescriptorPool() {
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,MAX_FRAMES_IN_FLIGHT};
        VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; ci.poolSizeCount=1; ci.pPoolSizes=&ps; ci.maxSets=MAX_FRAMES_IN_FLIGHT;
        if(vkCreateDescriptorPool(device,&ci,nullptr,&descriptorPool)!=VK_SUCCESS) throw std::runtime_error("Failed to create descriptor pool");
    }

    void createDescriptorSets() {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,descriptorSetLayout);
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; ai.descriptorPool=descriptorPool; ai.descriptorSetCount=MAX_FRAMES_IN_FLIGHT; ai.pSetLayouts=layouts.data();
        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT); if(vkAllocateDescriptorSets(device,&ai,descriptorSets.data())!=VK_SUCCESS) throw std::runtime_error("Failed to allocate descriptor sets");
        for(int i=0;i<MAX_FRAMES_IN_FLIGHT;++i) {
            VkDescriptorBufferInfo bi{uniformBuffers[i],0,sizeof(CameraUBO)};
            VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; wr.dstSet=descriptorSets[i]; wr.dstBinding=0; wr.descriptorCount=1; wr.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; wr.pBufferInfo=&bi;
            vkUpdateDescriptorSets(device,1,&wr,0,nullptr);
        }
    }

    void createCommandBuffers() {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; ai.commandPool=commandPool; ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount=static_cast<uint32_t>(commandBuffers.size());
        if(vkAllocateCommandBuffers(device,&ai,commandBuffers.data())!=VK_SUCCESS) throw std::runtime_error("Failed to allocate command buffers");
    }

    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT); renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT); inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO}; VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; fi.flags=VK_FENCE_CREATE_SIGNALED_BIT;
        for(int i=0;i<MAX_FRAMES_IN_FLIGHT;++i) if(vkCreateSemaphore(device,&si,nullptr,&imageAvailableSemaphores[i])!=VK_SUCCESS || vkCreateSemaphore(device,&si,nullptr,&renderFinishedSemaphores[i])!=VK_SUCCESS || vkCreateFence(device,&fi,nullptr,&inFlightFences[i])!=VK_SUCCESS) throw std::runtime_error("Failed to create sync objects");
    }

    void getAngles(float& hour,float& minute,float& second) {
        using namespace std::chrono;
        auto now=system_clock::now();
        auto tt=system_clock::to_time_t(now);
        std::tm local{};
#ifdef _WIN32
        localtime_s(&local,&tt);
#else
        localtime_r(&tt,&local);
#endif
        auto secFraction = duration<double>(now - system_clock::from_time_t(tt)).count();
        // from_time_t may be coarse / DST transformed; fractional component can be odd around transitions.
        auto ms = duration_cast<milliseconds>(now.time_since_epoch()).count()%1000;
        double s=double(local.tm_sec)+double(ms)/1000.0;
        double mn=double(local.tm_min)+s/60.0;
        double hr=double(local.tm_hour%12)+mn/60.0;
        (void)secFraction;
        second=static_cast<float>(-s/60.0*2.0*PI);
        minute=static_cast<float>(-mn/60.0*2.0*PI);
        hour=static_cast<float>(-hr/12.0*2.0*PI);
    }

    void updateUniformBuffer(uint32_t frame) {
        float t=std::chrono::duration<float>(std::chrono::steady_clock::now()-startTime).count();
        float autoYaw = autoOrbit ? 0.065f*std::sin(t*0.27f) : 0.0f;
        float autoPitch = autoOrbit ? 0.035f*std::sin(t*0.19f+0.8f) : 0.0f;
        float yy=yaw+autoYaw, pp=pitch+autoPitch;
        glm::vec3 eye{
            distance*std::sin(yy)*std::cos(pp),
            distance*std::sin(pp),
            distance*std::cos(yy)*std::cos(pp)
        };
        glm::mat4 view=glm::lookAt(eye,glm::vec3(0,0,0.02f),glm::vec3(0,1,0));
        glm::mat4 proj=glm::perspective(glm::radians(37.0f),float(swapChainExtent.width)/float(swapChainExtent.height),0.1f,12.0f);
        proj[1][1]*=-1;
        CameraUBO u{}; u.viewProj=proj*view; u.eyePos=glm::vec4(eye,1);
        u.lightPos=glm::vec4(1.65f+0.2f*std::sin(t*0.5f),1.8f,2.7f,1);
        u.ambient=glm::vec4(0.2f);
        std::memcpy(uniformBuffersMapped[frame],&u,sizeof(u));
    }

    void recordCommandBuffer(VkCommandBuffer cmd,uint32_t imageIndex) {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        if(vkBeginCommandBuffer(cmd,&bi)!=VK_SUCCESS) throw std::runtime_error("Failed to begin command buffer");
        std::array<VkClearValue,2> clear{};
        clear[0].color={{0.012f,0.017f,0.024f,1.0f}}; clear[1].depthStencil={1.0f,0};
        VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO}; rp.renderPass=renderPass; rp.framebuffer=swapChainFramebuffers[imageIndex]; rp.renderArea.offset={0,0}; rp.renderArea.extent=swapChainExtent; rp.clearValueCount=2; rp.pClearValues=clear.data();
        vkCmdBeginRenderPass(cmd,&rp,VK_SUBPASS_CONTENTS_INLINE); vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,graphicsPipeline);
        VkViewport viewport{0,0,float(swapChainExtent.width),float(swapChainExtent.height),0,1}; vkCmdSetViewport(cmd,0,1,&viewport);
        VkRect2D scissor{{0,0},swapChainExtent}; vkCmdSetScissor(cmd,0,1,&scissor);
        VkDeviceSize off=0; vkCmdBindVertexBuffers(cmd,0,1,&vertexBuffer,&off); vkCmdBindIndexBuffer(cmd,indexBuffer,0,VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,pipelineLayout,0,1,&descriptorSets[currentFrame],0,nullptr);
        auto draw=[&](DrawRange r,const glm::mat4& model){ PushConstants pc{model}; vkCmdPushConstants(cmd,pipelineLayout,VK_SHADER_STAGE_VERTEX_BIT,0,sizeof(pc),&pc); vkCmdDrawIndexed(cmd,r.indexCount,1,r.firstIndex,0,0); };
        draw(staticRange,glm::mat4(1));
        float ha,ma,sa; getAngles(ha,ma,sa);
        draw(hourRange,glm::rotate(glm::mat4(1),ha,glm::vec3(0,0,1)));
        draw(minuteRange,glm::rotate(glm::mat4(1),ma,glm::vec3(0,0,1)));
        draw(secondRange,glm::rotate(glm::mat4(1),sa,glm::vec3(0,0,1)));
        vkCmdEndRenderPass(cmd);
        if(vkEndCommandBuffer(cmd)!=VK_SUCCESS) throw std::runtime_error("Failed to record command buffer");
    }

    void drawFrame() {
        vkWaitForFences(device,1,&inFlightFences[currentFrame],VK_TRUE,UINT64_MAX);
        uint32_t imageIndex;
        VkResult result=vkAcquireNextImageKHR(device,swapChain,UINT64_MAX,imageAvailableSemaphores[currentFrame],VK_NULL_HANDLE,&imageIndex);
        if(result==VK_ERROR_OUT_OF_DATE_KHR) { recreateSwapChain(); return; }
        if(result!=VK_SUCCESS && result!=VK_SUBOPTIMAL_KHR) throw std::runtime_error("Failed to acquire swapchain image");
        vkResetFences(device,1,&inFlightFences[currentFrame]);
        vkResetCommandBuffer(commandBuffers[currentFrame],0);
        updateUniformBuffer(currentFrame); recordCommandBuffer(commandBuffers[currentFrame],imageIndex);
        VkSemaphore wait[]={imageAvailableSemaphores[currentFrame]}; VkPipelineStageFlags stages[]={VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT}; VkSemaphore signal[]={renderFinishedSemaphores[currentFrame]};
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.waitSemaphoreCount=1; submit.pWaitSemaphores=wait; submit.pWaitDstStageMask=stages; submit.commandBufferCount=1; submit.pCommandBuffers=&commandBuffers[currentFrame]; submit.signalSemaphoreCount=1; submit.pSignalSemaphores=signal;
        if(vkQueueSubmit(graphicsQueue,1,&submit,inFlightFences[currentFrame])!=VK_SUCCESS) throw std::runtime_error("Failed to submit frame");
        VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR}; present.waitSemaphoreCount=1; present.pWaitSemaphores=signal; present.swapchainCount=1; present.pSwapchains=&swapChain; present.pImageIndices=&imageIndex;
        result=vkQueuePresentKHR(presentQueue,&present);
        if(result==VK_ERROR_OUT_OF_DATE_KHR || result==VK_SUBOPTIMAL_KHR || framebufferResized) { framebufferResized=false; recreateSwapChain(); }
        else if(result!=VK_SUCCESS) throw std::runtime_error("Failed to present frame");
        currentFrame=(currentFrame+1)%MAX_FRAMES_IN_FLIGHT;
    }

    void cleanupSwapChain() {
        vkDestroyImageView(device,depthImageView,nullptr); vkDestroyImage(device,depthImage,nullptr); vkFreeMemory(device,depthImageMemory,nullptr);
        for(auto f:swapChainFramebuffers) vkDestroyFramebuffer(device,f,nullptr);
        for(auto v:swapChainImageViews) vkDestroyImageView(device,v,nullptr);
        vkDestroySwapchainKHR(device,swapChain,nullptr);
    }

    void recreateSwapChain() {
        int w=0,h=0; glfwGetFramebufferSize(window,&w,&h);
        while(w==0 || h==0) { glfwWaitEvents(); glfwGetFramebufferSize(window,&w,&h); }
        vkDeviceWaitIdle(device); cleanupSwapChain(); createSwapChain(); createImageViews(); createDepthResources(); createFramebuffers();
    }

    void mainLoop() {
        while(!glfwWindowShouldClose(window)) { glfwPollEvents(); drawFrame(); }
        vkDeviceWaitIdle(device);
    }

    void cleanup() {
        cleanupSwapChain();
        vkDestroyDescriptorPool(device,descriptorPool,nullptr);
        for(int i=0;i<MAX_FRAMES_IN_FLIGHT;++i) { vkDestroyBuffer(device,uniformBuffers[i],nullptr); vkFreeMemory(device,uniformBuffersMemory[i],nullptr); vkDestroySemaphore(device,renderFinishedSemaphores[i],nullptr); vkDestroySemaphore(device,imageAvailableSemaphores[i],nullptr); vkDestroyFence(device,inFlightFences[i],nullptr); }
        vkDestroyBuffer(device,indexBuffer,nullptr); vkFreeMemory(device,indexBufferMemory,nullptr);
        vkDestroyBuffer(device,vertexBuffer,nullptr); vkFreeMemory(device,vertexBufferMemory,nullptr);
        vkDestroyCommandPool(device,commandPool,nullptr); vkDestroyPipeline(device,graphicsPipeline,nullptr); vkDestroyPipelineLayout(device,pipelineLayout,nullptr); vkDestroyDescriptorSetLayout(device,descriptorSetLayout,nullptr); vkDestroyRenderPass(device,renderPass,nullptr);
        vkDestroyDevice(device,nullptr); vkDestroySurfaceKHR(instance,surface,nullptr); vkDestroyInstance(instance,nullptr);
        glfwDestroyWindow(window); glfwTerminate();
    }
};
}

int main() {
    try { App app; app.run(); }
    catch(const std::exception& e) { std::cerr << "Error: " << e.what() << "\n"; return EXIT_FAILURE; }
    return EXIT_SUCCESS;
}
