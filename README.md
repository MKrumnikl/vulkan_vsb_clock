# VŠB Vulkan Clock

A compact teaching demo: a real 3D analog clock rendered directly with Vulkan on Linux.
The clock face, raised markings, hands and a stylized VŠB logo are procedural geometry;
there are no texture/font assets to load.

## Features

- Vulkan 1.x rendering through GLFW
- perspective camera + depth buffer
- real local system time, including smooth seconds
- physically inspired diffuse/specular lighting
- raised 3D dial, rim, hour/minute ticks and digits
- raised turquoise VŠB logo and bars inspired by the supplied reference
- interactive orbit camera and zoom
- gentle automatic camera motion suitable for a projector/demo
- swapchain recreation on resize
- deliberately small dependency set: Vulkan, GLFW, GLM

## Ubuntu / Debian

```bash
sudo apt update
sudo apt install build-essential cmake libvulkan-dev vulkan-tools \
                 mesa-vulkan-drivers libglfw3-dev libglm-dev glslang-tools
```

For NVIDIA systems, use the normal proprietary NVIDIA driver instead of Mesa's Vulkan driver.
You can quickly verify Vulkan with:

```bash
vulkaninfo --summary
```

Then just run:

```bash
./run.sh
```

## Fedora

```bash
sudo dnf install gcc-c++ cmake vulkan-loader-devel vulkan-tools \
                 glfw-devel glm-devel glslang
./run.sh
```

## Arch Linux

```bash
sudo pacman -S --needed base-devel cmake vulkan-headers vulkan-icd-loader \
             vulkan-tools glfw-x11 glm glslang
./run.sh
```

## Controls

- **Left mouse drag** – rotate camera
- **Mouse wheel** – zoom
- **Space** – automatic subtle orbit on/off
- **R** – reset camera
- **F** – fullscreen on/off
- **Esc** – exit

## Build manually

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/vsb-clock
```

## Teaching notes

The source intentionally keeps the rendering architecture visible instead of hiding Vulkan
behind an engine. Interesting places to discuss in `src/main.cpp`:

1. Vulkan instance/device/surface and queue-family selection
2. swapchain creation and resize handling
3. render pass + depth attachment
4. shader modules and graphics pipeline
5. vertex/index buffers and staging uploads
6. uniform buffer + descriptor set
7. push constants for per-object model matrices
8. procedural mesh generation
9. frame synchronization with semaphores and fences
10. conversion of real local time to 3D hand rotations

The VŠB mark here is a lightweight geometric classroom rendition rather than an official
branding asset, so it stays dependency-free and remains clearly visible in 3D.
