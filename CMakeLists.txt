cmake_minimum_required(VERSION 3.20)
project(vsb_vulkan_clock LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(glm CONFIG QUIET)
if(NOT glm_FOUND)
    find_path(GLM_INCLUDE_DIR glm/glm.hpp REQUIRED)
endif()

add_executable(vsb-clock src/main.cpp)
target_link_libraries(vsb-clock PRIVATE Vulkan::Vulkan glfw)
if(glm_FOUND)
    target_link_libraries(vsb-clock PRIVATE glm::glm)
else()
    target_include_directories(vsb-clock PRIVATE ${GLM_INCLUDE_DIR})
endif()

target_compile_options(vsb-clock PRIVATE -Wall -Wextra -Wpedantic)

find_program(GLSLC glslc)
find_program(GLSLANG_VALIDATOR glslangValidator)

set(SHADER_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

function(compile_shader INPUT OUTPUT)
    if(GLSLC)
        add_custom_command(
            OUTPUT ${SHADER_OUTPUT_DIR}/${OUTPUT}
            COMMAND ${GLSLC} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${INPUT} -o ${SHADER_OUTPUT_DIR}/${OUTPUT}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${INPUT}
            VERBATIM)
    elseif(GLSLANG_VALIDATOR)
        add_custom_command(
            OUTPUT ${SHADER_OUTPUT_DIR}/${OUTPUT}
            COMMAND ${GLSLANG_VALIDATOR} -V ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${INPUT} -o ${SHADER_OUTPUT_DIR}/${OUTPUT}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${INPUT}
            VERBATIM)
    else()
        message(FATAL_ERROR "Neither glslc nor glslangValidator was found. Install glslang-tools or shaderc/glslc.")
    endif()
endfunction()

compile_shader(clock.vert clock.vert.spv)
compile_shader(clock.frag clock.frag.spv)
add_custom_target(clock-shaders ALL
    DEPENDS ${SHADER_OUTPUT_DIR}/clock.vert.spv ${SHADER_OUTPUT_DIR}/clock.frag.spv)
add_dependencies(vsb-clock clock-shaders)

target_compile_definitions(vsb-clock PRIVATE CLOCK_SHADER_DIR=\"${SHADER_OUTPUT_DIR}\")

install(TARGETS vsb-clock RUNTIME DESTINATION bin)
install(FILES ${SHADER_OUTPUT_DIR}/clock.vert.spv ${SHADER_OUTPUT_DIR}/clock.frag.spv
        DESTINATION share/vsb-vulkan-clock/shaders OPTIONAL)
