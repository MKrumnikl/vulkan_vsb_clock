#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(binding = 0) uniform CameraUBO {
    mat4 viewProj;
    vec4 eyePos;
    vec4 lightPos;
    vec4 ambient;
} ubo;

layout(push_constant) uniform Push {
    mat4 model;
} pushData;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;
layout(location = 3) out vec3 fragViewDir;
layout(location = 4) out vec3 fragLightDir;

void main() {
    vec4 worldPos = pushData.model * vec4(inPos, 1.0);
    mat3 normalMat = transpose(inverse(mat3(pushData.model)));

    fragWorldPos = worldPos.xyz;
    fragNormal = normalize(normalMat * inNormal);
    fragColor = inColor;
    fragViewDir = ubo.eyePos.xyz - worldPos.xyz;
    fragLightDir = ubo.lightPos.xyz - worldPos.xyz;
    gl_Position = ubo.viewProj * worldPos;
}
