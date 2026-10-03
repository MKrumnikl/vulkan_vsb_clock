#version 450

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 3) in vec3 fragViewDir;
layout(location = 4) in vec3 fragLightDir;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 N = normalize(fragNormal);
    vec3 L = normalize(fragLightDir);
    vec3 V = normalize(fragViewDir);
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 72.0);

    // Bright colored pieces (VSB teal / second hand red) get a tiny emissive lift.
    float chroma = max(fragColor.r, max(fragColor.g, fragColor.b)) -
                   min(fragColor.r, min(fragColor.g, fragColor.b));
    float emissive = smoothstep(0.22, 0.65, chroma) * 0.16;

    vec3 ambient = fragColor * 0.28;
    vec3 diffuse = fragColor * diff * 0.76;
    vec3 specular = vec3(1.0) * spec * 0.42;
    vec3 color = ambient + diffuse + specular + fragColor * emissive;

    // Gentle filmic-ish compression keeps the white dial from clipping.
    color = color / (color + vec3(0.72));
    color = pow(color, vec3(1.0 / 2.2));
    outColor = vec4(color, 1.0);
}
