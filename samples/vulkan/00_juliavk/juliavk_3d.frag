#version 450

layout(binding = 0) uniform sampler3D texSampler;
layout(binding = 1) uniform Layer { int layer; } l;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(texSampler, vec3(fragTexCoord, l.layer));
}
