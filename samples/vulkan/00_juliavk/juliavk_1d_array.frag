#version 450

layout(binding = 0) uniform sampler1DArray texSampler;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(texSampler, vec2(fragTexCoord.x, fragTexCoord.y*512));
}
