#version 450

layout(binding = 0) uniform sampler1DArray texSampler;
layout(binding = 1) uniform Layer {
    int layer;
    int count;
} l;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(texSampler, vec2(fragTexCoord.x, fragTexCoord.y*l.layer));
}
