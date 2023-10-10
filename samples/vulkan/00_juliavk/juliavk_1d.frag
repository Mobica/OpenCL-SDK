#version 450

layout(binding = 0) uniform sampler1D texSampler;
layout(binding = 1) uniform Layer {
    int layer;
    int count;
} l;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    float coord = fragTexCoord.x * l.layer + fragTexCoord.y * l.layer * l.layer;
    coord = coord / (l.layer * l.layer);
    outColor = texture(texSampler, coord);
}
