#version 450

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec2 inTexCoords;

void main() {
    gl_Position = inPosition;
    fragColor = vec4(1.0);
    fragTexCoord = inTexCoords[gl_VertexIndex];
}
