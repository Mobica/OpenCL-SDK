#version 450

layout(location = 0) out vec2 frag_tex_coord;
layout(location = 1) out vec4 ec_pos;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_tex_coords;

layout(set = 0, binding = 0) uniform sampler2D u_displacement_map;
layout(std140, set = 0, binding = 2) uniform ViewData {
    uniform mat4    view_mat;
    uniform mat4    proj_mat;
    uniform vec3    sun_dir;
    uniform float   choppiness;
    uniform float   alt_scale;
} view;

void main()
{
    vec3 displ = texture(u_displacement_map, in_tex_coords).rgb;
    displ.xz *= view.choppiness;
    displ.y *= view.alt_scale;
    vec3 ocean_vert = in_position + displ;
    ec_pos = view.view_mat * vec4(ocean_vert, 1.0);
    gl_Position = view.proj_mat * ec_pos;
    frag_tex_coord = in_tex_coords;
}
