#version 450

layout(location = 0) out vec4 frag_color;
layout(location = 1) out vec2 frag_tex_coord;
layout(location = 2) out vec4 view_coord;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_tex_coords;

layout(binding = 1) uniform sampler2D u_displacement_map;
layout(binding = 2) uniform ViewData {
    uniform mat4 view_mat;
	uniform mat4 proj_mat;
	uniform vec3 sun_dir;
	uniform vec3 cam_pos;
	uniform int patch_size;
	uniform int tex_size;
} view;

void main()
{
	vec3 ocean_vert = in_position + texture(u_displacement_map, in_tex_coords).rgb * (view.tex_size/view.patch_size);
	view_coord = view.view_mat * vec4(ocean_vert, 1.0);
    gl_Position = view.proj_mat * view_coord;
    frag_color = vec4(1.0);
    frag_tex_coord = in_tex_coords;
}
