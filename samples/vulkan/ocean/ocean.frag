#version 450

layout(location = 0) in vec4 frag_color;
layout(location = 1) in vec2 frag_tex_coord;
layout(location = 2) in vec4 view_coord;

layout(location = 0) out vec4 out_color;

layout(binding = 1) uniform sampler2D u_normal_map;
layout(binding = 2) uniform ViewData {
	uniform mat4 view_proj_mat;
	uniform vec3 sun_dir;
	uniform vec3 cam_pos;
	uniform int ocean_size;
	uniform int tex_size;
} view;


void main() {

#if 0
	vec3 sky_color = vec3(3.2f, 9.6f, 12.8f);
	vec3 ocean_color = vec3(0.004f, 0.016f, 0.047f);
	float exposure = 0.35f;

	vec3 normal = texture(u_normal_map, frag_tex_coord).rgb;

	vec3 view_dir = normalize(view.camera_pos - view_coord);
	float fresnel = 0.02f + 0.98f * pow(1.f - dot(normal, view_dir), 5.f);

	vec3 sky = fresnel * sky_color;
	float diffuse = clamp(dot(normal, normalize(-view.sun_dir)), 0.f, 1.f);
	vec3 water = (1.f - fresnel) * ocean_color * sky_color * diffuse;

	vec3 color = sky + water;

	out_color = vec4(HDR(color, exposure), 1.f);
#else

	vec3 normal = texture(u_normal_map, frag_tex_coord).rgb;
	out_color = vec4( frag_color.rgb * normal, 1.0 );
	//out_color = vec4( 1.0 );

#endif
}
