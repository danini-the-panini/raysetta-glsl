#version 330

uniform vec2 res;
in vec2 uv;
layout (location = 0) out vec4 outColor;

vec3 ray_at(vec3 orig, vec3 dir, float t) {
  return orig + t*dir; 
}

vec3 ray_color(vec3 ray_orig, vec3 ray_dir) {
  vec3 unit_dir = normalize(ray_dir);
  float a = 0.5*(unit_dir.y + 1.0);
  return (1.0-a)*vec3(1.0, 1.0, 1.0) + a*vec3(0.5, 0.7, 1.0);
}

void main() {
  vec2 scr = uv * res;

  float aspect = res.x / res.y;
  float focal_length = 1.0;
  float vp_height = 2.0;
  float vp_width = vp_height * aspect;
  vec3 cam_eye = vec3(0.0, 0.0, 0.0);

  vec3 vp_u = vec3(vp_width, 0.0, 0.0);
  vec3 vp_v = vec3(0.0, -vp_height, 0.0);

  vec3 px_du = vp_u / res.x;
  vec3 px_dv = vp_v / res.y;

  vec3 vp_upleft = cam_eye - vec3(0.0, 0.0, focal_length) - vp_u/2.0 - vp_v/2.0;
  vec3 px00 = vp_upleft + 0.5 * (px_du + px_dv);

  vec3 px = px00 + (scr.x * px_du) + (scr.y * px_dv);
  vec3 ray_dir = px - cam_eye;

  outColor = vec4(ray_color(cam_eye, ray_dir), 1.0);
}
