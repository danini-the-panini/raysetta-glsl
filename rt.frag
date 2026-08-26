#version 410
precision highp float;

const float pos_inf = 1e38;
const float neg_inf = -1e38;

uniform vec2 res;
uniform vec3 cam_eye;
uniform int samples;

uniform int noise_size;
uniform sampler1D noise;

int rand_index;

float rand() {
  vec4 r = texture(noise, float(rand_index) / float(noise_size) + 0.5);
  rand_index++;
  return r.x;
}

vec3 sample_square(vec2 st) {
  return vec3(rand() - 0.5, rand() - 0.5, 0.0);
}

struct Ray {
  vec3 orig;
  vec3 dir;
};

struct Range {
  float min;
  float max;
};

float length(Range self) {
  return self.max - self.min;
}

bool surrounds(Range self, float x) {
  return self.min < x && x < self.max;
}

float clamp(Range self, float x) {
  if (x < self.min) return self.min;
  if (x > self.max) return self.max;
  return x;
}

struct Hit {
  float t;
  bool front;
  vec3 p;
  vec3 n;
};

struct Sphere {
  vec3 center;
  float radius;
};

layout (std140) uniform ObjectBlock {
  Sphere spheres [100];
};

uniform int sphere_count;
in vec2 uv;
layout (location = 0) out vec4 outColor;

vec3 ray_at(Ray self, float t) {
  return self.orig + t*self.dir;
}

void set_face_normal(Ray ray, vec3 n, inout Hit hit) {
  hit.front = dot(ray.dir, n) < 0.0;
  hit.n = hit.front ? n : -n;
}

bool hit_sphere(Sphere self, Ray ray, Range ray_t, out Hit hit) {
  vec3 oc = self.center - ray.orig;
  float a = dot(ray.dir, ray.dir);
  float h = dot(ray.dir, oc);
  float c = dot(oc, oc) - self.radius*self.radius;

  float d = h*h - a*c;
  if (d < 0.0) {
    return false;
  }

  float sqrtd = sqrt(d);
  float root = (h - sqrtd) / a;
  if (!surrounds(ray_t, root)) {
    root = (h + sqrtd) / a;
    if (!surrounds(ray_t, root)) {
        return false;
    }
  }

  hit.t = root;
  vec3 point = ray_at(ray, root);
  hit.p = point;
  vec3 n = (point - self.center) / self.radius;
  set_face_normal(ray, n, hit);

  return true;
}

bool notfin(float v) {
  return isinf(v) || isnan(v);
}

bool hit_world(Ray ray, Range ray_t, out Hit hit) {
  Hit tmp;
  bool did_hit = false;
  float closest = ray_t.max;

  for (int i = 0; i < sphere_count; i++) {
    if (hit_sphere(spheres[i], ray, Range(ray_t.min, closest), tmp)) {
      did_hit = true;
      closest = tmp.t;
      hit = tmp;
    }
  }

  return did_hit;
}

vec2 scr;
float aspect;
float focal_length;
float vp_height;
float vp_width;

vec3 vp_u;
vec3 vp_v;

vec3 px_du;
vec3 px_dv;

vec3 vp_upleft;
vec3 px00;

Ray get_ray() {
  vec3 offset = sample_square(uv);
  vec3 px_sample = px00 + ((scr.x + offset.x) * px_du) + ((scr.y + offset.y) * px_dv);
  vec3 ray_orig = cam_eye;
  vec3 ray_dir = px_sample - ray_orig;

  return Ray(ray_orig, ray_dir);
}

vec3 ray_color(Ray ray) {
  Hit hit;
  if (hit_world(ray, Range(0.0, pos_inf), hit)) {
    return 0.5 * (hit.n + vec3(1.0, 1.0, 1.0));
  }
  vec3 unit_dir = normalize(ray.dir);
  float a = 0.5*(unit_dir.y + 1.0);
  return (1.0-a)*vec3(1.0, 1.0, 1.0) + a*vec3(0.5, 0.7, 1.0);
}

void main() {
  rand_index = int(floor(gl_FragCoord.x) + floor(res.x * gl_FragCoord.y)) % noise_size;

  scr = uv * res;

  aspect = res.x / res.y;
  focal_length = 1.0;
  vp_height = 2.0;
  vp_width = vp_height * aspect;

  vp_u = vec3(vp_width, 0.0, 0.0);
  vp_v = vec3(0.0, -vp_height, 0.0);

  px_du = vp_u / res.x;
  px_dv = vp_v / res.y;

  vp_upleft = cam_eye - vec3(0.0, 0.0, focal_length) - vp_u/2.0 - vp_v/2.0;
  px00 = vp_upleft + 0.5 * (px_du + px_dv);

  float sample_scale = 1.0 / float(samples);
  vec3 px_col = vec3(0.0, 0.0, 0.0);
  for (int s = 0; s < samples; s++) {
    px_col = px_col + ray_color(get_ray());
  }
  outColor = vec4(px_col * sample_scale, 1.0);
}
