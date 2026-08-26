#version 410
precision highp float;

const float PI = 3.1415926535897932385;

const float pos_inf = 1e38;
const float neg_inf = -1e38;

uniform vec2 res;
uniform vec3 cam_eye;
uniform int samples;

uniform int noise_size;
uniform sampler1D noise;

uniform float time;
uniform uint utime;

#define LAMBERT 0
#define METAL 1

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

struct Lambert {
  vec3 albedo;
};

struct Metal {
  vec3 albedo;
  float fuzz;
};

struct Material {
  int id;
  int type;
};

struct Sphere {
  vec3 center;
  float radius;
  Material mat;
};

struct Hit {
  float t;
  bool front;
  vec3 p;
  vec3 n;
  Material mat;
};

layout (std140) uniform ObjBlock {
  Sphere spheres [100];
};

layout (std140) uniform MatBlock {
  Lambert lamberts [100];
  Metal metals [100];
};

uniform int sphere_count;
in vec2 uv;
layout (location = 0) out vec4 outColor;

int rand_index;
float rand() {
  vec4 r = texture(noise, float(rand_index) / float(noise_size) + 0.5);
  rand_index++; // = int(floor(r.x*float(noise_size))) % noise_size;
  return r.x;
}

float rand(float min, float max) {
  return min + (max-min)*rand();
}

vec3 jitter(vec3 d, float phi, float sina, float cosa) {
  vec3 w = normalize(d), u = normalize(cross(w.yzx, w)), v = cross(w, u);
  return (u*cos(phi) + v*sin(phi)) * sina + w * cosa;
}

vec3 rand3() {
  return vec3(rand(), rand(), rand());
}

vec3 rand3(float min, float max) {
  return vec3(rand(min, max), rand(min, max), rand(min, max));
}

vec2 gauss() {
  float theta = 2.0 * PI * rand();
  float r = sqrt(-2.0 * log(rand()));
  return vec2(r * cos(theta), r * sin(theta));
}

vec3 rand_unit() {
  return normalize(vec3(gauss(), gauss().x));
}

vec3 rand_hemi(vec3 n) {
  vec3 u = rand_unit();
  if (dot(u, n) > 0.0) return u;
  return -u;
}

vec3 sample_square(vec2 st) {
  return vec3(rand() - 0.5, rand() - 0.5, 0.0);
}

bool nearz(vec3 v) {
  return abs(v.x) < 1e-6 && abs(v.y) < 1e-6 && abs(v.z) < 1e-6;
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

vec3 ray_at(Ray self, float t) {
  return self.orig + t*self.dir;
}

void set_face_normal(Ray ray, vec3 n, inout Hit hit) {
  hit.front = dot(ray.dir, n) < 0.0;
  hit.n = hit.front ? n : -n;
}

bool scat_lambert(Ray r_in, Hit hit, out vec3 att, out Ray scat) {
  Lambert mat = lamberts[hit.mat.id];
  vec3 scat_dir = hit.n + rand_unit();
  if (nearz(scat_dir)) scat_dir = hit.n;
  scat = Ray(hit.p, scat_dir);
  att = mat.albedo;
  return true;
}

bool scat_metal(Ray ray_in, Hit hit, out vec3 att, out Ray scat) {
  Metal mat = metals[hit.mat.id];
  vec3 r = normalize(reflect(ray_in.dir, hit.n)) + (mat.fuzz * rand_unit());
  scat = Ray(hit.p, r);
  att = mat.albedo;
  return (dot(r, hit.n) > 0.0);
}

bool scat(Ray ray_in, Hit hit, out vec3 att, out Ray scat) {
  if (hit.mat.type == LAMBERT) {
    return scat_lambert(ray_in, hit, att, scat);
  } else if (hit.mat.type == METAL) {
    return scat_metal(ray_in, hit, att, scat);
  } else {
    return false;
  }
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
  hit.mat = self.mat;

  return true;
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

Ray get_ray() {
  vec3 offset = sample_square(uv);
  vec3 px_sample = px00 + ((scr.x + offset.x) * px_du) + ((scr.y + offset.y) * px_dv);
  vec3 ray_orig = cam_eye;
  vec3 ray_dir = px_sample - ray_orig;

  return Ray(ray_orig, ray_dir);
}

bool ray_color1(Ray ray, out Ray ray2, out vec3 col) {
  Hit hit;
  if (hit_world(ray, Range(0.001, pos_inf), hit)) {
    if (scat(ray, hit, col, ray2)) {
      return true;
    }
    col = vec3(0.0);
    return false;
  }
  vec3 unit_dir = normalize(ray.dir);
  float a = 0.5*(unit_dir.y + 1.0);
  col = (1.0-a)*vec3(1.0, 1.0, 1.0) + a*vec3(0.5, 0.7, 1.0);
  return false;
}

vec3 ray_color(Ray ray) {
  Ray ray2 = ray;
  vec3 att = vec3(1.0, 1.0, 1.0);
  for (int i = 0; i < 10; i++) {
    vec3 col;
    if (ray_color1(ray, ray2, col)) {
      att *= col;
      ray = ray2;
    } else {
      return col*att;
    }
  }
  return vec3(0.0, 0.0, 0.0);
}

void main() {
  rand_index = int(time + floor(gl_FragCoord.x) + floor(res.x * gl_FragCoord.y) + noise_size * time) % noise_size;

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
