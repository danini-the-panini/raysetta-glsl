#version 410
precision highp float;

#define MAX_SPH 500
#define MAX_PLN 100
const int  MAX_OBJS = MAX_SPH+MAX_PLN;
#define MAX_MATS MAX_OBJS
#define MAX_TEX MAX_MATS
#define MAX_IMG 64
#define MAX_PERLIN 8
const int MAX_BVH = MAX_OBJS*2+1;

const float PI = 3.1415926535897932385;

const ivec3 CUBE[8] = ivec3[](
  ivec3(0, 0, 0),
  ivec3(0, 0, 1),
  ivec3(0, 1, 0),
  ivec3(0, 1, 1),
  ivec3(1, 0, 0),
  ivec3(1, 0, 1),
  ivec3(1, 1, 0),
  ivec3(1, 1, 1)
);

float atan2(in float y, in float x) {
  if (x > 0.0) {
    return atan(y/x);
  } else if (x < 0.0) {
    if (y >= 0.0) {
      return atan(y/x)+PI;
    } else {
      return atan(y/x)-PI;
    }
  } else {
    if (y > 0) {
      return PI/2.0;
    } else if (y < 0) {
      return -PI/2.0;
    } else {
      return 0.0/0.0; // NaN
    }
  }
}

vec3 smthstp(vec3 v) {
  return v * v * (vec3(3.0) - 2.0 * v);
}

const float pos_inf = 1.0/0.0;
const float neg_inf = -1.0/0.0;

uniform vec2 res;
uniform vec3 eye;
uniform vec3 tgt;
uniform vec3 vup;
uniform float vfov;
uniform float defocus_angle;
uniform float focus_dist;
uniform int samples;
uniform int depth;

uniform sampler2DArray images;
uniform sampler2D prev_frame;

uniform float time;
uniform uint utime;
uniform uint frame;

#define LAMBERT 0
#define METAL 1
#define GLASS 2
#define LIGHT 3

#define SPHERE 0
#define BVH 1
#define PLANE 2

#define QUAD 0
#define TRI 1

#define SOLID 0
#define CHECKER 1
#define IMAGE 2
#define NOISE 3

#define GRAD 1
#define CUBEMAP 2
#define SPHMAP 3

vec2 scr;
float aspect;
float flen;

vec3 vp_u;
vec3 vp_v;

vec3 px_du;
vec3 px_dv;

vec3 vp_upleft;
vec3 px00;

vec3 cu;
vec3 cv;
vec3 cw;

vec3 df_u;
vec3 df_v;

struct TypeId {
  int id;
  int type;
};

struct AABB {
  vec3 min;
  vec3 max;
};

struct Lambert {
  TypeId tex;
};

struct Light {
  TypeId tex;
};

struct Metal {
  float fuzz;
  TypeId tex;
};

struct Glass {
  float index;
};

struct SolidColor {
  vec3 albedo;
};

struct Checker {
  float inv_scale;
  TypeId even;
  TypeId odd;
};

struct Image {
  int id;
  int width;
  int height;
};

#define POINT_COUNT 256
struct Perlin {
  vec3 randvec[POINT_COUNT];
  ivec3 perm[POINT_COUNT];
};

struct Noise {
  int id;
  float scale;
  int depth;
  int axis;
};

struct Sphere {
  vec3 center;
  vec3 vec;
  float radius;
  TypeId mat;
};

struct Plane {
  vec3 q;
  vec3 u;
  vec3 v;
  vec3 n;
  float d;
  vec3 w;
  int type;
  TypeId mat;
};

struct Hit {
  float t;
  bool front;
  vec3 p;
  vec3 n;
  vec2 uv;
  TypeId mat;
};

struct BVHNode {
  int left_id;
  int left_type;
  int right_id;
  int right_type;
  AABB bbox;
};

struct Gradient {
  vec3 top;
  vec3 bottom;
};

struct CubeMap {
  TypeId ids[6];
};

layout (std140) uniform ObjBlock {
  Sphere spheres[MAX_SPH];
  Plane planes[MAX_PLN];
};

layout (std140) uniform MatBlock {
  Lambert lamberts[MAX_MATS];
  Metal metals[MAX_MATS];
  Glass glass[MAX_MATS];
  Light lights[MAX_MATS];
};

layout (std140) uniform TexBlock {
  SolidColor solid_colors[MAX_TEX];
  Checker checkers[MAX_TEX];
  Image image_tex[MAX_IMG];
  Noise noises[MAX_TEX];
};

layout (std140) uniform PerlinBlock {
  Perlin perlins[MAX_PERLIN];
};

layout (std140) uniform BvhBlock {
  BVHNode bvh[MAX_BVH];
};

uniform int bg_type;
layout (std140) uniform BgBlock {
  SolidColor solid_bg;
  Gradient grad_bg;
  TypeId sphere_map;
  CubeMap cube_map;
};

uniform int bvh_count;
in vec2 uv;
layout (location = 0) out vec4 outColor;

float seed;
float rand() { return fract(sin(seed++)*43758.5453123); }

float rand(float min, float max) {
  return min + (max-min)*rand();
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

vec2 rand_unit_disk() {
  float r = sqrt(rand());
  float t = rand() * 2.0 * PI;
  return vec2(r * cos(t), r * sin(t));
}

bool nearz(vec3 v) {
  return abs(v.x) < 1e-6 && abs(v.y) < 1e-6 && abs(v.z) < 1e-6;
}

struct Ray {
  vec3 orig;
  vec3 dir;
  float tm;
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

vec3 solid_sample(SolidColor tex, vec2 uv, vec3 pt) {
  return tex.albedo;
}

vec3 image_sample(Image tex, vec2 uv, vec3 pt) {
  float u = clamp(uv.x, 0.0, 1.0)*tex.width;
  float v = (1.0 - clamp(uv.y, 0.0, 1.0))*tex.height;

  int iu = clamp(int(floor(u)), 0, tex.width-2);
  int iv = clamp(int(floor(v)), 0, tex.height-2);

  vec4 p1 = texelFetch(images, ivec3(iu, iv, tex.id), 0);
  vec4 p2 = texelFetch(images, ivec3(iu + 1, iv, tex.id), 0);
  vec4 p3 = texelFetch(images, ivec3(iu, iv + 1, tex.id), 0);
  vec4 p4 = texelFetch(images, ivec3(iu + 1, iv + 1, tex.id), 0);

  float fx = u - iu;
  float fy = v - iv;

  return mix(
    mix(p1, p2, fx),
    mix(p3, p4, fx),
    fy
  ).xyz;
}

float perlin_interp(vec3 c[8], vec3 vc) {
  vec3 v = smthstp(vc);
  float sum = 0.0;

  for (int n = 0; n < 8; n++) {
    int i = CUBE[n].x;
    int j = CUBE[n].y;
    int k = CUBE[n].z;

    vec3 w = vc - vec3(i, j, k);
    sum +=
      (i * v.x + (1 - i) * (1 - v.x)) *
      (j * v.y + (1 - j) * (1 - v.y)) *
      (k * v.z + (1 - k) * (1 - v.z)) *
      dot(c[n], w);
  }
  return sum;
}

float perlin_noise(Perlin self, vec3 pt) {
  ivec3 iv = ivec3(floor(pt));
  vec3 vc = fract(pt);

  vec3 c[8];

  for (int n = 0; n < 8; n++) {
    c[n] = self.randvec[
      self.perm[(iv.x + CUBE[n].x) & 255].x ^
      self.perm[(iv.y + CUBE[n].y) & 255].y ^
      self.perm[(iv.z + CUBE[n].z) & 255].z
    ];
  }

  return perlin_interp(c, vc);
}

float turb(Perlin self, vec3 pt, int d) {
  float accum = 0.0;
  vec3 tmp = pt;
  float w = 1.0;

  for (int i = 0; i < d; i++) {
    accum += w * perlin_noise(self, tmp);
    w *= 0.5;
    tmp *= 2.0;
  }

  return abs(accum);
}

vec3 noise_sample(Noise tex, vec2 uv, vec3 pt) {
  Perlin noise = perlins[tex.id];
  if (tex.axis < 0) {
    return vec3(1.0) * turb(noise, pt, tex.depth);
  } else {
    return vec3(0.5) * (1.0 + sin(tex.scale * pt[tex.axis] + 10.0 * turb(noise, pt, tex.depth)));
  }
}

vec3 nochk_tex_sample(TypeId tex, vec2 uv, vec3 pt) {
  if (tex.type == SOLID) {
    return solid_sample(solid_colors[tex.id], uv, pt);
  } else if (tex.type == IMAGE) {
    return image_sample(image_tex[tex.id], uv, pt);
  } else if (tex.type == NOISE) {
    return noise_sample(noises[tex.id], uv, pt);
  } else {
    return vec3(1.0, 0.0, 1.0);
  }
}

vec3 chk_subtex_sample(TypeId tex, vec2 uv, vec3 pt) {
  if (tex.type == CHECKER) {
    // can't recurse checker
    return vec3(1.0, 0.0, 0.0);
  } else {
    return nochk_tex_sample(tex, uv, pt);
  }
}

vec3 checker_sample(Checker tex, vec2 uv, vec3 pt) {
  int x = int(floor(tex.inv_scale * pt.x));
  int y = int(floor(tex.inv_scale * pt.y));
  int z = int(floor(tex.inv_scale * pt.z));

  if ((x + y + z) % 2 == 0) {
    return chk_subtex_sample(tex.even, uv, pt);
  } else {
    return chk_subtex_sample(tex.odd, uv, pt);
  }
}

vec3 tex_sample(TypeId tex, vec2 uv, vec3 pt) {
  if (tex.type == CHECKER) {
    return checker_sample(checkers[tex.id], uv, pt);
  } else {
    return nochk_tex_sample(tex, uv, pt);
  }
}

bool scat_lambert(Ray r_in, Hit hit, out vec3 att, out Ray scat) {
  Lambert mat = lamberts[hit.mat.id];
  vec3 scat_dir = hit.n + rand_unit();
  if (nearz(scat_dir)) scat_dir = hit.n;
  scat = Ray(hit.p, scat_dir, r_in.tm);
  att = tex_sample(mat.tex, hit.uv, hit.p);
  return true;
}

bool scat_metal(Ray r_in, Hit hit, out vec3 att, out Ray scat) {
  Metal mat = metals[hit.mat.id];
  vec3 r = normalize(reflect(r_in.dir, hit.n)) + (mat.fuzz * rand_unit());
  scat = Ray(hit.p, r, r_in.tm);
  att = tex_sample(mat.tex, hit.uv, hit.p);
  return (dot(r, hit.n) > 0.0);
}

float reflectance(float cosine, float ri) {
  // Use Schlick's approximation for reflectance.
  float r0 = (1.0 - ri) / (1.0 + ri);
  r0 = r0*r0;
  return r0 + (1.0-r0)*pow((1.0 - cosine), 5.0);
}

bool scat_glass(Ray r_in, Hit hit, out vec3 att, out Ray scat) {
  Glass mat = glass[hit.mat.id];
  att = vec3(1.0, 1.0, 1.0);
  float ri = hit.front ? (1.0 / mat.index) : mat.index;

  vec3 unit = normalize(r_in.dir);
  float cos_t = min(dot(-unit, hit.n), 1.0);
  float sin_t = sqrt(1.0 - cos_t*cos_t);

  vec3 dir;

  if (ri * sin_t > 1.0 || reflectance(cos_t, ri) > rand()) {
    dir = reflect(unit, hit.n);
  } else {
    dir = refract(unit, hit.n, ri);
  }

  scat = Ray(hit.p, dir, r_in.tm);
  return true;
}

vec3 emit_light(Light self, vec2 uv, vec3 pt) {
  return tex_sample(self.tex, uv, pt);
}

vec3 emit(Hit hit) {
  if (hit.mat.type == LIGHT) {
    return emit_light(lights[hit.mat.id], hit.uv, hit.p);
  } else {
    return vec3(0.0, 0.0, 0.0);
  }
}

bool scat(Ray r_in, Hit hit, out vec3 att, out Ray scat) {
  if (hit.mat.type == LAMBERT) {
    return scat_lambert(r_in, hit, att, scat);
  } else if (hit.mat.type == METAL) {
    return scat_metal(r_in, hit, att, scat);
  } else if (hit.mat.type == GLASS) {
    return scat_glass(r_in, hit, att, scat);
  } else {
    return false;
  }
}

bool hit_aabb(AABB self, Ray r, Range ray_t) {
  vec3 orig = r.orig;
  vec3 dir = r.dir;

  for (int axis = 0; axis < 3; axis++) {
    Range ax = Range(self.min[axis], self.max[axis]);
    float adinv = 1.0 / dir[axis];

    float t0 = (ax.min - orig[axis]) * adinv;
    float t1 = (ax.max - orig[axis]) * adinv;

    if (t0 < t1) {
      if (t0 > ray_t.min) ray_t.min = t0;
      if (t1 < ray_t.max) ray_t.max = t1;
    } else {
      if (t1 > ray_t.min) ray_t.min = t1;
      if (t0 < ray_t.max) ray_t.max = t0;
    }

    if (ray_t.max <= ray_t.min) {
      return false;
    }
  }
  return true;
}

vec2 sphere_uv(vec3 p) {
  float theta = acos(-p.y);
  float phi = atan2(-p.z, p.x) + PI;

  return vec2(phi / (2.0*PI), theta / PI);
}

bool hit_sphere(Sphere self, Ray ray, Range ray_t, out Hit hit) {
  vec3 center = self.center + self.vec * ray.tm;
  vec3 oc = center - ray.orig;
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
  vec3 n = (point - center) / self.radius;
  set_face_normal(ray, n, hit);
  hit.uv = sphere_uv(n);
  hit.mat = self.mat;

  return true;
}

bool quad_interior(float a, float b, out vec2 uv) {
  Range unit = Range(0.0, 1.0);

  if (!surrounds(unit, a) || !surrounds(unit, b)) return false;
  
  uv = vec2(a, b);
  return true;
}

bool tri_interior(float a, float b, out vec2 uv) {
  if (a < 0.0 || b < 0.0 || a + b > 1.0) return false;

  uv = vec2(a, b);
  return true;
}

bool hit_plane(Plane self, Ray ray, Range ray_t, out Hit hit) {
  float denom = dot(self.n, ray.dir);

  // No hit if the ray is parallel to the plane.
  if (abs(denom) < 1e-6) return false;

  // Return if the hit point parameter t is outside the ray interval.
  float t = (self.d - dot(self.n, ray.orig)) / denom;
  if (!surrounds(ray_t, t)) return false;

  vec3 inter = ray_at(ray, t);
  vec3 phv = inter - self.q;
  float a = dot(self.w, cross(phv, self.v));
  float b = dot(self.w, cross(self.u, phv));

  if (self.type == QUAD) {
    if (!quad_interior(a, b, hit.uv)) return false;
  } else if (self.type == TRI) {
    if (!tri_interior(a, b, hit.uv)) return false;
  } else {
    return false;
  }

  hit.p = inter;
  hit.t = t;
  hit.mat = self.mat;
  set_face_normal(ray, self.n, hit);

  return true;
}

bool hit_obj(int type, int id, Ray r, Range rt, out Hit hit) {
  if (type == SPHERE) {
    return hit_sphere(spheres[id], r, rt, hit);
  } else if (type == PLANE) {
    return hit_plane(planes[id], r, rt, hit);
  } else {
    return false;
  }
}

bool hit_bvh(BVHNode self, Ray ray, Range ray_t, out Hit hit) {
  int stack[MAX_BVH];
  int si = 0;
  bool did_hit = false;
  stack[si++] = bvh_count-1;
  float closest = ray_t.max;
  while (si > 0) {
    int id = stack[si-1];
    si--;
    if (hit_aabb(bvh[id].bbox, ray, Range(ray_t.min, closest))) {
      if (bvh[id].left_type == BVH) {
        stack[si++] = bvh[id].right_id;
        stack[si++] = bvh[id].left_id;
      } else {
        bool hit_left = hit_obj(bvh[id].left_type, bvh[id].left_id, ray, Range(ray_t.min, closest), hit);
        bool hit_right = hit_left;
        if (hit_left) closest = hit.t;
        if (bvh[id].right_type != bvh[id].left_type || bvh[id].right_id != bvh[id].left_id) {
          hit_right = hit_obj(bvh[id].right_type, bvh[id].right_id, ray, Range(ray_t.min, closest), hit);
          if (hit_right) closest = hit.t;
        }

        did_hit = did_hit || hit_left || hit_right;
      }
    };
  }
  return did_hit;
}

bool hit_world(Ray ray, Range ray_t, out Hit hit) {
  return (hit_bvh(bvh[bvh_count-1], ray, ray_t, hit));
}

vec3 defocus_disk_sample() {
  vec2 p = rand_unit_disk();
  return eye + (p.x * df_u) + (p.y * df_v);
}

Ray get_ray() {
  vec3 offset = sample_square(uv);
  vec3 px_sample = px00 + ((scr.x + offset.x) * px_du) + ((scr.y + offset.y) * px_dv);
  vec3 ray_orig = defocus_angle <= 0 ? eye : defocus_disk_sample();
  vec3 ray_dir = px_sample - ray_orig;

  return Ray(ray_orig, ray_dir, rand());
}

vec3 sample_cube_map(Ray ray) {
  vec3 v = ray.dir;
  vec3 vabs = abs(v);

  int fi = -1;
  float ma;
  vec2 uv;
  if (vabs.z >= vabs.x && vabs.z >= vabs.y) {
    // front/back
    fi = v.z < 0.0 ? 5 : 4;
    ma = 0.5 / vabs.z;
    uv = vec2(v.z < 0.0 ? -v.x : v.x, v.y);
  } else if (vabs.y >= vabs.x) {
    // top/bottom
    fi = v.y < 0.0 ? 3 : 2;
    ma = 0.5 / vabs.y;
    uv = vec2(v.x, v.y < 0.0 ? v.z : -v.z);
  } else {
    // left/right
    fi = v.x < 0.0 ? 1 : 0;
    ma = 0.5 / vabs.x;
    uv = vec2(v.x < 0.0 ? v.z : -v.z, v.y);
  }

  uv = uv * ma + vec2(0.5);
  return tex_sample(cube_map.ids[fi], uv, normalize(v));
}

vec3 bg_color(Ray ray) {
  if (bg_type == SOLID) {
    return solid_bg.albedo;
  } else if (bg_type == GRAD) {
    vec3 unit_dir = normalize(ray.dir);
    float a = 0.5*(unit_dir.y + 1.0);
    return mix(grad_bg.bottom, grad_bg.top, a);
  } else if (bg_type == SPHMAP) {
    vec3 unit_dir = normalize(ray.dir);
    vec2 uv = sphere_uv(unit_dir);
    return tex_sample(sphere_map, uv, unit_dir);
  } else if (bg_type == CUBEMAP) {
    return sample_cube_map(ray);
  } else {
    return vec3(1.0, 0.0, 0.0);
  }
}

bool ray_color1(Ray ray, out Ray ray2, out vec3 att, out vec3 em) {
  Hit hit;
  if (!hit_world(ray, Range(0.001, pos_inf), hit)) {
    em = bg_color(ray);
    return false;
  }
  em = emit(hit);
  if (!scat(ray, hit, att, ray2)) {
    return false;
  }
  return true;
}

vec3 ray_color(Ray ray) {
  Ray ray2 = ray;
  vec3 total_att = vec3(1.0);
  vec3 acc = vec3(0.0);

  vec3 att, em;
  for (int i = 0; i < depth; i++) {
    if (ray_color1(ray, ray2, att, em)) {
      acc += em*total_att;
      total_att *= att;
      ray = ray2;
    } else {
      acc += em*total_att;
      break;
    }
  }
  return acc*total_att;
}

void main() {
  seed = time + res.y * gl_FragCoord.x / res.x + gl_FragCoord.y / res.y;

  scr = uv * res;

  aspect = res.x / res.y;
  float theta = radians(vfov);
  float h = tan(theta/2.0);
  float vp_h = 2.0 * h * focus_dist;
  float vp_w = vp_h * aspect;

  cw = normalize(eye - tgt);
  cu = normalize(cross(vup, cw));
  cv = cross(cw, cu);

  vp_u = vp_w*cu;
  vp_v = vp_h*-cv;

  px_du = vp_u / res.x;
  px_dv = vp_v / res.y;

  vp_upleft = eye - (focus_dist*cw) - vp_u/2.0 - vp_v/2.0;
  px00 = vp_upleft + 0.5 * (px_du + px_dv);

  float df_r = focus_dist * tan(radians(defocus_angle / 2.0));
  df_u = cu * df_r;
  df_v = cv * df_r;

  vec3 px_col = vec3(0.0, 0.0, 0.0);
  for (int s = 0; s < samples; s++) {
    px_col = px_col + ray_color(get_ray());
  }
  if (frame == 0) {
    outColor = vec4(px_col / float(samples), 1.0);
  } else {
    vec4 accum = texture(prev_frame, gl_FragCoord.xy / res.xy);
    float total_samples = accum.a + float(samples);
    outColor = vec4((accum.rgb * accum.a + px_col)/total_samples, total_samples);
  }
}
