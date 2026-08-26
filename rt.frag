#version 410
precision highp float;

const float pos_inf = 3e10;
const float neg_inf = -3e10;

struct Ray {
  vec3 orig;
  vec3 dir;
};

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
uniform vec2 res;
in vec2 uv;
layout (location = 0) out vec4 outColor;

vec3 ray_at(Ray self, float t) {
  return self.orig + t*self.dir;
}

void set_face_normal(Ray ray, vec3 n, inout Hit hit) {
  hit.front = dot(ray.dir, n) < 0.0;
  hit.n = hit.front ? n : -n;
}

bool hit_sphere(Sphere self, Ray ray, float ray_tmin, float ray_tmax, out Hit hit) {
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
  if (root <= ray_tmin || ray_tmax <= root) {
    root = (h + sqrtd) / a;
    if (root <= ray_tmin || ray_tmax <= root) {
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

bool hit_world(Ray ray, float ray_tmin, float ray_tmax, out Hit hit) {
  Hit tmp;
  bool did_hit = false;
  float closest = ray_tmax;

  for (int i = 0; i < sphere_count; i++) {
    if (hit_sphere(spheres[i], ray, ray_tmin, closest, tmp)) {
      did_hit = true;
      closest = tmp.t;
      hit = tmp;
    }
  }

  return did_hit;
}

vec3 ray_color(Ray ray) {
  Hit hit;
  if (hit_world(ray, 0.0, pos_inf, hit)) {
    return 0.5 * (hit.n + vec3(1.0, 1.0, 1.0));
  }
  vec3 unit_dir = normalize(ray.dir);
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

  outColor = vec4(ray_color(Ray(cam_eye, ray_dir)), 1.0);
}
