#include "linmath.h"
#include <float.h>
#include <math.h>

typedef struct camera {
  vec3 eye;
  vec3 tgt;
  vec3 vup;
  float vfov;
} Camera;

static inline void rodrigues(vec3 r, vec3 v, vec3 k, float cs, float sn) {
  vec3 b, c;

  vec3_scale(r, v, cs);

  vec3_mul_cross(b, k, v);
  vec3_scale(b, b, sn);

  vec3_scale(c, k, vec3_mul_inner(k, v) * (1.0 - cs));

  vec3_add(r, r, b);
  vec3_add(r, r, c);
}

static void rotate_around_with_fixed_up(Camera *self, vec3 point, float x, float y) {
  // Since rotations in linear algebra always describe rotations about the origin, we
  // subtract the point, do all rotations, and add the point again
  vec3 pos, tgt, up, k_x, k_y, tmp, pos_x, tgt_x, pos_y, tgt_y, dir;
  vec3_sub(pos, self->eye, point);
  vec3_sub(tgt, self->tgt, point);
  vec3_norm(up, self->vup);
  // We use Rodrigues' rotation formula to rotate around the fixed `up` vector and around the
  // horizon which is calculated from the camera's view direction and `up`
  // https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
  vec3_copy(k_x, up);
  vec3_sub(tmp, tgt, pos);
  vec3_mul_cross(k_y, tmp, up);
  vec3_norm(k_y, k_y);
  // Prepare cos and sin terms, inverted because the method rotates left and up while
  // rotations follow the right hand rule
  float cos_x = cosf(-x);
  float sin_x = sinf(-x);
  float cos_y = cosf(-y);
  float sin_y = sinf(-y);
  // Do the rotations following the rotation formula
  rodrigues(pos_x, pos, k_x, cos_x, sin_x);
  rodrigues(tgt_x, tgt, k_x, cos_x, sin_x);
  rodrigues(pos_y, pos_x, k_y, cos_y, sin_y);
  rodrigues(tgt_y, tgt_x, k_y, cos_y, sin_y);
  // Forbid to face the camera exactly up or down, fall back to just rotate in x direction
  vec3_sub(dir, tgt_y, pos_y);
  vec3_norm(dir, dir);
  if (fabsf(vec3_mul_inner(dir, up)) < 0.999) {
    vec3_add(self->eye, pos_y, point);
    vec3_add(self->tgt, tgt_y, point);
  } else {
    vec3_add(self->eye, pos_x, point);
    vec3_add(self->tgt, tgt_x, point);
  }
  vec3_copy(self->vup, up);
}

static inline float clampf(float x, float min, float max) {
  if (x < min) return min;
  if (x > max) return max;
  return x;
}

static inline void zoom_towards(Camera *self, vec3 point, float delta, float min_dist, float max_dist) {
  vec3 vw, tow;
  vec3_sub(vw, self->tgt, self->eye);
  float dist = vec3_len(vw);
  vec3_norm(vw, vw);

  vec3_sub(tow, point, self->eye);
  vec3_norm(tow, tow);

  float cos_t = vec3_mul_inner(vw, tow);
  if (fabsf(cos_t) > FLT_EPSILON) {
    min_dist = fmaxf(min_dist, FLT_EPSILON);
    max_dist = fmaxf(max_dist, min_dist);
    float dclamp = dist - clampf(dist - delta, min_dist, max_dist);
    vec3 a, b;
    vec3_scale(a, vw, dclamp);
    vec3_scale(b, tow, dclamp / cos_t);

    vec3_add(self->eye, self->eye, b);
    vec3_sub(a, b, a);
    vec3_add(self->tgt, self->tgt, a);
  }
}
