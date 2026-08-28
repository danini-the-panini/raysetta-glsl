#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>

#define GLAD_GL_IMPLEMENTATION
#include "gl.h"

#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "linmath.h"
#include "camera.h"

#define WIDTH 400
#define HEIGHT 200

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

#define MAX_SPH 500
#define MAX_PLN 100
#define MAX_MATS MAX_OBJS
#define MAX_TEX MAX_MATS
#define MAX_IMG 64
#define MAX_PERLIN 8

static const int MAX_OBJS = MAX_SPH+MAX_PLN;
static const int MAX_BVH = MAX_OBJS*2+1;

typedef vec4 aabb[2];

typedef struct bvh_node {
  int left_id;
  int left_type;
  int right_id;
  int right_type;
  aabb bbox;
} BVHNode;

typedef struct type_id {
  int id;
  int type;
  float __1[2];
} TypeId;

typedef struct sphere {
  vec3 center;
  float __0;
  vec3 vec;
  float radius;
  TypeId mat;
} Sphere;

typedef struct plane {
  vec3 q;
  float __0;
  vec3 u;
  float __1;
  vec3 v;
  float __2;
  vec3 n;
  float d;
  vec3 w;
  int type;
  TypeId mat;
} Plane;

typedef struct object {
  int id;
  int type;
} Object;

typedef struct lambert {
  TypeId tex;
} Lambert;

typedef struct metal {
  float fuzz;
  float __0[3];
  TypeId tex;
} Metal;

typedef struct glass {
  float index;
  float __0[3];
} Glass;

typedef struct light {
  TypeId tex;
} Light;

typedef struct solid_color {
  vec3 albedo;
  float __0;
} SolidColor;

typedef struct checker {
  float inv_scale;
  float __0[3];
  TypeId even;
  TypeId odd;
} Checker;

typedef struct image_tex {
  int id;
  int width;
  int height;
  float __0;
} ImageTex;

#define POINT_COUNT 256
typedef int ivec3[3];
typedef int ivec4[4];
typedef struct perlin {
  vec4 randvec[POINT_COUNT];
  ivec4 perm[POINT_COUNT];
} Perlin;

typedef struct noise {
  int id;
  float scale;
  int depth;
  int axis;
} Noise;

static inline void glErrorCheck(const char *msg) {
  GLenum err = glGetError();
  if (err == GL_NO_ERROR) return;

  switch (err) {
  case GL_INVALID_ENUM: printf("%s: GL_INVALID_ENUM\n", msg); break;
  case GL_INVALID_VALUE: printf("%s: GL_INVALID_VALUE\n", msg); break;
  case GL_INVALID_OPERATION: printf("%s: GL_INVALID_OPERATION\n", msg); break;
  case GL_INVALID_FRAMEBUFFER_OPERATION: printf("%s: GL_INVALID_FRAMEBUFFER_OPERATION\n", msg); break;
  case GL_OUT_OF_MEMORY: printf("%s: GL_OUT_OF_MEMORY\n", msg); break;
  default: printf("Unknown error %i\n", err); break;
  }
}

static inline void shuffle_perm(ivec4 *array, size_t n) {
  if (n > 1)  {
    size_t i, j, k;
    int t;
    for (i = 0; i < n - 1; i++)  {
      for (k = 0; k < 4; k++) {
        j = i + rand() / (RAND_MAX / (n - i) + 1);
        t = array[j][k];
        array[j][k] = array[i][k];
        array[i][k] = t;
      }
    }
  }
}

static inline unsigned int next_po2(unsigned int v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return ++v;
}

static inline void vec3_set(vec3 v, float x, float y, float z) {
  v[0] = x;
  v[1] = y;
  v[2] = z;
}

static inline void ivec3_set(ivec3 v, int x, int y, int z) {
  v[0] = x;
  v[1] = y;
  v[2] = z;
}

static bool load_shader(const char *filename, GLenum type, GLuint *ret) {
  FILE *file = fopen(filename, "r");
  if (!file) {
    fprintf(stderr, "could not open file %s\n", filename);
    return false;
  }

  fseek(file, 0, SEEK_END);
  size_t length = ftell(file);
  fseek(file, 0, SEEK_SET);
  GLchar* str = malloc(length+1);
  fread(str, 1, length, file);
  str[length] = '\0';

  const GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &str, NULL);
  glCompileShader(shader);
  free(str);
  *ret = shader;
  return true;
}

typedef struct image_data {
  float *data;
  int width;
  int height;
} ImageData;
static ImageData images[MAX_IMG];
static int image_count = 0;
static int max_width = 1;
static int max_height = 1;
static int load_image(const char *filename) {
  if (image_count >= MAX_IMG) {
    fprintf(stderr, "maximum number of images reached (%i)\n", MAX_IMG);
    return -1;
  }
  int width, height, channels;
  float *data = stbi_loadf(filename, &width, &height, &channels, 3);
  if (!data) {
    fprintf(stderr, "failed to load image %s\n", filename);
    return -1;
  }
  if (width > max_width) max_width = width;
  if (height > max_height) max_height = height;
  images[image_count].data = data;
  images[image_count].width = width;
  images[image_count].height = height;
  return image_count++;
}

void print_shader_info_log(GLuint shader) {
  GLint len = 0;
  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
  if (len <= 0) return;

  GLchar *log = malloc(len);
  glGetShaderInfoLog(shader, len, &len, log);

  fprintf(stderr, "%s\n", log);
}

void print_program_info_log(GLuint program) {
  GLint len = 0;
  glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
  if (len <= 0) return;

  GLchar *log = malloc(len);
  glGetProgramInfoLog(program, len, &len, log);

  fprintf(stderr, "%s\n", log);
  free(log);
}

bool check_program(GLuint program) {
  GLint link_status = GL_FALSE;
  glGetProgramiv(program, GL_LINK_STATUS, &link_status);
  if (link_status == GL_TRUE) return true;
  return false;
}

static inline float randf() {
  return (float)rand() / (float)RAND_MAX;
}

static inline float randf_r(float min, float max) {
  return min + (max-min)*randf();
}

static inline int randi_r(int min, int max) {
  return (int)(randf_r(min, max+1));
}

static inline void vec3_rand(vec3 r) {
  r[0] = randf();
  r[1] = randf();
  r[2] = randf();
}

static int unibuf_count = 0;
static inline GLuint make_unibuffer(GLuint program, GLsizei size, const char *name) {
  GLuint loc = glGetUniformBlockIndex(program, name);
  glUniformBlockBinding(program, loc, unibuf_count);
  GLuint buf;
  glGenBuffers(1, &buf);
  glBindBuffer(GL_UNIFORM_BUFFER, buf);
  glBufferData(GL_UNIFORM_BUFFER, size, NULL, GL_STATIC_DRAW);
  glBindBufferBase(GL_UNIFORM_BUFFER, unibuf_count, buf);
  glBindBufferRange(GL_UNIFORM_BUFFER, unibuf_count, buf, 0, size);
  glBindBuffer(GL_UNIFORM_BUFFER, 0);
  glErrorCheck("make unibuffer");
  unibuf_count++;
  return buf;
}

static bool clear = true;
static Camera cam;
static double last_mouse[2];
static float orbit_max = 1000.0;
static float orbit_min = 0.1;
static GLuint frame = 0;

static int bvh_count = 0;
static int sphere_count = 0;
static int plane_count = 0;
static int obj_count = 0;
static int lambert_count = 0;
static int light_count = 0;
static int metal_count = 0;
static int glass_count = 0;
static int solid_color_count = 0;
static int checker_count = 0;
static int image_tex_count = 0;
static int perlin_count;
static int noise_count;
static Sphere spheres[MAX_SPH];
static Plane planes[MAX_PLN];
static BVHNode bvh_nodes[MAX_BVH];
static Object objects[MAX_OBJS];
static Lambert lamberts[MAX_MATS];
static Light lights[MAX_MATS];
static Metal metals[MAX_MATS];
static Glass glass[MAX_MATS];
static SolidColor solid_colors[MAX_TEX];
static Checker checkers[MAX_TEX];
static ImageTex image_tex[MAX_IMG];
static Perlin perlins[MAX_PERLIN];
static Noise noises[MAX_TEX];

static inline void aabb_copy(aabb r, aabb const b) {
  vec3_copy(r[0], b[0]);
  vec3_copy(r[1], b[1]);
}

static inline void aabb_padmin(aabb self) {
  float delta = 0.0001; 
  float pad = delta / 2.0;
  float size;
  for (int i = 0; i < 3; i++) {
    size = self[1][i] - self[0][i];
    if (size < delta) {
      self[0][i] -= pad;
      self[1][i] += pad;
    }
  }
}

static inline void aabb_add(aabb r, aabb const a, aabb const b) {
  r[0][0] = fminf(a[0][0], b[0][0]);
  r[0][1] = fminf(a[0][1], b[0][1]);
  r[0][2] = fminf(a[0][2], b[0][2]);

  r[1][0] = fmaxf(a[1][0], b[1][0]);
  r[1][1] = fmaxf(a[1][1], b[1][1]);
  r[1][2] = fmaxf(a[1][2], b[1][2]);

  aabb_padmin(r);
}

static inline void aabb_make(aabb r, vec3 const a, vec3 const b) {
  r[0][0] = fminf(a[0], b[0]);
  r[0][1] = fminf(a[1], b[1]);
  r[0][2] = fminf(a[2], b[2]);

  r[1][0] = fmaxf(a[0], b[0]);
  r[1][1] = fmaxf(a[1], b[1]);
  r[1][2] = fmaxf(a[2], b[2]);
  aabb_padmin(r);
}

static inline int aabb_longest(aabb const a) {
  float xsize = a[1][0] - a[0][0];
  float ysize = a[1][1] - a[0][1];
  float zsize = a[1][2] - a[0][2];

  if (xsize > ysize) {
    return xsize > zsize ? 0 : 2;
  } else {
    return ysize > zsize ? 1 : 2;
  }
}

static inline void sphere_bbox(aabb r, const Sphere *sph) {
  vec3 rvec = (vec3){sph->radius, sph->radius, sph->radius};
  aabb box0, box1;  vec3 a, b, center2;
  vec3_sub(a, sph->center, rvec);
  vec3_add(b, sph->center, rvec);
  aabb_make(box0, a, b);
  vec3_add(center2, sph->center, sph->vec);
  vec3_sub(a, center2, rvec);
  vec3_add(b, center2, rvec);
  aabb_make(box1, a, b);
  aabb_add(r, box0, box1);
}

static inline void quad_bbox(aabb r, const Plane *pln) {
  aabb d1, d2;
  vec3 a, b;
  vec3_add(a, pln->q, pln->u);
  vec3_add(b, a, pln->v);
  aabb_make(d1, pln->q, b);
  vec3_add(b, pln->q, pln->v);
  aabb_make(d2, a, b);
  aabb_add(r, d1, d2);
}

static inline void tri_bbox(aabb r, const Plane *pln) {
  aabb d1, d2;
  vec3 a;
  vec3_add(a, pln->q, pln->u);
  aabb_make(d1, pln->q, a);
  vec3_add(a, pln->q, pln->v);
  aabb_make(d2, pln->q, a);
  aabb_add(r, d1, d2);
}

static inline void plane_bbox(aabb r, const Plane *pln) {
  switch (pln->type) {
  case QUAD: quad_bbox(r, pln); break;
  case TRI: tri_bbox(r, pln); break;
  }
}

static inline void obj_bbox(aabb r, int type, int id) {
  switch (type) {
  case SPHERE: sphere_bbox(r, spheres+id); break;
  case PLANE: plane_bbox(r, planes+id); break;
  case BVH: aabb_copy(r, bvh_nodes[id].bbox); break;
  }
}

static inline int make_bvh_node(int left_type, int left_id, int right_type, int right_id) {
  if (left_id < 0 || right_id < 0) {
    fprintf(stderr, "invalid obj id (%i)\n'", left_id < 0 ? left_id : right_id);
    return -1;
  }
  if (bvh_count >= MAX_BVH) {
    fprintf(stderr, "maximum number of bvh nodes reached (%i)\n", MAX_BVH);
    return -1;
  }

  bvh_nodes[bvh_count].left_type = left_type;
  bvh_nodes[bvh_count].left_id = left_id;
  bvh_nodes[bvh_count].right_type = right_type;
  bvh_nodes[bvh_count].right_id = right_id;
  aabb lbox, rbox;
  obj_bbox(lbox, left_type, left_id);
  obj_bbox(rbox, right_type, right_id);
  aabb_add(bvh_nodes[bvh_count].bbox, lbox, rbox);
  return bvh_count++;
}

typedef int (*cmpfn)(const void*, const void*);

static int cmp_axis(const Object *a, const Object *b, int axis) {
  aabb box0, box1;
  obj_bbox(box0, a->type, a->id);
  obj_bbox(box1, b->type, b->id);
  return box0[0][axis] < box1[0][axis] ? -1 : 1;
}

static int cmp_x(const void *a, const void *b) { return cmp_axis(a, b, 0); }
static int cmp_y(const void *a, const void *b) { return cmp_axis(a, b, 1); }
static int cmp_z(const void *a, const void *b) { return cmp_axis(a, b, 2); }

static int make_bvh_nodes(int start, int end) {
  aabb bbox;
  vec3_set(bbox[0], INFINITY, INFINITY, INFINITY);
  vec3_set(bbox[1], -INFINITY, -INFINITY, -INFINITY);
  for (int i = start; i < end; i++) {
    aabb obox;
    obj_bbox(obox, objects[i].type, objects[i].id);
    aabb_add(bbox, bbox, obox);
  }
  int axis = aabb_longest(bbox);

  cmpfn cmp = axis == 0 ? cmp_x : (axis == 1 ? cmp_y : cmp_z);

  int span = end - start;

  if (span == 1) {
    return make_bvh_node(objects[start].type, objects[start].id, objects[start].type, objects[start].id);
  } else if (span == 2) {
    return make_bvh_node(objects[start].type, objects[start].id, objects[start+1].type, objects[start+1].id);
  } else {
    qsort(objects+start, span, sizeof(Object), cmp);
    int mid = start + span / 2;
    int l = make_bvh_nodes(start, mid);
    int r = make_bvh_nodes(mid, end);
    if (l < 0 || r < 0) {
      return -1;
    }
    return make_bvh_node(BVH, l, BVH, r);
  }
}

static inline int make_moving_sphere(vec3 center1, vec3 center2, float radius, int mat_type, int mat_id) {
  if (mat_id < 0) {
    fprintf(stderr, "invalid material id (%i)\n'", mat_id);
    return -1;
  }
  if (sphere_count >= MAX_SPH) {
    fprintf(stderr, "maximum number of spheres reached (%i)\n", MAX_SPH);
    return -1;
  }
  if (obj_count >= MAX_OBJS) {
    fprintf(stderr, "maximum number of objects reached (%i)\n", MAX_OBJS);
    return -1;
  }
  vec3_copy(spheres[sphere_count].center, center1);
  vec3_sub(spheres[sphere_count].vec, center2, center1);
  spheres[sphere_count].radius = radius;
  spheres[sphere_count].mat.type = mat_type;
  spheres[sphere_count].mat.id = mat_id;
  objects[obj_count].type = SPHERE;
  objects[obj_count].id = sphere_count;
  obj_count++;
  return sphere_count++;
}

static inline int make_sphere(vec3 center, float radius, int mat_type, int mat_id) {
  return make_moving_sphere(center, center, radius, mat_type, mat_id);
}

static inline int make_plane(vec3 q, vec3 u, vec3 v, int plane_type, int mat_type, int mat_id) {
  if (mat_id < 0) {
    fprintf(stderr, "invalid material id (%i)\n'", mat_id);
    return -1;
  }
  if (plane_count >= MAX_PLN) {
    fprintf(stderr, "maximum number of planes reached (%i)\n", MAX_PLN);
    return -1;
  }
  if (obj_count >= MAX_OBJS) {
    fprintf(stderr, "maximum number of objects reached (%i)\n", MAX_OBJS);
    return -1;
  }
  vec3_copy(planes[plane_count].q, q);
  vec3_copy(planes[plane_count].u, u);
  vec3_copy(planes[plane_count].v, v);
  planes[plane_count].mat.type = mat_type;
  planes[plane_count].mat.id = mat_id;
  planes[plane_count].type = plane_type;

  vec3 n;
  vec3_mul_cross(n, u, v);
  vec3_norm(planes[plane_count].n, n);
  planes[plane_count].d = vec3_mul_inner(planes[plane_count].n, q);
  vec3_scale(planes[plane_count].w, n, 1.0 / vec3_mul_inner(n, n));

  objects[obj_count].type = PLANE;
  objects[obj_count].id = plane_count;
  obj_count++;
  return plane_count++;
}

static inline int make_quad(vec3 q, vec3 u, vec3 v, int mat_type, int mat_id) {
  return make_plane(q, u, v, QUAD, mat_type, mat_id);
}

static inline int make_tri(vec3 a, vec3 b, vec3 c, int mat_type, int mat_id) {
  vec3 u, v;
  vec3_sub(u, b, a);
  vec3_sub(v, c, a);
  return make_plane(a, u, v, TRI, mat_type, mat_id);
}

static inline void make_box(vec3 a, vec3 b, int mat_type, int mat_id) {
  vec3 dx, dy, dz, ndx, ndz;

  float min_x = fminf(a[0], b[0]); float max_x = fmaxf(a[0], b[0]);
  float min_y = fminf(a[1], b[1]); float max_y = fmaxf(a[1], b[1]);
  float min_z = fminf(a[2], b[2]); float max_z = fmaxf(a[2], b[2]);

  vec3_set(dx, max_x - min_x, 0.0, 0.0);
  vec3_set(dy, 0.0, max_y - min_y, 0.0);
  vec3_set(dz, 0.0, 0.0, max_z - min_z);

  vec3_scale(ndx, dx, -1.0);
  vec3_scale(ndz, dz, -1.0);

  make_quad((vec3){min_x, min_y, max_z},  dx,  dy, mat_type, mat_id);  // front
  make_quad((vec3){max_x, min_y, max_z}, ndz,  dy, mat_type, mat_id);  // right
  make_quad((vec3){max_x, min_y, min_z}, ndx,  dy, mat_type, mat_id);  // back
  make_quad((vec3){min_x, min_y, min_z},  dz,  dy, mat_type, mat_id);  // left
  make_quad((vec3){min_x, max_y, max_z},  dx, ndz, mat_type, mat_id);  // top
  make_quad((vec3){min_x, min_y, min_z},  dx,  dz, mat_type, mat_id);  // bottom
}

static inline int make_solid_color(vec3 albedo) {
  if (solid_color_count > MAX_TEX) {
    fprintf(stderr, "maximum number of solid colors reached (%i)", MAX_TEX);
    return -1;
  }
  vec3_copy(solid_colors[solid_color_count].albedo, albedo);
  return solid_color_count++;
}

static inline int make_checker(float scale, int even_type, int even_id, int odd_type, int odd_id) {
  if (checker_count > MAX_TEX) {
    fprintf(stderr, "maximum number of checker textures reached (%i)", MAX_TEX);
    return -1;
  }
  checkers[checker_count].inv_scale = 1.0 / scale;
  checkers[checker_count].even.type = even_type;
  checkers[checker_count].even.id = even_id;
  checkers[checker_count].odd.type = odd_type;
  checkers[checker_count].odd.id = odd_id;
  return checker_count++;
}

static inline int make_solid_checker(float scale, vec3 even, vec3 odd) {
  return make_checker(scale, SOLID, make_solid_color(even), SOLID, make_solid_color(odd));
}

static inline int make_image_tex(int id) {
  if (image_tex_count > MAX_IMG) {
    fprintf(stderr, "maximum number of image textures reached (%i)", MAX_IMG);
    return -1;
  }
  if (id < 0) {
    fprintf(stderr, "invalid image id %i\n", id);
    return -1;
  }
  image_tex[image_tex_count].id = id;
  image_tex[image_tex_count].width = images[id].width;
  image_tex[image_tex_count].height = images[id].height;
  return image_tex_count++;
}

static inline int make_image_tex_from_file(const char *filename) {
  return make_image_tex(load_image(filename));
}

static inline int make_perlin() {
  if (perlin_count > MAX_PERLIN) {
    fprintf(stderr, "maximum number of perlin noises reached (%i)", MAX_PERLIN);
    return -1;
  }
  vec3 tmp;
  for (int i = 0; i < POINT_COUNT; i++) {
    vec3_set(tmp, randf_r(-1.0, 1.0), randf_r(-1.0, 1.0), randf_r(-1.0, 1.0));
    vec3_norm(perlins[perlin_count].randvec[i], tmp);
    ivec3_set(perlins[perlin_count].perm[i], i, i, i);
  }
  shuffle_perm(perlins[perlin_count].perm, POINT_COUNT);
  return perlin_count++;
}

static inline int make_noise(int id, float scale, int depth, int axis) {
  if (noise_count > MAX_TEX) {
    fprintf(stderr, "maximum number of noise textures reached (%i)", MAX_PERLIN);
    return -1;
  }
  noises[noise_count].id = id;
  noises[noise_count].scale = scale;
  noises[noise_count].depth = depth;
  noises[noise_count].axis = axis;
  return noise_count++;
}

static inline int make_perlin_noise(float scale, int depth, int axis) {
  return make_noise(make_perlin(), scale, depth, axis);
}

static inline int make_lambert(int tex_type, int tex_id) {
  if (lambert_count >= MAX_MATS) {
    fprintf(stderr, "maximum number of lamberts reached (%i)", MAX_MATS);
    return -1;
  }
  lamberts[lambert_count].tex.id = tex_id;
  lamberts[lambert_count].tex.type = tex_type;
  return lambert_count++;
}

static inline int make_lambert_solid(vec3 albedo) {
  return make_lambert(SOLID, make_solid_color(albedo));
}

static inline int make_light(int tex_type, int tex_id) {
  if (light_count >= MAX_MATS) {
    fprintf(stderr, "maximum number of lights reached (%i)", MAX_MATS);
    return -1;
  }
  lights[light_count].tex.id = tex_id;
  lights[light_count].tex.type = tex_type;
  return light_count++;
}

static inline int make_light_solid(vec3 albedo) {
  return make_light(SOLID, make_solid_color(albedo));
}

static inline int make_metal(int tex_type, int tex_id, float fuzz) {
  if (metal_count >= MAX_MATS) {
    fprintf(stderr, "maximum number of metals reached (%i)", MAX_MATS);
    return -1;
  }
  metals[metal_count].tex.id = tex_id;
  metals[metal_count].tex.type = tex_type;
  metals[metal_count].fuzz = fuzz;
  return metal_count++;
}

static inline int make_metal_solid(vec3 albedo, float fuzz) {
  return make_metal(SOLID, make_solid_color(albedo), fuzz);
}

static inline int make_glass(float index) {
  if (glass_count >= MAX_MATS) {
    fprintf(stderr, "maximum number of glasss reached (%i)", MAX_MATS);
    return -1;
  }
  glass[glass_count].index = index;
  return glass_count++;
}

void on_resize(GLFWwindow *win, int w, int h) {
  clear = true;
}

void on_mouse_move(GLFWwindow *win, double x, double y) {
  float dx = x - last_mouse[0];
  float dy = y - last_mouse[1];

  if (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS) {
    float speed = 0.01;
    rotate_around_with_fixed_up(&cam, cam.tgt, speed * dx, speed * dy);
    clear = true;
  }

  last_mouse[0] = x;
  last_mouse[1] = y;
}

void on_scroll(GLFWwindow *win, double dx, double dy) {
  vec3 vw;
  vec3_sub(vw, cam.tgt, cam.eye);
  float dist = vec3_len(vw);
  float zoom = dist * (1.0 - expf(-dy * 0.01));
  zoom_towards(&cam, cam.tgt, zoom, orbit_min, orbit_max);
  clear = true;
}

int main(void) {
  srand(time(NULL));
  glfwInit();

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "raysetta-gl", NULL, NULL);
  glfwMakeContextCurrent(window);

  int version = gladLoadGL(glfwGetProcAddress);
  printf("GL %d.%d\n", GLAD_VERSION_MAJOR(version), GLAD_VERSION_MINOR(version));

  GLsizei max_tex = 0;
  glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_tex);

  glEnable(GL_FRAMEBUFFER_SRGB);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  float *noise = malloc(sizeof(float)*max_tex);
  for (GLsizei i = 0; i < max_tex; i++) {
    noise[i] = randf();
  }

  GLuint vao;
  glGenVertexArrays(1, &vao);

  GLuint vert, frag;
  if (!load_shader("full_screen.vert", GL_VERTEX_SHADER, &vert)) return 1;
  if (!load_shader("rt.frag", GL_FRAGMENT_SHADER, &frag)) return 1;

  const GLuint program = glCreateProgram();
  glAttachShader(program, vert);
  glAttachShader(program, frag);
  glLinkProgram(program);

  if (!check_program(program)) {
    print_shader_info_log(vert);
    print_shader_info_log(frag);
    print_program_info_log(program);
    return 1;
  }

  glUseProgram(program);
  glErrorCheck("shader");

  GLuint time_loc = glGetUniformLocation(program, "time");
  GLuint utime_loc = glGetUniformLocation(program, "utime");
  GLuint frame_loc = glGetUniformLocation(program, "frame");

  GLuint noise_tex = 0;
  glGenTextures(1, &noise_tex);
  glBindTexture(GL_TEXTURE_1D, noise_tex);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);	
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexStorage1D(GL_TEXTURE_1D, 1, GL_R32F, max_tex);
  glTexSubImage1D(GL_TEXTURE_1D, 0, 0, max_tex, GL_RED, GL_FLOAT, noise);
  glBindTexture(GL_TEXTURE_1D, 0);
  glErrorCheck("noise tex");

  GLuint noise_loc = glGetUniformLocation(program, "noise");
  glUniform1i(noise_loc, 0);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_1D, noise_tex);

  GLuint noise_size_loc = glGetUniformLocation(program, "noise_size");
  glUniform1i(noise_size_loc, max_tex);

  const int samples = 2;
  const int depth = 10;

  GLuint samples_loc = glGetUniformLocation(program, "samples");
  glUniform1i(samples_loc, samples);
  GLuint depth_loc = glGetUniformLocation(program, "depth");
  glUniform1i(depth_loc, depth);

  GLuint res_loc = glGetUniformLocation(program, "res");
  glUniform2f(res_loc, WIDTH, HEIGHT);

  vec3_set(cam.eye, 13.0, 2.0, 3.0);
  vec3_set(cam.tgt, 0.0, 0.0, 0.0);
  vec3_set(cam.vup, 0.0, 1.0, 0.0);
  cam.vfov = 20.0;
  cam.defocus_angle = 0.0;
  cam.focus_dist = 10.0;

  GLuint eye_loc = glGetUniformLocation(program, "eye");
  glUniform3f(eye_loc, cam.eye[0], cam.eye[1], cam.eye[2]);
  GLuint tgt_loc = glGetUniformLocation(program, "tgt");
  glUniform3f(tgt_loc, cam.tgt[0], cam.tgt[1], cam.tgt[2]);
  GLuint vup_loc = glGetUniformLocation(program, "vup");
  glUniform3f(vup_loc, cam.vup[0], cam.vup[1], cam.vup[2]);
  GLuint vfov_loc = glGetUniformLocation(program, "vfov");
  glUniform1f(vfov_loc, cam.vfov);
  GLuint defocus_angle_loc = glGetUniformLocation(program, "defocus_angle");
  glUniform1f(defocus_angle_loc, cam.defocus_angle);
  GLuint focus_dist_loc = glGetUniformLocation(program, "focus_dist");
  glUniform1f(focus_dist_loc, cam.focus_dist);
  glErrorCheck("cam");

  // int gmat = make_glass(1.5);

  // // ground
  // int noise_id = make_perlin_noise(10.0, 3, 2);
  // int ground_mat = make_lambert(CHECKER, make_checker(2.0, SOLID, make_solid_color((vec3){0.2, 0.3, 0.1}), NOISE, noise_id));
  // // int ground_mat = make_lambert(NOISE, noise_id);
  // make_sphere((vec3){0.0, -1000.0, 0.0}, 1000.0, LAMBERT, ground_mat);

  // make_sphere((vec3){0.0, 1.0, 0.0}, 1.0, GLASS, gmat);
  // make_sphere((vec3){-4.0, 1.0, 0.0}, 1.0, LAMBERT, make_lambert(IMAGE, make_image_tex_from_file("moon.png")));
  // // make_sphere((vec3){-4.0, 1.0, 0.0}, 1.0, LAMBERT, make_lambert(NOISE, noise_id));
  // make_sphere((vec3){4.0, 1.0, 0.0}, 1.0, METAL, make_metal(IMAGE, make_image_tex_from_file("earth.png"), 0.0));

  // for (int a = -11; a < 11; a++) {
  //   for (int b = -11; b < 11; b++) {
  //     float choose_mat = randf();
  //     vec3 center;
  //     vec3_set(center, a + 0.9*randf(), 0.2, b + 0.9*randf());

  //     vec3 tmp;
  //     vec3_sub(tmp, center, (vec3){4.0, 0.2, 0.0});
  //     if (vec3_len(tmp) > 0.9) {
  //       if (choose_mat < 0.8) {
  //         // diffuse
  //         vec3 r, c, center2;
  //         vec3_rand(r);
  //         vec3_rand(c);
  //         vec3_mul(c, c, r);
  //         vec3_add(center2, center, (vec3){0.0, randf()*0.5, 0.0});
  //         make_moving_sphere(center, center2, 0.2, LAMBERT, make_lambert_solid(c));
  //       } else if (choose_mat < 0.95) {
  //         // metal
  //         vec3 r, c;
  //         vec3_rand(r);
  //         vec3_rand(c);
  //         vec3_mul(c, c, r);
  //         make_sphere(center, 0.2, METAL, make_metal_solid(c, randf()*0.5));
  //       } else {
  //         // glass
  //         make_sphere(center, 0.2, GLASS, gmat);
  //       }
  //     }
  //   }
  // }

  make_tri((vec3){-3.0, -2.0, 5.0}, (vec3){-3.0, -2.0, 1.0}, (vec3){-3.0, 2.0,  5.0}, LAMBERT, make_lambert_solid((vec3){1.0, 0.2, 0.2}));
  make_quad((vec3){-2.0, -2.0, 0.0}, (vec3){4.0, 0.0,  0.0}, (vec3){0.0, 4.0,  0.0}, LAMBERT, make_lambert_solid((vec3){0.2, 1.0, 0.2}));
  make_quad((vec3){ 3.0, -2.0, 1.0}, (vec3){0.0, 0.0,  4.0}, (vec3){0.0, 4.0,  0.0}, LAMBERT, make_lambert_solid((vec3){0.2, 0.2, 1.0}));
  make_quad((vec3){-2.0,  3.0, 1.0}, (vec3){4.0, 0.0,  0.0}, (vec3){0.0, 0.0,  4.0}, LAMBERT, make_lambert_solid((vec3){1.0, 0.5, 0.0}));
  make_quad((vec3){-2.0, -3.0, 5.0}, (vec3){4.0, 0.0,  0.0}, (vec3){0.0, 0.0, -4.0}, LAMBERT, make_lambert_solid((vec3){0.2, 0.8, 0.8}));

  make_box(
    (vec3){-0.5, -0.5, 1.5},
    (vec3){0.5, 0.5, 2.5},
    LIGHT,
    make_light_solid((vec3){10.0, 10.0, 10.0})
  );

  make_bvh_nodes(0, obj_count);

  GLuint bvh_count_loc = glGetUniformLocation(program, "bvh_count");
  glUniform1i(bvh_count_loc, bvh_count);

  GLuint obj_buf = make_unibuffer(program, sizeof(Sphere)*MAX_SPH+sizeof(Plane)*MAX_PLN, "ObjBlock");
  glBindBuffer(GL_UNIFORM_BUFFER, obj_buf);
  if (sphere_count > 0)
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(Sphere)*sphere_count, spheres);
  if (plane_count> 0)
    glBufferSubData(GL_UNIFORM_BUFFER, sizeof(Sphere)*MAX_SPH, sizeof(Plane)*plane_count, planes);
  glBindBuffer(GL_UNIFORM_BUFFER, 0);
  glErrorCheck("bind obj");

  GLuint mat_buf = make_unibuffer(program,
                    sizeof(Lambert)*MAX_MATS+
                      sizeof(Metal)*MAX_MATS+
                      sizeof(Glass)*MAX_MATS+
                      sizeof(Light)*MAX_MATS,
                    "MatBlock");
  glBindBuffer(GL_UNIFORM_BUFFER, mat_buf);
  if (lambert_count > 0)
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(Lambert)*lambert_count, lamberts);
  if (metal_count > 0) glBufferSubData(GL_UNIFORM_BUFFER,
                  sizeof(Lambert)*MAX_MATS,
                  sizeof(Metal)*metal_count,
                  metals);
  if (glass_count > 0) glBufferSubData(GL_UNIFORM_BUFFER,
                  sizeof(Lambert)*MAX_MATS + sizeof(Metal)*MAX_MATS,
                  sizeof(Glass)*glass_count,
                  glass);
  if (light_count > 0) glBufferSubData(GL_UNIFORM_BUFFER,
                  sizeof(Lambert)*MAX_MATS + sizeof(Metal)*MAX_MATS+sizeof(Glass)*MAX_MATS,
                  sizeof(Light)*light_count,
                  lights);
  glBindBuffer(GL_UNIFORM_BUFFER, 0);

  GLuint bvh_buf = make_unibuffer(program, sizeof(BVHNode)*MAX_BVH, "BvhBlock");
  glBindBuffer(GL_UNIFORM_BUFFER, bvh_buf);
  if (bvh_count > 0)
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(BVHNode)*bvh_count, bvh_nodes);
 ; glBindBuffer(GL_UNIFORM_BUFFER, 0);
  glErrorCheck("bind mat");

  GLuint tex_buf = make_unibuffer(program,
                    sizeof(SolidColor)*MAX_TEX+
                      sizeof(Checker)*MAX_TEX+
                      sizeof(ImageTex)*MAX_IMG+
                      sizeof(Noise)*MAX_TEX,
                    "TexBlock");
  glBindBuffer(GL_UNIFORM_BUFFER, tex_buf);
  if (solid_color_count > 0)
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(SolidColor)*solid_color_count, solid_colors);
  if (checker_count > 0) glBufferSubData(GL_UNIFORM_BUFFER,
                  sizeof(SolidColor)*MAX_TEX,
                  sizeof(Checker)*checker_count, checkers);
  if (image_tex_count > 0) glBufferSubData(GL_UNIFORM_BUFFER,
                  sizeof(SolidColor)*MAX_TEX + sizeof(Checker)*MAX_TEX,
                  sizeof(ImageTex)*image_tex_count, image_tex);
  if (noise_count > 0) glBufferSubData(GL_UNIFORM_BUFFER,
                  sizeof(SolidColor)*MAX_TEX + sizeof(Checker)*MAX_TEX + sizeof(ImageTex)*MAX_IMG,
                  sizeof(Noise)*noise_count, noises);
  glBindBuffer(GL_UNIFORM_BUFFER, 0);
  glErrorCheck("bind tex");

  GLuint perl_buf = make_unibuffer(program, sizeof(Perlin)*MAX_PERLIN, "PerlinBlock");
  glBindBuffer(GL_UNIFORM_BUFFER, perl_buf);
  if (perlin_count > 0)
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(Perlin)*perlin_count, perlins);
  glBindBuffer(GL_UNIFORM_BUFFER, 0);
  glErrorCheck("bind perlin");

  GLuint images_tex = 0;
  glGenTextures(1, &images_tex);
  glBindTexture(GL_TEXTURE_2D_ARRAY, images_tex);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT);	
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glErrorCheck("tex param");
  glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGB32F, next_po2(max_width), next_po2(max_height), image_count < 1 ? 1 : image_count);
  glErrorCheck("tex storage 3d");
  for (int i = 0; i < image_count; i++) {
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, i, images[i].width, images[i].height, 1, GL_RGB, GL_FLOAT, images[i].data);
    printf("i %i\n", i);
    glErrorCheck("tex sub data 3d");
  }
  glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
  glErrorCheck("setup images");

  GLuint images_loc = glGetUniformLocation(program, "images");
  glUniform1i(images_loc, 1);
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D_ARRAY, images_tex);

  glfwGetCursorPos(window, &last_mouse[0], &last_mouse[1]);

  glfwSetFramebufferSizeCallback(window, on_resize);
  glfwSetCursorPosCallback(window, on_mouse_move);
  glfwSetScrollCallback(window, on_scroll);

  glErrorCheck("before draw");

  int nframes = 0;
  float last_time = glfwGetTime();
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    float t = glfwGetTime();
    glUniform1f(time_loc, t);
    glUniform1ui(utime_loc, (GLuint)t);

    nframes++;
    if (t - last_time > 1.0) {
      printf("FPS %i\n", nframes);
      nframes = 0;
      last_time += 1.0;
    }

    glUniform3f(eye_loc, cam.eye[0], cam.eye[1], cam.eye[2]);

    for (GLsizei i = 0; i < max_tex; i++) {
      noise[i] = randf();
    }
    glBindTexture(GL_TEXTURE_1D, noise_tex);
    glTexSubImage1D(GL_TEXTURE_1D, 0, 0, max_tex, GL_RED, GL_FLOAT, noise);

    if (clear) {
      glClearColor(0.7f, 0.9f, 0.1f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);
      int w, h;
      glfwGetFramebufferSize(window, &w, &h);
      glViewport(0, 0, w, h);
      glUniform2f(res_loc, w, h);
      clear = false;
      frame = 0;
    } else {
      glClear(GL_DEPTH_BUFFER_BIT);
    }

    glUniform1ui(frame_loc, frame);
    frame++;

    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glfwSwapBuffers(window);
  }

  glfwTerminate();

  free(noise);
  return 0;
}
