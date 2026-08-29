#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

#define GLAD_GL_IMPLEMENTATION
#include "gl.h"

#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "json.h"
#include "hashmap.h"
#include "base64.h"

typedef struct json_value_s *JAny;
typedef struct json_object_s *JObj;
typedef struct json_object_element_s *JEntry;
typedef struct json_array_s *JAry;
typedef struct json_array_element_s *JItem;
typedef struct json_string_s *JStr;
typedef struct json_number_s *JNum;

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

#define GRAD 1
#define CUBEMAP 2
#define SPHMAP 3

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

#define TBITS 4

static inline size_t t2i(int type, int id) {
  return (id << TBITS) + type;
}

static inline void i2t(size_t i, int *type, int *id) {
  *id = i >> TBITS;
  *type = i & ((1 << TBITS)-1);
}

static inline void i2ti(size_t i, TypeId *ti) {
  i2t(i, &ti->type, &ti->id);
}

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

typedef struct gradient {
  vec4 top;
  vec4 bottom;
} Gradient;

typedef struct cube_map {
  TypeId tex[6];
} CubeMap;

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

static FILE *open_file(const char *filename, size_t *len) {
  FILE *file = fopen(filename, "r");
  if (!file) {
    fprintf(stderr, "could not open file %s\n", filename);
    return NULL;
  }

  fseek(file, 0, SEEK_END);
  *len = ftell(file);
  fseek(file, 0, SEEK_SET);
  
  return file;
}

static char *load_file(const char *filename, size_t *len) {
  size_t length;
  FILE *file = open_file(filename, &length);
  if (len) *len = length;
  GLchar* str = malloc(length);
  fread(str, 1, length, file);
  fclose(file);
  return str;
}

static char *load_file_cstr(const char *filename) {
  size_t length;
  FILE *file = open_file(filename, &length);
  GLchar* str = malloc(length+1);
  fread(str, 1, length, file);
  str[length] = '\0';
  fclose(file);
  return str;
}

static bool load_shader(const char *filename, GLenum type, GLuint *ret) {
  GLchar *str = load_file_cstr(filename);
  if (!str) return false;

  const GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, (const GLchar**)&str, NULL);
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

static inline double randd() {
  return (double)rand() / (double)RAND_MAX;
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

static SolidColor solid_bg;
static Gradient grad_bg;
static TypeId sphere_map;
static CubeMap cube_map;
static int bg_type = -1;

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
    fprintf(stderr, "maximum number of solid colors reached (%i)\n", MAX_TEX);
    return -1;
  }
  vec3_copy(solid_colors[solid_color_count].albedo, albedo);
  return solid_color_count++;
}

static inline int make_checker(float scale, int even_type, int even_id, int odd_type, int odd_id) {
  if (checker_count > MAX_TEX) {
    fprintf(stderr, "maximum number of checker textures reached (%i)\n", MAX_TEX);
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
    fprintf(stderr, "maximum number of image textures reached (%i)\n", MAX_IMG);
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

static inline void make_solid_bg(vec3 col) {
  vec3_copy(solid_bg.albedo, col);
  bg_type = SOLID;
}

static inline void make_grad_bg(vec3 top, vec3 bottom) {
  vec3_copy(grad_bg.top, top);
  vec3_copy(grad_bg.bottom, bottom);
  bg_type = GRAD;
}

static inline void make_sphere_map(int type, int id) {
  sphere_map.type = type;
  sphere_map.id = id;
  bg_type = SPHMAP;
}

static inline void make_sphere_map_from_file(const char *filename) {
  make_sphere_map(IMAGE, make_image_tex_from_file(filename));
}

static inline void make_cube_map(
  int type0, int id0,
  int type1, int id1,
  int type2, int id2,
  int type3, int id3,
  int type4, int id4,
  int type5, int id5
) {
  cube_map.tex[0].type = type0; cube_map.tex[0].id = id0;
  cube_map.tex[1].type = type1; cube_map.tex[1].id = id1;
  cube_map.tex[2].type = type2; cube_map.tex[2].id = id2;
  cube_map.tex[3].type = type3; cube_map.tex[3].id = id3;
  cube_map.tex[4].type = type4; cube_map.tex[4].id = id4;
  cube_map.tex[5].type = type5; cube_map.tex[5].id = id5;
  bg_type = CUBEMAP;
}

static inline void make_cube_map_from_files(
  const char *f0,
  const char *f1,
  const char *f2,
  const char *f3,
  const char *f4,
  const char *f5
) {
  make_cube_map(
    IMAGE, make_image_tex_from_file(f0),
    IMAGE, make_image_tex_from_file(f1),
    IMAGE, make_image_tex_from_file(f2),
    IMAGE, make_image_tex_from_file(f3),
    IMAGE, make_image_tex_from_file(f4),
    IMAGE, make_image_tex_from_file(f5)
  );
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

#define JSON_OBJ_EACH(obj, key, val, block) if (obj->length > 0) { \
  JEntry __el = (obj)->start; \
  do { \
    JStr key = __el->name; \
    JAny val = __el->value; \
    { block } \
    __el = __el->next; \
  } while (__el); }

#define JSON_OBJ_EACH_CHK(obj, key, val, block) { \
  JOBj __obj = json_value_as_object(obj); \
  if (!__obj) fprintf(stderr, "object is not object\n"); \
  else JSON_OBJ_EACH(__obj, key, val, block); }

#define JSON_ARY_EACH(ary, it, i, block) if (ary->length > 0) { \
    size_t i = 0; \
    JItem __it = (ary)->start; \
    do { \
      JAny it = __it->value; \
      { block } \
      i++; \
      __it = __it->next; \
    } while (__it); }

#define JSON_ARY_EACH_CHK(ary, it, i, block) { \
    JOBj __ary = json_value_as_array(ary); \
    if (!__ary) fprintf(stderr, "array is not array\n"); \
    else JSON_ARY_EACH(__ary, it, i, block) };

static inline JAny json_aref_nochk(JObj obj, const char *key) {
  JSON_OBJ_EACH(obj, k, v, {
    if (0 == strcmp(k->string, key)) {
      return v;
    } });
  return NULL;
}

static inline JAny json_aref(JObj obj, const char *key) {
  JAny val = json_aref_nochk(obj, key);
  if (!val) fprintf(stderr, "obj is missing key '%s'\n", key);
  return val;
}

static inline float json_float(JAny num) {
  JNum jnum = json_value_as_number(num);
  if (!jnum) { fprintf(stderr, "float is not a number\n") ; return NAN; }
  return strtof(jnum->number, NULL);
}

static inline int json_int(JAny num) {
  JNum jnum = json_value_as_number(num);
  if (!jnum) { fprintf(stderr, "int is not a number\n") ; return 0; }
  return strtol(jnum->number, NULL, 10);
}

static inline bool json_vec3(vec3 r, JAny j) {
  JAry vec = json_value_as_array(j);
  if (!vec) { fprintf(stderr, "vec3 is not an array\n"); return false; }
  if (vec->length != 3) { fprintf(stderr, "vec3 does not have 3 components\n"); return false; }
  JItem i = vec->start;
  r[0] = json_float(i->value);
  i = i->next;
  r[1] = json_float(i->value);
  i = i->next;
  r[2] = json_float(i->value);
  return true;
}

typedef struct hashmap_s Hash;

#define HNEW(name, size) \
  if (0 != hashmap_create((size), (&(name)))) { \
    fprintf(stderr, "failed to create hash " #name "\n"); \
  }

#define HPUT(hash, key, val) \
  if (0 != hashmap_put(&(hash), (key)->string, (key)->string_size, (void*)(size_t)(val))) { \
    fprintf(stderr, "failed to put key %s into hashmap " #hash "\n", (key)->string); \
  }

#define HGET(hash, key) (size_t)hashmap_get(&(hash), (key)->string, (key)->string_size)
#define HGETR(hash, jsn, key) HGET(hash, json_value_as_string(json_aref((jsn), #key)))

static Hash noises_h;
static Hash images_h;
static Hash textures_h;
static Hash materials_h;

static int parse_noise(JObj json) {
  JSON_ARY_EACH(json_value_as_array(json_aref(json, "randvec")), it, i, {
    json_vec3(perlins[perlin_count].randvec[i], it);
  });
  JSON_ARY_EACH(json_value_as_array(json_aref(json, "perm_x")), it, i, {
    perlins[perlin_count].perm[i][0] = json_int(it);
  });
  JSON_ARY_EACH(json_value_as_array(json_aref(json, "perm_y")), it, i, {
    perlins[perlin_count].perm[i][1] = json_int(it);
  });
  JSON_ARY_EACH(json_value_as_array(json_aref(json, "perm_z")), it, i, {
    perlins[perlin_count].perm[i][2] = json_int(it);
  });
  return perlin_count++;
}

static int parse_image(JObj json) {
  JStr data_url = json_value_as_string(json_aref(json, "data"));
  char *encoded = strchr(data_url->string, ',')+1;
  size_t len = 0;
  unsigned char *decoded = base64_decode(encoded, data_url->string_size - (encoded - data_url->string), &len);
  int width, height, channels;
  float *data = stbi_loadf_from_memory(decoded, len, &width, &height, &channels, 3);
  if (!data) {
    free(decoded);
    fprintf(stderr, "failed to load image\n");
    return -1;
  }
  if (width > max_width) max_width = width;
  if (height > max_height) max_height = height;
  images[image_count].data = data;
  images[image_count].width = width;
  images[image_count].height = height;
  free(decoded);
  return image_count++;
}

static int parse_texture(JObj json);

static int parse_solid_color(JObj json) {
  json_vec3(
    solid_colors[solid_color_count].albedo,
    json_aref(json, "albedo")
  );
  return solid_color_count++;
}

static int parse_checker_texture(JObj json) {
  checkers[checker_count].inv_scale = 1.0 / json_float(json_aref(json, "scale"));
  i2ti(parse_texture(json_value_as_object(json_aref(json, "even"))), &checkers[checker_count].even);
  i2ti(parse_texture(json_value_as_object(json_aref(json, "odd"))), &checkers[checker_count].odd);
  return checker_count++;
}

static int parse_image_texture(JObj json) {
  int id = HGETR(images_h, json, image);
  ImageData img = images[id];
  image_tex[image_tex_count].id = id;
  image_tex[image_tex_count].width = img.width;
  image_tex[image_tex_count].height = img.height;
  return image_tex_count++;
}

static int parse_noise_texture(JObj json) {
  noises[noise_count].scale = json_float(json_aref(json, "scale"));
  noises[noise_count].depth = json_int(json_aref(json, "depth"));
  JStr jaxis = json_value_as_string(json_aref(json, "marble_axis"));
  int axis = -1;
  if (jaxis) {
    if (0 == strcmp(jaxis->string, "x")) {
      axis = 0;
    } else if (0 == strcmp(jaxis->string, "y")) {
      axis = 1;
    } else if (0 == strcmp(jaxis->string, "z")) {
      axis = 2;
    }
  }
  noises[noise_count].axis = axis;
  noises[noise_count].id = HGETR(noises_h, json, noise);
  return noise_count++;
}

static int parse_texture(JObj json) {
  JStr type_str = json_value_as_string(json_aref(json, "type"));
  int type = -1;
  int id = -1;
  if (0 == strcmp(type_str->string, "SolidColor")) {
    type = SOLID;
    id = parse_solid_color(json);
  } else if (0 == strcmp(type_str->string, "Checker")) {
    type = CHECKER;
    id = parse_checker_texture(json);
  } else if (0 == strcmp(type_str->string, "Image")) {
    type = IMAGE;
    id = parse_image_texture(json);
  } else if (0 == strcmp(type_str->string, "Noise")) {
    type = NOISE;
    id = parse_noise_texture(json);
  } else {
    fprintf(stderr, "unknown texture type %s\n", type_str->string);
  }
  if (type < 0 || id < 0) return -1;
  return t2i(type, id);
}

static int parse_lambert(JObj json) {
  i2ti(HGETR(textures_h, json, texture), &lamberts[lambert_count].tex);
  return lambert_count++;
}

static int parse_metal(JObj json) {
  i2ti(HGETR(textures_h, json, texture), &metals[metal_count].tex);
  metals[metal_count].fuzz = json_float(json_aref(json, "fuzz"));
  return metal_count++;
}

static int parse_glass(JObj json) {
  glass[glass_count].index = json_float(json_aref(json, "refraction_index"));
  return glass_count++;
}

static int parse_light(JObj json) {
  i2ti(HGETR(textures_h, json, texture), &lights[light_count].tex);
  return light_count++;
}

static int parse_material(JObj json) {
  JStr type_str = json_value_as_string(json_aref(json, "type"));
  int type = -1;
  int id = -1;
  if (0 == strcmp(type_str->string, "Lambertian")) {
    type = LAMBERT;
    id = parse_lambert(json);
  } else if (0 == strcmp(type_str->string, "Metal")) {
    type = METAL;
    id = parse_metal(json);
  } else if (0 == strcmp(type_str->string, "Dielectric")) {
    type = GLASS;
    id = parse_glass(json);
  } else if (0 == strcmp(type_str->string, "DiffuseLight")) {
    type = LIGHT;
    id = parse_light(json);
  } else {
    fprintf(stderr, "unknown material type %s\n", type_str->string);
  }
  if (type < 0 || id < 0) return -1;
  return t2i(type, id);
}

static int parse_sphere(JObj obj) {
  vec3 center;
  json_vec3(center, json_aref(obj, "center"));
  int mat_type, mat_id;
  i2t(HGETR(materials_h, obj, material), &mat_type, &mat_id);
  return make_sphere(center, json_float(json_aref(obj, "radius")), mat_type, mat_id);
}

static int parse_moving_sphere(JObj obj) {
  vec3 center1, center2;
  json_vec3(center1, json_aref(obj, "center1"));
  json_vec3(center2, json_aref(obj, "center2"));
  int mat_type, mat_id;
  i2t(HGETR(materials_h, obj, material), &mat_type, &mat_id);
  return make_moving_sphere(center1, center2, json_int(json_aref(obj, "radius")), mat_type, mat_id);
}

static int parse_quad(JObj obj) {
  vec3 q, u, v;
  json_vec3(q, json_aref(obj, "q"));
  json_vec3(u, json_aref(obj, "u"));
  json_vec3(v, json_aref(obj, "v"));
  int mat_type, mat_id;
  i2t(HGETR(materials_h, obj, material), &mat_type, &mat_id);
  return make_quad(q, u, v, mat_type, mat_id);
}

static int parse_tri(JObj obj) {
  vec3 a, b, c;
  json_vec3(a, json_aref(obj, "a"));
  json_vec3(b, json_aref(obj, "b"));
  json_vec3(c, json_aref(obj, "c"));
  int mat_type, mat_id;
  i2t(HGETR(materials_h, obj, material), &mat_type, &mat_id);
  return make_tri(a, b, c, mat_type, mat_id);
}

static void parse_box(JObj obj) {
  vec3 a, b;
  json_vec3(a, json_aref(obj, "a"));
  json_vec3(b, json_aref(obj, "b"));
  int mat_type, mat_id;
  i2t(HGETR(materials_h, obj, material), &mat_type, &mat_id);
  make_box(a, b, mat_type, mat_id);
}

static void parse_solid_bg(JObj bg) {
  vec3 col;
  json_vec3(col, json_aref(bg, "albedo"));
  make_solid_bg(col);
}

static void parse_grad_bg(JObj bg) {
  vec3 top, bottom;
  json_vec3(top, json_aref(bg, "top"));
  json_vec3(bottom, json_aref(bg, "bottom"));
  make_grad_bg(top, bottom);
}

static void parse_sphere_map(JObj bg) {
  int type, id;
  i2t(HGETR(textures_h, bg, texture), &type, &id);
  make_sphere_map(type, id);
}

static void parse_cube_map(JObj bg) {
  JAry ary = json_value_as_array(json_aref(bg, "textures"));
  if (ary->length != 6) {
    fprintf(stderr, "wrong number of textures for cubemap (given %lu, expected 6)\n", ary->length);
  }
  JSON_ARY_EACH(ary, it, i, {
    i2ti(HGET(textures_h, json_value_as_string(it)), &cube_map.tex[i]);
  });
  bg_type = CUBEMAP;
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

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "must pass exactly one argument, path to scene.json\n");
    return 1;
  }


  size_t scene_filesize;
  char *scene_file = load_file(argv[1], &scene_filesize);
  JAny scene_json = json_parse(scene_file, scene_filesize);
  free(scene_file);
  if (!scene_json) {
    fprintf(stderr, "could not parse scene\n");
    return 1;
  }
  JObj scene = json_value_as_object(scene_json);
  if (!scene) {
    fprintf(stderr, "could not parse scene as object\n");
    return 1;
  }

  build_decoding_table();

  srand(time(NULL));
  glfwInit();

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "raysetta-gl", NULL, NULL);
  glfwMakeContextCurrent(window);

  int version = gladLoadGL(glfwGetProcAddress);
  printf("GL %d.%d\n", GLAD_VERSION_MAJOR(version), GLAD_VERSION_MINOR(version));

  glEnable(GL_FRAMEBUFFER_SRGB);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

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
  GLuint frame_loc = glGetUniformLocation(program, "frame");

  const int samples = 1;
  const int depth = 5;

  GLuint samples_loc = glGetUniformLocation(program, "samples");
  glUniform1i(samples_loc, samples);
  GLuint depth_loc = glGetUniformLocation(program, "depth");
  glUniform1i(depth_loc, depth);

  GLuint res_loc = glGetUniformLocation(program, "res");
  glUniform2f(res_loc, WIDTH, HEIGHT);

  vec3_set(cam.eye, 0.0, 0.0, 0.0);
  vec3_set(cam.tgt, 0.0, 0.0, -1.0);
  vec3_set(cam.vup, 0.0, 1.0, 0.0);
  cam.vfov = 90.0;
  cam.defocus_angle = 0.0;
  cam.focus_dist = 10.0;

  JObj cam_json = json_value_as_object(json_aref(scene, "camera"));
  if (!cam_json) { fprintf(stderr, "failed to parse scene, missing camera object\n"); return 1; }
  JAny vfov_val = json_aref(cam_json, "vfov");
  if (vfov_val) cam.vfov = json_float(vfov_val);
  JAny eye_val = json_aref(cam_json, "lookfrom");
  if (eye_val) json_vec3(cam.eye, eye_val);
  JAny tgt_val = json_aref(cam_json, "lookat");
  if (tgt_val) json_vec3(cam.tgt, tgt_val);
  JAny vup_val = json_aref(cam_json, "vup");
  if (vup_val) json_vec3(cam.vup, vup_val);
  JAny defocus_angle_val = json_aref(cam_json, "defocus_angle");
  if (defocus_angle_val) cam.defocus_angle = json_float(defocus_angle_val);
  JAny focus_dist_val = json_aref(cam_json, "focus_dist");
  if (focus_dist_val) cam.focus_dist = json_float(focus_dist_val);

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

  JObj noises_json = json_value_as_object(json_aref(scene, "noises"));
  if (!noises_json) { fprintf(stderr, "noises missing or is not an object!\n"); return 1; }
  JObj images_json = json_value_as_object(json_aref(scene, "images"));
  if (!images_json) { fprintf(stderr, "images missing or is not an object!\n"); return 1; }
  JObj textures_json = json_value_as_object(json_aref(scene, "textures"));
  if (!textures_json) { fprintf(stderr, "textures missing or is not an object!\n"); return 1; }
  JObj materials_json = json_value_as_object(json_aref(scene, "materials"));
  if (!materials_json) { fprintf(stderr, "materials missing or is not an object!\n"); return 1; }

  HNEW(noises_h, noises_json->length);
  HNEW(images_h, images_json->length);
  HNEW(textures_h, textures_json->length);
  HNEW(materials_h, materials_json->length);

  JSON_OBJ_EACH(noises_json, k, v, {
    HPUT(noises_h, k, parse_noise(json_value_as_object(v)));
  });
  JSON_OBJ_EACH(images_json, k, v, {
    HPUT(images_h, k, parse_image(json_value_as_object(v)));
  });
  JSON_OBJ_EACH(textures_json, k, v, {
    HPUT(textures_h, k, parse_texture(json_value_as_object(v)));
  });
  JSON_OBJ_EACH(materials_json, k, v, {
    HPUT(materials_h, k, parse_material(json_value_as_object(v)));
  });

  JObj world_json = json_value_as_object(json_aref(scene, "world"));
  if (!world_json) { fprintf(stderr, "world missing or is not an object!\n"); return 1; }
  JSON_OBJ_EACH(world_json, _, val, {
    JObj obj = json_value_as_object(val);
    JStr type_str = json_value_as_string(json_aref(obj, "type"));
    if (0 == strcmp(type_str->string, "Sphere")) {
      parse_sphere(obj);
    } else if (0 == strcmp(type_str->string, "MovingSphere")) {
      parse_moving_sphere(obj);
    } else if (0 == strcmp(type_str->string, "Quad")) {
      parse_quad(obj);
    } else if (0 == strcmp(type_str->string, "Tri")) {
      parse_tri(obj);
    } else if (0 == strcmp(type_str->string, "Box")) {
      parse_box(obj);
    } else {
      fprintf(stderr, "unknown object type %s\n", type_str->string);
    }
  });

  JObj bg_json = json_value_as_object(json_aref(scene, "background"));
  if (!bg_json) { fprintf(stderr, "background missing or is not an object!\n"); return 1; }
  JStr bg_type_str = json_value_as_string(json_aref(bg_json, "type"));
  if (0 == strcmp(bg_type_str->string, "Solid")) {
    parse_solid_bg(bg_json);
  } else if (0 == strcmp(bg_type_str->string, "Gradient")) {
    parse_grad_bg(bg_json);
  } else if (0 == strcmp(bg_type_str->string, "SphereMap")) {
    parse_sphere_map(bg_json);
  } else if (0 == strcmp(bg_type_str->string, "CubeMap")) {
    parse_cube_map(bg_json);
  } else {
    fprintf(stderr, "unknown object type %s\n", bg_type_str->string);
  }

  hashmap_destroy(&noises_h);
  hashmap_destroy(&images_h);
  hashmap_destroy(&textures_h);
  hashmap_destroy(&materials_h);

  free(scene_json);

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
  glBindBuffer(GL_UNIFORM_BUFFER, 0);
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

  GLuint bg_buf = make_unibuffer(program, sizeof(SolidColor)+sizeof(Gradient)+sizeof(TypeId)+sizeof(CubeMap), "BgBlock");
  glBindBuffer(GL_UNIFORM_BUFFER, bg_buf);
  switch (bg_type) {
  case SOLID: glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(SolidColor), &solid_bg); break;
  case GRAD: glBufferSubData(GL_UNIFORM_BUFFER, sizeof(SolidColor), sizeof(Gradient), &grad_bg); break;
  case SPHMAP: glBufferSubData(GL_UNIFORM_BUFFER, sizeof(SolidColor)+sizeof(Gradient), sizeof(TypeId), &sphere_map); break;
  case CUBEMAP: glBufferSubData(GL_UNIFORM_BUFFER, sizeof(SolidColor)+sizeof(Gradient)+sizeof(TypeId), sizeof(CubeMap), &cube_map); break;
  }
  glBindBuffer(GL_UNIFORM_BUFFER, 0);
  GLuint bg_type_loc = glGetUniformLocation(program, "bg_type");
  glUniform1i(bg_type_loc, bg_type);
  glErrorCheck("bind bg");

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
    free(images[i].data);
    glErrorCheck("tex sub data 3d");
  }
  glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
  glErrorCheck("setup images");

  GLuint images_loc = glGetUniformLocation(program, "images");
  glUniform1i(images_loc, 0);
  glActiveTexture(GL_TEXTURE0);
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

    nframes++;
    if (t - last_time > 1.0) {
      printf("FPS %i\n", nframes);
      nframes = 0;
      last_time += 1.0;
    }

    glUniform3f(eye_loc, cam.eye[0], cam.eye[1], cam.eye[2]);

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

  return 0;
}
