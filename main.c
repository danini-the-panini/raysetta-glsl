#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>

#define GLAD_GL_IMPLEMENTATION
#include "gl.h"

#include <GLFW/glfw3.h>

#include "linmath.h"
#include "camera.h"

#define WIDTH 720
#define HEIGHT 400

#define LAMBERT 0
#define METAL 1
#define GLASS 2

#define SPHERE 0
#define BVH 1

#define MAX_SPH 500
#define MAX_OBJS MAX_SPH
#define MAX_MATS MAX_OBJS

typedef vec4 aabb[2];

typedef struct bvh_node {
  int left_id;
  int left_type;
  int right_id;
  int right_type;
  aabb bbox;
} BVHNode;

typedef struct material {
  int id;
  int type;
  float __1[2];
} Material;

typedef struct sphere {
  vec3 center;
  float __0;
  vec3 vec;
  float radius;
  Material mat;
} Sphere;

typedef struct object {
  int id;
  int type;
} Object;

typedef struct lambert {
  vec3 albedo;
  float __0;
} Lambert;

typedef struct metal {
  vec3 albedo;
  float fuzz;
} Metal;

typedef struct glass {
  float index;
  float __0[3];
} Glass;

static inline void vec3_set(vec3 v, float x, float y, float z) {
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

static bool clear = true;
static Camera cam;
static double last_mouse[2];
static float orbit_max = 100.0;
static float orbit_min = 0.1;
static GLuint frame = 0;

static int bvh_count = 0;
static int sphere_count = 0;
static int obj_count = 0;
static int lambert_count = 0;
static int metal_count = 0;
static int glass_count = 0;
static Sphere spheres[MAX_SPH];
static const int MAX_BVH = MAX_OBJS*2+1;
static BVHNode bvh_nodes[MAX_OBJS*2+1];
static Object objects[MAX_OBJS];
static Lambert lamberts[MAX_MATS];
static Metal metals[MAX_MATS];
static Glass glass[MAX_MATS];

static inline void aabb_copy(aabb r, aabb b) {
  vec3_copy(r[0], b[0]);
  vec3_copy(r[1], b[1]);
}

static inline void aabb_add(aabb r, aabb a, aabb b) {
  r[0][0] = fminf(a[0][0], b[0][0]);
  r[0][1] = fminf(a[0][1], b[0][1]);
  r[0][2] = fminf(a[0][2], b[0][2]);

  r[1][0] = fmaxf(a[1][0], b[1][0]);
  r[1][1] = fmaxf(a[1][1], b[1][1]);
  r[1][2] = fmaxf(a[1][2], b[1][2]);
}

static inline void aabb_make(aabb r, vec3 a, vec3 b) {
  r[0][0] = fminf(a[0], b[0]);
  r[0][1] = fminf(a[1], b[1]);
  r[0][2] = fminf(a[2], b[2]);

  r[1][0] = fmaxf(a[0], b[0]);
  r[1][1] = fmaxf(a[1], b[1]);
  r[1][2] = fmaxf(a[2], b[2]);
}

static inline int aabb_longest(aabb a) {
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

static inline void obj_bbox(aabb r, int type, int id) {
  switch (type) {
  case SPHERE: sphere_bbox(r, spheres+id); break;
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
    fprintf(stderr, "maximum number of objectss reached (%i)\n", MAX_OBJS);
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

static inline int make_lambert(vec3 albedo) {
  if (lambert_count >= MAX_MATS) {
    fprintf(stderr, "maximum number of lamberts reached (%i)", MAX_MATS);
    return -1;
  }
  vec3_copy(lamberts[lambert_count].albedo, albedo);
  return lambert_count++;
}

static inline int make_metal(vec3 albedo, float fuzz) {
  if (metal_count >= MAX_MATS) {
    fprintf(stderr, "maximum number of metals reached (%i)", MAX_MATS);
    return -1;
  }
  vec3_copy(metals[metal_count].albedo, albedo);
  metals[metal_count].fuzz = fuzz;
  return metal_count++;
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

  GLuint time_loc = glGetUniformLocation(program, "time");
  GLuint utime_loc = glGetUniformLocation(program, "utime");
  GLuint frame_loc = glGetUniformLocation(program, "frame");

  GLuint noise_tex = 0;
  glGenTextures(1, &noise_tex);
  glBindTexture(GL_TEXTURE_1D, noise_tex);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexStorage1D(GL_TEXTURE_1D, 1, GL_R32F, max_tex);
  glTexSubImage1D(GL_TEXTURE_1D, 0, 0, max_tex, GL_RED, GL_FLOAT, noise);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_1D, 0);

  GLuint noise_loc = glGetUniformLocation(program, "noise");
  GLuint noise_size_loc = glGetUniformLocation(program, "noise_size");
  glUniform1i(noise_loc, 0);
  glUniform1i(noise_size_loc, max_tex);

  const int samples = 1;
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
  cam.defocus_angle = 0.6;
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

  int gmat = make_glass(1.5);

  // ground
  int ground_mat = make_lambert((vec3){0.5, 0.5, 0.5});
  make_sphere((vec3){0.0, -1000.0, 0.0}, 1000.0, LAMBERT, ground_mat);

  make_sphere((vec3){0.0, 1.0, 0.0}, 1.0, GLASS, gmat);
  make_sphere((vec3){-4.0, 1.0, 0.0}, 1.0, LAMBERT, make_lambert((vec3){0.4, 0.2, 0.1}));
  make_sphere((vec3){4.0, 1.0, 0.0}, 1.0, METAL, make_metal((vec3){0.7, 0.6, 0.5}, 0.0));

  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      float choose_mat = randf();
      vec3 center;
      vec3_set(center, a + 0.9*randf(), 0.2, b + 0.9*randf());

      vec3 tmp;
      vec3_sub(tmp, center, (vec3){4.0, 0.2, 0.0});
      if (vec3_len(tmp) > 0.9) {
        if (choose_mat < 0.8) {
          // diffuse
          vec3 r, c, center2;
          vec3_rand(r);
          vec3_rand(c);
          vec3_mul(c, c, r);
          vec3_add(center2, center, (vec3){0.0, randf()*0.5, 0.0});
          make_moving_sphere(center, center2, 0.2, LAMBERT, make_lambert(c));
        } else if (choose_mat < 0.95) {
          // metal
          vec3 r, c;
          vec3_rand(r);
          vec3_rand(c);
          vec3_mul(c, c, r);
          make_sphere(center, 0.2, METAL, make_metal(c, randf()*0.5));
        } else {
          // glass
          make_sphere(center, 0.2, GLASS, gmat);
        }
      }
    }
  }

  make_bvh_nodes(0, sphere_count);

  GLuint sphere_count_loc = glGetUniformLocation(program, "sphere_count");
  glUniform1i(sphere_count_loc, sphere_count);

  GLuint bvh_count_loc = glGetUniformLocation(program, "bvh_count");
  glUniform1i(bvh_count_loc, bvh_count);

  GLuint sph_loc = glGetUniformBlockIndex(program, "ObjBlock");
  glUniformBlockBinding(program, sph_loc, 0);
  GLuint sph_buf;
  glGenBuffers(1, &sph_buf);
  glBindBuffer(GL_UNIFORM_BUFFER, sph_buf);
  glBufferData(GL_UNIFORM_BUFFER, sizeof(Sphere)*MAX_SPH, NULL, GL_STATIC_DRAW);
  glBindBufferBase(GL_UNIFORM_BUFFER, 0, sph_buf);
  glBindBufferRange(GL_UNIFORM_BUFFER, 0, sph_buf, 0, sizeof(Sphere)*MAX_SPH);
  glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(Sphere)*sphere_count, spheres);
  glBindBuffer(GL_UNIFORM_BUFFER, 0);

  GLuint mat_loc = glGetUniformBlockIndex(program, "MatBlock");
  glUniformBlockBinding(program, mat_loc, 1);
  GLuint mat_buf;
  glGenBuffers(1, &mat_buf);
  glBindBuffer(GL_UNIFORM_BUFFER, mat_buf);
  glBufferData(GL_UNIFORM_BUFFER,
               sizeof(Lambert)*MAX_MATS+sizeof(Metal)*MAX_MATS+sizeof(Glass)*MAX_MATS,
               NULL, GL_STATIC_DRAW);
  glBindBufferBase(GL_UNIFORM_BUFFER, 1, mat_buf);
  glBindBufferRange(GL_UNIFORM_BUFFER, 1, mat_buf, 0,
                    sizeof(Lambert)*MAX_MATS + sizeof(Metal)*MAX_MATS + sizeof(Glass)*MAX_MATS);
  glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(Lambert)*lambert_count, lamberts);
  glBufferSubData(GL_UNIFORM_BUFFER,
                  sizeof(Lambert)*MAX_MATS,
                  sizeof(Metal)*metal_count,
                  metals);
  glBufferSubData(GL_UNIFORM_BUFFER,
                  sizeof(Lambert)*MAX_MATS + sizeof(Metal)*MAX_MATS,
                  sizeof(Glass)*glass_count,
                  glass);
  glBindBuffer(GL_UNIFORM_BUFFER, 0);

  GLuint bvh_loc = glGetUniformBlockIndex(program, "BvhBlock");
  glUniformBlockBinding(program, bvh_loc, 2);
  GLuint bvh_buf;
  glGenBuffers(1, &bvh_buf);
  glBindBuffer(GL_UNIFORM_BUFFER, bvh_buf);
  glBufferData(GL_UNIFORM_BUFFER, sizeof(BVHNode)*MAX_BVH, NULL, GL_STATIC_DRAW);
  glBindBufferBase(GL_UNIFORM_BUFFER, 2, bvh_buf);
  glBindBufferRange(GL_UNIFORM_BUFFER, 2, bvh_buf, 0, sizeof(BVHNode)*MAX_BVH);
  glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(BVHNode)*bvh_count, bvh_nodes);
  glBindBuffer(GL_UNIFORM_BUFFER, 0);

  glfwGetCursorPos(window, &last_mouse[0], &last_mouse[1]);

  glfwSetFramebufferSizeCallback(window, on_resize);
  glfwSetCursorPosCallback(window, on_mouse_move);
  glfwSetScrollCallback(window, on_scroll);

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
