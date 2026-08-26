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

#define WIDTH 400
#define HEIGHT 200

#define LAMBERT 0
#define METAL 1
#define GLASS 2

typedef struct material {
  int id;
  int type;
  float __0;
  float __1;
} Material;

typedef struct sphere {
  vec3 center;
  float radius;
  Material mat;
} Sphere;

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
  float __0;
  float __1;
  float __2;
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

float randf() {
  return (float)rand() / (float)RAND_MAX;
}

float randf_r(float min, float max) {
  return min + (max-min)*randf();
}

static Camera cam;
static double last_mouse[2];
static float orbit_max = 100.0;
static float orbit_min = 0.1;

void on_mouse_move(GLFWwindow *win, double x, double y) {
  float dx = x - last_mouse[0];
  float dy = y - last_mouse[1];

  if (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS) {
    float speed = 0.01;
    rotate_around_with_fixed_up(&cam, cam.tgt, speed * dx, speed * dy);
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
  printf("MAX TEX: %i\n", max_tex);

  glEnable(GL_FRAMEBUFFER_SRGB);

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

  GLuint noise_tex = 0;
  glGenTextures(1, &noise_tex);
  glBindTexture(GL_TEXTURE_1D, noise_tex);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexStorage1D(GL_TEXTURE_1D, 1, GL_R32F, max_tex);
  glTexSubImage1D(GL_TEXTURE_1D, 0, 0, max_tex, GL_RED, GL_FLOAT, noise);
  glActiveTexture(GL_TEXTURE0);

  GLuint noise_loc = glGetUniformLocation(program, "noise");
  GLuint noise_size_loc = glGetUniformLocation(program, "noise_size");
  glUniform1i(noise_loc, 0);
  glUniform1i(noise_size_loc, max_tex);

  const int samples = 10;
  const int depth = 10;

  GLuint samples_loc = glGetUniformLocation(program, "samples");
  glUniform1i(samples_loc, samples);
  GLuint depth_loc = glGetUniformLocation(program, "depth");
  glUniform1i(depth_loc, depth);

  GLuint res_loc = glGetUniformLocation(program, "res");
  glUniform2f(res_loc, WIDTH, HEIGHT);

  vec3_set(cam.eye, -2.0, 1.0, 1.0);
  vec3_set(cam.tgt, 0.0, 0.0, -1.0);
  vec3_set(cam.vup, 0.0, 1.0, 0.0);
  cam.vfov = 45.0;

  GLuint eye_loc = glGetUniformLocation(program, "eye");
  glUniform3f(eye_loc, cam.eye[0], cam.eye[1], cam.eye[2]);
  GLuint tgt_loc = glGetUniformLocation(program, "tgt");
  glUniform3f(tgt_loc, cam.tgt[0], cam.tgt[1], cam.tgt[2]);
  GLuint vup_loc = glGetUniformLocation(program, "vup");
  glUniform3f(vup_loc, cam.vup[0], cam.vup[1], cam.vup[2]);
  GLuint vfov_loc = glGetUniformLocation(program, "vfov");
  glUniform1f(vfov_loc, cam.vfov);

  const int sphere_count = 5;
  const int lambert_count = 2;
  const int metal_count = 1;
  const int glass_count = 2;
  Sphere spheres[100];
  Lambert lamberts[100];
  Metal metals[100];
  Glass glass[100];

  vec3_set(lamberts[0].albedo, 0.8, 0.8, 0.0);
  vec3_set(lamberts[1].albedo, 0.1, 0.2, 0.5);
  vec3_set(metals[0].albedo, 0.8, 0.6, 0.2);
  metals[0].fuzz = 0.2;
  glass[0].index = 1.5;
  glass[1].index = 1.0 / 1.5;

  vec3_set(spheres[0].center, 0.0, -100.5, -1.0);
  spheres[0].radius = 100.0;
  spheres[0].mat.type = LAMBERT;
  spheres[0].mat.id = 0;

  vec3_set(spheres[1].center, 0.0, 0.0, -1.2);
  spheres[1].radius = 0.5;
  spheres[1].mat.type = LAMBERT;
  spheres[1].mat.id = 1;

  vec3_set(spheres[2].center, -1.0, 0.0, -1.0);
  spheres[2].radius = 0.5;
  spheres[2].mat.type = GLASS;
  spheres[2].mat.id = 0;

  vec3_set(spheres[3].center, -1.0, 0.0, -1.0);
  spheres[3].radius = 0.45;
  spheres[3].mat.type = GLASS;
  spheres[3].mat.id = 1;

  vec3_set(spheres[4].center, 1.0, 0.0, -1.0);
  spheres[4].radius = 0.5;
  spheres[4].mat.type = METAL;
  spheres[4].mat.id = 0;

  GLuint sphere_count_loc = glGetUniformLocation(program, "sphere_count");
  glUniform1i(sphere_count_loc, sphere_count);

  GLuint sph_loc = glGetUniformBlockIndex(program, "ObjBlock");
  glUniformBlockBinding(program, sph_loc, 0);
  GLuint sph_buf;
  glGenBuffers(1, &sph_buf);
  glBindBuffer(GL_UNIFORM_BUFFER, sph_buf);
  glBufferData(GL_UNIFORM_BUFFER, sizeof(Sphere)*100, NULL, GL_STATIC_DRAW);
  glBindBufferBase(GL_UNIFORM_BUFFER, 0, sph_buf);
  glBindBufferRange(GL_UNIFORM_BUFFER, 2, sph_buf, 0, sizeof(Sphere)*100);
  glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(Sphere)*sphere_count, spheres);
  glBindBuffer(GL_UNIFORM_BUFFER, 0);

  GLuint mat_loc = glGetUniformBlockIndex(program, "MatBlock");
  glUniformBlockBinding(program, mat_loc, 1);
  GLuint mat_buf;
  glGenBuffers(1, &mat_buf);
  glBindBuffer(GL_UNIFORM_BUFFER, mat_buf);
  glBufferData(GL_UNIFORM_BUFFER,
               sizeof(Lambert)*100+sizeof(Metal)*100+sizeof(Glass)*100,
               NULL, GL_STATIC_DRAW);
  glBindBufferBase(GL_UNIFORM_BUFFER, 1, mat_buf);
  glBindBufferRange(GL_UNIFORM_BUFFER, 2, mat_buf, 0,
                    sizeof(Lambert)*100 + sizeof(Metal)*100 + sizeof(Glass)*100);
  glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(Lambert)*lambert_count, lamberts);
  glBufferSubData(GL_UNIFORM_BUFFER,
                  sizeof(Lambert)*100,
                  sizeof(Metal)*metal_count,
                  metals);
  glBufferSubData(GL_UNIFORM_BUFFER,
                  sizeof(Lambert)*100 + sizeof(Metal)*100,
                  sizeof(Glass)*glass_count,
                  glass);
  glBindBuffer(GL_UNIFORM_BUFFER, 0);

  glfwGetCursorPos(window, &last_mouse[0], &last_mouse[1]);

  glfwSetCursorPosCallback(window, on_mouse_move);
  glfwSetScrollCallback(window, on_scroll);

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    int w, h;
    glfwGetWindowSize(window,&w, &h);
    glUniform2f(res_loc, w, h);
    float t = glfwGetTime();
    glUniform1f(time_loc, t);
    glUniform1ui(utime_loc, (GLuint)t);

    glUniform3f(eye_loc, cam.eye[0], cam.eye[1], cam.eye[2]);

    for (GLsizei i = 0; i < max_tex; i++) {
      noise[i] = randf();
    }
    glTexSubImage1D(GL_TEXTURE_1D, 0, 0, max_tex, GL_RED, GL_FLOAT, noise);

    glClearColor(0.7f, 0.9f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glfwSwapBuffers(window);
  }

  glfwTerminate();

  free(noise);
  return 0;
}
