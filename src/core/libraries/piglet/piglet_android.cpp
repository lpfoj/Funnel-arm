// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/libraries/libs.h"
#include "core/libraries/piglet/piglet_android.h"

#ifdef __ANDROID__
#include "core/aerolib/stubs.h"

#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <android/log.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdarg>
#include <cstdlib>
#include <atomic>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/stb.h"

namespace Libraries::Pad {
bool ExecutorConsumeUiLaunchRequest();
std::uint32_t ExecutorConsumeUiNavButtons();
}

namespace Libraries::Piglet {

#ifdef __ANDROID__

extern "C" void* executor_lsx4_runtime_get_native_window();

namespace {

EGLDisplay g_display = EGL_NO_DISPLAY;
EGLConfig g_config = nullptr;
EGLSurface g_surface = EGL_NO_SURFACE;
EGLContext g_context = EGL_NO_CONTEXT;
void* g_surface_window = nullptr;
bool g_initialized = false;
float g_clear_color[4] = {0.1211f, 0.1211f, 0.1211f, 1.0f};
std::uint64_t g_swap_count = 0;
std::uint64_t g_auto_swap_count = 0;
std::chrono::steady_clock::time_point g_last_auto_swap{};
bool g_guest_egl_controls_swap = false;
int g_guest_egl_swap_control_log_budget = 8;
int g_guest_swap_freeze_log_budget = 4;
GLint g_surface_width = 0;
GLint g_surface_height = 0;
std::unordered_map<GLuint, GLenum> g_shader_types;
int g_program_link_log_budget = 32;
int g_location_log_budget = 64;
int g_draw_error_log_budget = 32;
int g_draw_success_log_budget = 8;
int g_draw_elements_success_log_budget = 16;
int g_vertex_probe_log_budget = 24;
int g_uniform4f_log_budget = 48;
int g_draw_probe_log_budget = 16;
int g_readback_probe_log_budget = 16;
int g_cover_draw_log_budget = 6000;
int g_pixel_space_readback_log_budget = 32;
int g_clear_log_budget = 32;
int g_viewport_override_log_budget = 24;
int g_texture_upload_log_budget = 96;
int g_texture_driver_stage_log_budget = 2048;
std::uint64_t g_texture_upload_sequence = 0;
std::uint64_t g_draw_call_sequence = 0;
int g_texture_state_log_budget = 96;
int g_fbo_log_budget = 64;
int g_uniform_sampler_log_budget = 64;
int g_texture_rebind_log_budget = 24;
int g_shader_sanitize_log_budget = 24;
int g_shader_source_log_budget = 32;
int g_font_atlas_draw_log_budget = 256;
int g_buffer_shadow_log_budget = 40;
int g_vbo_draw_probe_log_budget = 40;
int g_synthetic_attrib_log_budget = 48;
int g_itemzflow_font_atlas_assist_log_budget = 24;
int g_itemzflow_ascii_atlas_reconstruct_log_budget = 24;
int g_itemzflow_text_texcoord_repair_log_budget = 24;
int g_itemzflow_provider_gate_log_budget = 32;
std::uint64_t g_itemzflow_text_texcoord_repair_sequence = 0;
GLuint g_active_texture_unit = 0;
std::array<GLuint, 32> g_bound_texture_2d{};
std::array<GLuint, 32> g_last_nonzero_texture_2d{};
std::array<GLuint, 32> g_last_alpha_texture_2d{};
struct FontAtlasWatch {
    GLuint texture = 0;
    const GLubyte* buffer = nullptr;
    GLsizei width = 0;
    GLsizei height = 0;
    std::size_t last_nonzero = 0;
    int frame_throttle = 0;
};
FontAtlasWatch g_font_atlas_watch;
GLuint g_synthetic_texcoord_vbo = 0;
GLuint g_synthetic_color_vbo = 0;

struct ClientAttribState {
    bool active = false;
    GLint size = 0;
    GLenum type = GL_FLOAT;
    GLboolean normalized = GL_FALSE;
    GLsizei stride = 0;
    const void* pointer = nullptr;
};

std::array<ClientAttribState, 16> g_client_attribs{};
std::array<GLuint, 16> g_client_attrib_vbos{};
int g_client_attrib_upload_log_budget = 32;
GLuint g_current_program = 0;
int g_program_state_reset_log_budget = 32;
std::unordered_map<std::uint64_t, GLint> g_uniform_location_map;
int g_uniform_location_log_budget = 64;
std::unordered_map<GLuint, std::vector<std::uint8_t>> g_array_buffer_shadow;
std::unordered_map<GLuint, std::vector<std::uint8_t>> g_element_buffer_shadow;
std::unordered_map<GLuint, std::string> g_shader_sources;
std::unordered_map<GLuint, std::vector<GLuint>> g_program_attached_shaders;

struct AlphaTextureShadow {
    GLsizei width = 0;
    GLsizei height = 0;
    std::vector<GLubyte> alpha;
    std::size_t nonzero_alpha = 0;
    std::size_t first_nonzero_alpha = SIZE_MAX;
};

std::unordered_map<GLuint, AlphaTextureShadow> g_alpha_texture_shadow;

struct TextureUploadInfo {
    GLsizei width = 0;
    GLsizei height = 0;
    GLint internalformat = 0;
    GLenum format = 0;
    GLenum type = 0;
    std::size_t bytes = 0;
    std::uint64_t sequence = 0;
    bool alpha_upload = false;
    bool icon_clamped = false;
};

std::unordered_map<GLuint, TextureUploadInfo> g_texture_upload_info;

struct ProgramShaderInfo {
    bool has_fragment_source = false;
    bool samples_texture = false;
    bool reads_alpha = false;
    bool reads_red = false;
    bool mentions_color = false;
};

std::unordered_map<GLuint, ProgramShaderInfo> g_program_shader_info;

struct Uniform4fState {
    bool valid = false;
    GLfloat values[4] = {};
};

std::array<Uniform4fState, 64> g_uniform4f_states{};

struct AttribBounds {
    bool valid = false;
    GLuint index = 0;
    GLfloat min_x = 0.0f;
    GLfloat min_y = 0.0f;
    GLfloat max_x = 0.0f;
    GLfloat max_y = 0.0f;
};

AttribBounds g_last_attrib0_bounds{};
std::uint64_t g_window_draw_count = 0;
std::uint64_t g_degenerate_window_draw_count = 0;
bool g_default_fbo_drawn_since_present = false;
std::uint64_t g_clear_present_count = 0;
std::atomic<int> g_clears_without_guest_swap{0};
std::uint64_t g_itemzflow_overlay_frame_count = 0;
int g_itemzflow_overlay_log_budget = 12;
int g_itemzflow_authentic_log_budget = 12;
int g_itemzflow_readable_menu_assist_log_budget = 16;
bool g_itemzflow_overlay_enabled = false;
GLuint g_itemzflow_real_cover_texture = 0;
GLint g_itemzflow_real_cover_w = 0;
GLint g_itemzflow_real_cover_h = 0;

constexpr GLenum GL_BGRA_EXT_COMPAT = 0x80E1;

int g_piglet_last_gl_fd = -2;
std::uint64_t g_piglet_last_gl_line_count = 0;
std::once_flag g_piglet_last_gl_init_once;

bool PigletLastGlTraceEnabled() {
    const char* env = std::getenv("EXECUTOR_PIGLET_PERSIST_GL_TRACE");
    return env == nullptr || env[0] == '\0' || std::strcmp(env, "0") != 0;
}

void InitPigletLastGlTrace() {
    if (!PigletLastGlTraceEnabled()) {
        g_piglet_last_gl_fd = -1;
        return;
    }
    g_piglet_last_gl_fd =
        open("/data/data/app.lsx4.android/files/executor-piglet-last-gl-events.log",
             O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (g_piglet_last_gl_fd >= 0) {
        const char header[] =
            "[EXECUTOR_PIGLET_LAST_GL] trace-start version=icon-tile-2026-05-30\n";
        write(g_piglet_last_gl_fd, header, sizeof(header) - 1);
    }
}

void PigletLastGlTrace(const char* format, ...) {
    std::call_once(g_piglet_last_gl_init_once, InitPigletLastGlTrace);
    if (g_piglet_last_gl_fd < 0 || g_piglet_last_gl_line_count >= 20000) {
        return;
    }
    char line[1024];
    va_list args;
    va_start(args, format);
    const int written = std::vsnprintf(line, sizeof(line), format, args);
    va_end(args);
    if (written <= 0) {
        return;
    }
    const std::size_t length =
        std::min<std::size_t>(static_cast<std::size_t>(written), sizeof(line) - 2);
    line[length] = '\n';
    write(g_piglet_last_gl_fd, line, length + 1);
    ++g_piglet_last_gl_line_count;
}
int g_itemzflow_launch_banner_frames = 0;
int g_itemzflow_selected_index = 0;

bool EnvEnabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

bool ItemzflowAuthenticUiRequired() {
    return EnvEnabled("EXECUTOR_ITEMZFLOW_REQUIRE_AUTHENTIC_UI");
}

bool ItemzflowProviderGlyphStrict() {
    return EnvEnabled("EXECUTOR_ITEMZFLOW_REQUIRE_PROVIDER_GLYPH_CP");
}

bool ItemzflowFontAtlasAssistEnabled() {
    return g_itemzflow_overlay_enabled && ItemzflowAuthenticUiRequired() &&
           !ItemzflowProviderGlyphStrict() &&
           !EnvEnabled("EXECUTOR_ITEMZFLOW_DISABLE_FONT_ATLAS_ASSIST");
}

bool ItemzflowReadableMenuAssistEnabled() {
    return g_itemzflow_overlay_enabled && ItemzflowAuthenticUiRequired() &&
           EnvEnabled("EXECUTOR_ITEMZFLOW_FORCE_READABLE_MENU_ASSIST") &&
           !EnvEnabled("EXECUTOR_ITEMZFLOW_DISABLE_READABLE_MENU_ASSIST");
}

bool ItemzflowAsciiAtlasReconstructEnabled() {
    return g_itemzflow_overlay_enabled && ItemzflowAuthenticUiRequired() &&
           !ItemzflowProviderGlyphStrict() &&
           !EnvEnabled("EXECUTOR_ITEMZFLOW_DISABLE_ASCII_ATLAS_RECONSTRUCT");
}

GLint SyntheticUniformLocation(const char* name) {
    if (name == nullptr) {
        return -1;
    }
    if (std::strcmp(name, "u_color") == 0) return 1001;
    if (std::strcmp(name, "u_time") == 0) return 1002;
    if (std::strcmp(name, "u_TextureUnit") == 0) return 1003;
    if (std::strcmp(name, "texture") == 0) return 1004;
    if (std::strcmp(name, "tex") == 0) return 1005;
    if (std::strcmp(name, "Texture") == 0) return 1006;
    if (std::strcmp(name, "model") == 0) return 1010;
    if (std::strcmp(name, "view") == 0) return 1011;
    if (std::strcmp(name, "projection") == 0) return 1012;
    if (std::strcmp(name, "Color") == 0) return 1013;
    if (std::strcmp(name, "time") == 0) return 1014;
    if (std::strcmp(name, "mouse") == 0) return 1015;
    if (std::strcmp(name, "resolution") == 0) return 1016;
    return -1;
}

std::uint64_t UniformMapKey(GLuint program, GLint synthetic_location) {
    return (static_cast<std::uint64_t>(program) << 32) |
           static_cast<std::uint32_t>(synthetic_location);
}

GLint ResolveUniformLocationForCurrentProgram(GLint location) {
    if (location < 0) {
        return location;
    }
    GLint program = 0;
    glGetIntegerv(GL_CURRENT_PROGRAM, &program);
    if (program <= 0) {
        return location;
    }
    const auto it = g_uniform_location_map.find(
        UniformMapKey(static_cast<GLuint>(program), location));
    return it != g_uniform_location_map.end() ? it->second : location;
}

void Log(const char* format, ...) {
    va_list args;
    va_start(args, format);
    __android_log_vprint(ANDROID_LOG_INFO, "LSX4Native", format, args);
    va_end(args);
}

void DrainGlErrors(const char* label) {
    if (eglGetCurrentContext() == EGL_NO_CONTEXT) {
        return;
    }

    static int drain_log_budget = 16;
    GLenum error = GL_NO_ERROR;
    for (int drained = 0; drained < 8; ++drained) {
        error = glGetError();
        if (error == GL_NO_ERROR) {
            return;
        }
        if (drained < 2 && drain_log_budget-- > 0) {
            Log("[EXECUTOR_PIGLET] drained stale GL error before %s: 0x%x", label, error);
        }
    }
}

void LogGlErrorAfter(const char* label) {
    if (eglGetCurrentContext() == EGL_NO_CONTEXT) {
        return;
    }
    const GLenum error = glGetError();
    if (error != GL_NO_ERROR && g_draw_error_log_budget-- > 0) {
        Log("[EXECUTOR_PIGLET] %s produced GL error=0x%x", label, error);
    }
}

bool LooksLikeClientPointer(const void* pointer) {
    const auto value = reinterpret_cast<std::uintptr_t>(pointer);
    return value >= 0x10000;
}

std::size_t GlTypeSize(GLenum type) {
    switch (type) {
    case GL_BYTE:
    case GL_UNSIGNED_BYTE:
        return 1;
    case GL_SHORT:
    case GL_UNSIGNED_SHORT:
        return 2;
    case GL_FLOAT:
    case GL_FIXED:
        return 4;
    default:
        return 0;
    }
}

bool IsFiniteFloat(GLfloat value) {
    return std::isfinite(value) && value > -100000.0f && value < 100000.0f;
}

bool LooksLikeUnitFloat(GLfloat value) {
    return IsFiniteFloat(value) && value >= -0.01f && value <= 1.01f;
}

bool LooksLikeColorComponent(GLfloat value) {
    return IsFiniteFloat(value) && value >= -0.01f && value <= 255.01f;
}

GLfloat NormalizePackedColorComponent(GLfloat value) {
    if (!std::isfinite(value)) {
        return 1.0f;
    }
    if (value > 1.0f && value <= 255.0f) {
        return std::clamp(value / 255.0f, 0.0f, 1.0f);
    }
    return std::clamp(value, 0.0f, 1.0f);
}

bool LooksLikeItemzPackedVertex(const ClientAttribState& attrib, GLsizei vertex_count) {
    if (!attrib.active || attrib.pointer == nullptr || attrib.type != GL_FLOAT ||
        attrib.size != 3 || vertex_count < 2) {
        return false;
    }
    if (attrib.stride == 0) {
        return false;
    }
    const std::size_t guest_stride =
        attrib.stride > 0 ? static_cast<std::size_t>(attrib.stride) : 3U * sizeof(GLfloat);
    if (guest_stride != 3U * sizeof(GLfloat)) {
        return false;
    }

    const auto* values = static_cast<const GLfloat*>(attrib.pointer);
    const GLfloat x0 = values[0];
    const GLfloat y0 = values[1];
    const GLfloat z0 = values[2];
    const GLfloat s0 = values[3];
    const GLfloat t0 = values[4];
    const GLfloat r0 = values[5];
    const GLfloat g0 = values[6];
    const GLfloat b0 = values[7];
    const GLfloat a0 = values[8];
    const GLfloat x1 = values[9];
    const GLfloat y1 = values[10];
    const bool first_position_ok = IsFiniteFloat(x0) && IsFiniteFloat(y0) && IsFiniteFloat(z0);
    const bool uv_ok = LooksLikeUnitFloat(s0) && LooksLikeUnitFloat(t0);
    const bool color_ok = LooksLikeColorComponent(r0) && LooksLikeColorComponent(g0) &&
                          LooksLikeColorComponent(b0) && LooksLikeColorComponent(a0);
    const bool next_position_ok = IsFiniteFloat(x1) && IsFiniteFloat(y1) &&
                                  (std::abs(x1) > 2.0f || std::abs(y1) > 2.0f ||
                                   std::abs(x0) > 2.0f || std::abs(y0) > 2.0f);
    return first_position_ok && uv_ok && color_ok && next_position_ok;
}

std::size_t EffectiveClientAttribStride(const ClientAttribState& attrib, GLsizei vertex_count,
                                        std::size_t element_size) {
    if (LooksLikeItemzPackedVertex(attrib, vertex_count)) {
        return 9U * sizeof(GLfloat);
    }
    return attrib.stride > 0 ? static_cast<std::size_t>(attrib.stride) : element_size;
}

void RememberUniform4f(GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3) {
    if (location < 0 || location >= static_cast<GLint>(g_uniform4f_states.size())) {
        return;
    }
    auto& state = g_uniform4f_states[static_cast<std::size_t>(location)];
    state.valid = true;
    state.values[0] = v0;
    state.values[1] = v1;
    state.values[2] = v2;
    state.values[3] = v3;
}

struct VertexAttribRuntimeState {
    GLint enabled = 0;
    GLint buffer = 0;
    GLint size = 0;
    GLint type = 0;
    GLint stride = 0;
    GLint normalized = 0;
    void* pointer = nullptr;
};

VertexAttribRuntimeState QueryVertexAttribState(GLuint index) {
    VertexAttribRuntimeState state{};
    glGetVertexAttribiv(index, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &state.enabled);
    glGetVertexAttribiv(index, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, &state.buffer);
    glGetVertexAttribiv(index, GL_VERTEX_ATTRIB_ARRAY_SIZE, &state.size);
    glGetVertexAttribiv(index, GL_VERTEX_ATTRIB_ARRAY_TYPE, &state.type);
    glGetVertexAttribiv(index, GL_VERTEX_ATTRIB_ARRAY_STRIDE, &state.stride);
    glGetVertexAttribiv(index, GL_VERTEX_ATTRIB_ARRAY_NORMALIZED, &state.normalized);
    glGetVertexAttribPointerv(index, GL_VERTEX_ATTRIB_ARRAY_POINTER, &state.pointer);
    return state;
}

void LogDrawProbe(const char* label, GLenum mode, GLint first, GLsizei count) {
    if (g_draw_probe_log_budget-- <= 0) {
        return;
    }

    GLint program = 0;
    GLint framebuffer = 0;
    GLint viewport[4] = {};
    const GLboolean blend = glIsEnabled(GL_BLEND);
    const GLboolean depth_test = glIsEnabled(GL_DEPTH_TEST);
    const GLboolean cull_face = glIsEnabled(GL_CULL_FACE);
    const GLboolean scissor_test = glIsEnabled(GL_SCISSOR_TEST);
    GLboolean color_mask[4] = {};
    glGetIntegerv(GL_CURRENT_PROGRAM, &program);
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &framebuffer);
    glGetIntegerv(GL_VIEWPORT, viewport);
    const auto attrib0 = QueryVertexAttribState(0);
    const auto attrib1 = QueryVertexAttribState(1);
    const auto attrib2 = QueryVertexAttribState(2);
    const auto attrib3 = QueryVertexAttribState(3);
    const auto attrib4 = QueryVertexAttribState(4);
    glGetBooleanv(GL_COLOR_WRITEMASK, color_mask);
    const GLenum framebuffer_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    const auto& color0 = g_uniform4f_states[0];
    Log("[EXECUTOR_PIGLET] %s mode=0x%x first=%d count=%d program=%d framebuffer=%d "
        "viewport=%d,%d %dx%d attrib0=enabled:%d buffer:%d size:%d type:0x%x stride:%d "
        "pointer:%p attrib1=enabled:%d buffer:%d size:%d type:0x%x stride:%d "
        "attrib2=enabled:%d buffer:%d size:%d type:0x%x stride:%d "
        "attrib3=enabled:%d buffer:%d size:%d type:0x%x stride:%d "
        "attrib4=enabled:%d buffer:%d size:%d type:0x%x stride:%d "
        "uniform0=%s %.3f %.3f %.3f %.3f blend:%d depth:%d cull:%d scissor:%d "
        "colorMask:%d%d%d%d fboStatus:0x%x",
        label, mode, first, count, program, framebuffer, viewport[0], viewport[1], viewport[2],
        viewport[3], attrib0.enabled, attrib0.buffer, attrib0.size, attrib0.type,
        attrib0.stride, attrib0.pointer, attrib1.enabled, attrib1.buffer, attrib1.size,
        attrib1.type, attrib1.stride, attrib2.enabled, attrib2.buffer, attrib2.size,
        attrib2.type, attrib2.stride, attrib3.enabled, attrib3.buffer, attrib3.size,
        attrib3.type, attrib3.stride, attrib4.enabled, attrib4.buffer, attrib4.size,
        attrib4.type, attrib4.stride, color0.valid ? "valid" : "unset", color0.values[0],
        color0.values[1], color0.values[2], color0.values[3], blend ? 1 : 0,
        depth_test ? 1 : 0, cull_face ? 1 : 0, scissor_test ? 1 : 0,
        color_mask[0] ? 1 : 0, color_mask[1] ? 1 : 0, color_mask[2] ? 1 : 0,
        color_mask[3] ? 1 : 0, framebuffer_status);
}

bool BoundsAreDegenerate(const AttribBounds& bounds) {
    if (!bounds.valid) {
        return false;
    }
    constexpr GLfloat kMinSpan = 0.001f;
    return (bounds.max_x - bounds.min_x) > -kMinSpan &&
           (bounds.max_x - bounds.min_x) < kMinSpan &&
           (bounds.max_y - bounds.min_y) > -kMinSpan &&
           (bounds.max_y - bounds.min_y) < kMinSpan;
}

void NoteWindowDrawBounds() {
    if (!g_last_attrib0_bounds.valid) {
        return;
    }
    ++g_window_draw_count;
    if (BoundsAreDegenerate(g_last_attrib0_bounds)) {
        ++g_degenerate_window_draw_count;
    }
}

void LogCoverDrawCandidate(const char* api, GLenum mode, GLsizei count) {
    if (g_cover_draw_log_budget <= 0) {
        return;
    }
    if (count <= 2 || count > 64) {
        return;
    }
    const GLuint bound =
        g_active_texture_unit < g_bound_texture_2d.size() ? g_bound_texture_2d[g_active_texture_unit] : 0;
    const GLuint last_nonzero =
        g_active_texture_unit < g_last_nonzero_texture_2d.size()
            ? g_last_nonzero_texture_2d[g_active_texture_unit]
            : 0;
    if (bound == 0 && last_nonzero == 0) {
        return;
    }
    --g_cover_draw_log_budget;
    if (g_last_attrib0_bounds.valid) {
        Log("[EXECUTOR_COVERDRAW] %s mode=0x%x count=%d unit=%u tex=%u lastTex=%u "
            "objBounds=(%.3f,%.3f)-(%.3f,%.3f)",
            api, mode, count, g_active_texture_unit, bound, last_nonzero,
            g_last_attrib0_bounds.min_x, g_last_attrib0_bounds.min_y, g_last_attrib0_bounds.max_x,
            g_last_attrib0_bounds.max_y);
    } else {
        Log("[EXECUTOR_COVERDRAW] %s mode=0x%x count=%d unit=%u tex=%u lastTex=%u objBounds=none",
            api, mode, count, g_active_texture_unit, bound, last_nonzero);
    }
}

void RememberPositionAttribBounds(GLuint index, const ClientAttribState& attrib,
                                  GLsizei vertex_count, std::size_t stride) {
    g_last_attrib0_bounds = {};
    if (!attrib.active || attrib.pointer == nullptr || attrib.type != GL_FLOAT ||
        attrib.size < 2 || vertex_count <= 0) {
        return;
    }

    const auto* base = static_cast<const std::uint8_t*>(attrib.pointer);
    GLfloat min_x = 0.0f;
    GLfloat min_y = 0.0f;
    GLfloat max_x = 0.0f;
    GLfloat max_y = 0.0f;
    for (GLsizei i = 0; i < vertex_count; ++i) {
        GLfloat x = 0.0f;
        GLfloat y = 0.0f;
        std::memcpy(&x, base + static_cast<std::size_t>(i) * stride, sizeof(x));
        std::memcpy(&y, base + static_cast<std::size_t>(i) * stride + sizeof(GLfloat), sizeof(y));
        if (i == 0) {
            min_x = max_x = x;
            min_y = max_y = y;
        } else {
            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
            max_x = std::max(max_x, x);
            max_y = std::max(max_y, y);
        }
    }

    g_last_attrib0_bounds = {.valid = true,
                             .index = index,
                             .min_x = min_x,
                             .min_y = min_y,
                             .max_x = max_x,
                             .max_y = max_y};
}

void FillScissorRectTopLeft(const GLint surface_width, const GLint surface_height, const GLint x,
                            const GLint y, const GLint width, const GLint height,
                            const float r, const float g, const float b, const float a) {
    if (width <= 0 || height <= 0 || surface_width <= 0 || surface_height <= 0) {
        return;
    }
    const GLint clamped_x = std::clamp(x, 0, surface_width);
    const GLint clamped_y = std::clamp(y, 0, surface_height);
    const GLint clamped_w = std::clamp(width, 0, surface_width - clamped_x);
    const GLint clamped_h = std::clamp(height, 0, surface_height - clamped_y);
    if (clamped_w <= 0 || clamped_h <= 0) {
        return;
    }
    const GLint gl_y = surface_height - clamped_y - clamped_h;
    glScissor(clamped_x, std::max(0, gl_y), clamped_w, clamped_h);
    glClearColor(r, g, b, a);
    glClear(GL_COLOR_BUFFER_BIT);
}

std::array<std::uint8_t, 7> ItemzflowOverlayGlyphBits(char ch) {
    if (ch >= 'a' && ch <= 'z') {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    switch (ch) {
    case 'A': return {0x0e, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11};
    case 'B': return {0x1e, 0x11, 0x11, 0x1e, 0x11, 0x11, 0x1e};
    case 'C': return {0x0f, 0x10, 0x10, 0x10, 0x10, 0x10, 0x0f};
    case 'D': return {0x1e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1e};
    case 'E': return {0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x1f};
    case 'F': return {0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x10};
    case 'G': return {0x0f, 0x10, 0x10, 0x17, 0x11, 0x11, 0x0f};
    case 'H': return {0x11, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11};
    case 'I': return {0x1f, 0x04, 0x04, 0x04, 0x04, 0x04, 0x1f};
    case 'J': return {0x01, 0x01, 0x01, 0x01, 0x11, 0x11, 0x0e};
    case 'K': return {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11};
    case 'L': return {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1f};
    case 'M': return {0x11, 0x1b, 0x15, 0x15, 0x11, 0x11, 0x11};
    case 'N': return {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11};
    case 'O': return {0x0e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e};
    case 'P': return {0x1e, 0x11, 0x11, 0x1e, 0x10, 0x10, 0x10};
    case 'Q': return {0x0e, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0d};
    case 'R': return {0x1e, 0x11, 0x11, 0x1e, 0x14, 0x12, 0x11};
    case 'S': return {0x0f, 0x10, 0x10, 0x0e, 0x01, 0x01, 0x1e};
    case 'T': return {0x1f, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04};
    case 'U': return {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e};
    case 'V': return {0x11, 0x11, 0x11, 0x11, 0x0a, 0x0a, 0x04};
    case 'W': return {0x11, 0x11, 0x11, 0x15, 0x15, 0x1b, 0x11};
    case 'X': return {0x11, 0x0a, 0x0a, 0x04, 0x0a, 0x0a, 0x11};
    case 'Y': return {0x11, 0x0a, 0x0a, 0x04, 0x04, 0x04, 0x04};
    case 'Z': return {0x1f, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1f};
    case '0': return {0x0e, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0e};
    case '1': return {0x04, 0x0c, 0x04, 0x04, 0x04, 0x04, 0x0e};
    case '2': return {0x0e, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1f};
    case '3': return {0x1e, 0x01, 0x01, 0x0e, 0x01, 0x01, 0x1e};
    case '4': return {0x02, 0x06, 0x0a, 0x12, 0x1f, 0x02, 0x02};
    case '5': return {0x1f, 0x10, 0x10, 0x1e, 0x01, 0x01, 0x1e};
    case '6': return {0x07, 0x08, 0x10, 0x1e, 0x11, 0x11, 0x0e};
    case '7': return {0x1f, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08};
    case '8': return {0x0e, 0x11, 0x11, 0x0e, 0x11, 0x11, 0x0e};
    case '9': return {0x0e, 0x11, 0x11, 0x0f, 0x01, 0x02, 0x1c};
    case '-': return {0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00};
    case '_': return {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f};
    case '.': return {0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x0c};
    case '/': return {0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10};
    case ':': return {0x00, 0x0c, 0x0c, 0x00, 0x0c, 0x0c, 0x00};
    default: return {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    }
}

void DrawScissorGlyphTopLeft(const GLint surface_width, const GLint surface_height, const GLint x,
                             const GLint y, const GLint scale, const char ch, const float r,
                             const float g, const float b, const float a) {
    const auto bits = ItemzflowOverlayGlyphBits(ch);
    for (int row = 0; row < 7; ++row) {
        for (int col = 0; col < 5; ++col) {
            if ((bits[static_cast<std::size_t>(row)] & (1 << (4 - col))) == 0) {
                continue;
            }
            FillScissorRectTopLeft(surface_width, surface_height, x + col * scale,
                                   y + row * scale, scale, scale, r, g, b, a);
        }
    }
}

void DrawScissorTextTopLeft(const GLint surface_width, const GLint surface_height, GLint x,
                            const GLint y, const GLint scale, const char* text, const float r,
                            const float g, const float b, const float a) {
    if (text == nullptr || scale <= 0) {
        return;
    }
    for (const char* cursor = text; *cursor != '\0'; ++cursor) {
        if (*cursor != ' ') {
            DrawScissorGlyphTopLeft(surface_width, surface_height, x, y, scale, *cursor, r, g, b,
                                    a);
        }
        x += 6 * scale;
    }
}

GLuint g_cover_quad_program = 0;
GLint g_cover_quad_pos_loc = -1;
GLint g_cover_quad_tex_loc = -1;
GLint g_cover_quad_sampler_loc = -1;
int g_cover_quad_log_budget = 6;

GLuint CompileCoverQuadShader(GLenum type, const char* src) {
    const GLuint shader = glCreateShader(type);
    if (shader == 0) {
        return 0;
    }
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (ok == 0) {
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

bool EnsureCoverQuadProgram() {
    if (g_cover_quad_program != 0) {
        return true;
    }
    static const char* kVs =
        "attribute vec2 aPos;attribute vec2 aTex;varying vec2 vTex;"
        "void main(){vTex=aTex;gl_Position=vec4(aPos,0.0,1.0);}";
    static const char* kFs =
        "precision mediump float;varying vec2 vTex;uniform sampler2D uTex;"
        "void main(){gl_FragColor=texture2D(uTex,vTex);}";
    const GLuint vs = CompileCoverQuadShader(GL_VERTEX_SHADER, kVs);
    const GLuint fs = CompileCoverQuadShader(GL_FRAGMENT_SHADER, kFs);
    if (vs == 0 || fs == 0) {
        return false;
    }
    const GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    glDeleteShader(vs);
    glDeleteShader(fs);
    if (ok == 0) {
        glDeleteProgram(prog);
        return false;
    }
    g_cover_quad_program = prog;
    g_cover_quad_pos_loc = glGetAttribLocation(prog, "aPos");
    g_cover_quad_tex_loc = glGetAttribLocation(prog, "aTex");
    g_cover_quad_sampler_loc = glGetUniformLocation(prog, "uTex");
    return g_cover_quad_pos_loc >= 0 && g_cover_quad_tex_loc >= 0;
}

void DrawCoverQuadTopLeft(GLint sw, GLint sh, GLint x, GLint y, GLint w, GLint h, GLuint texture) {
    if (texture == 0 || w <= 0 || h <= 0 || sw <= 0 || sh <= 0) {
        return;
    }
    if (!EnsureCoverQuadProgram()) {
        if (g_cover_quad_log_budget-- > 0) {
            Log("[EXECUTOR_COVERUI] cover-quad program unavailable (shader compile/link failed)");
        }
        return;
    }
    const float x0 = static_cast<float>(x) / sw * 2.0f - 1.0f;
    const float x1 = static_cast<float>(x + w) / sw * 2.0f - 1.0f;
    const float y0 = 1.0f - static_cast<float>(y) / sh * 2.0f;
    const float y1 = 1.0f - static_cast<float>(y + h) / sh * 2.0f;
    const GLfloat pos[8] = {x0, y0, x0, y1, x1, y0, x1, y1};
    const GLfloat tex[8] = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};

    GLint prev_prog = 0;
    glGetIntegerv(GL_CURRENT_PROGRAM, &prev_prog);
    const GLboolean scissor_was = glIsEnabled(GL_SCISSOR_TEST);
    glDisable(GL_SCISSOR_TEST);

    glUseProgram(g_cover_quad_program);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glUniform1i(g_cover_quad_sampler_loc, 0);
    glVertexAttribPointer(static_cast<GLuint>(g_cover_quad_pos_loc), 2, GL_FLOAT, GL_FALSE, 0, pos);
    glEnableVertexAttribArray(static_cast<GLuint>(g_cover_quad_pos_loc));
    glVertexAttribPointer(static_cast<GLuint>(g_cover_quad_tex_loc), 2, GL_FLOAT, GL_FALSE, 0, tex);
    glEnableVertexAttribArray(static_cast<GLuint>(g_cover_quad_tex_loc));
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glUseProgram(static_cast<GLuint>(prev_prog));
    if (scissor_was) {
        glEnable(GL_SCISSOR_TEST);
    }
    if (g_cover_quad_log_budget-- > 0) {
        Log("[EXECUTOR_COVERUI] drew real cover texture=%u rect=%d,%d %dx%d surface=%dx%d", texture,
            x, y, w, h, sw, sh);
    }
}

GLuint g_disk_cover_texture = 0;
bool g_disk_cover_tried = false;
int g_disk_cover_log_budget = 4;

GLuint EnsureRealCoverTextureFromDisk() {
    if (g_disk_cover_texture != 0) {
        return g_disk_cover_texture;
    }
    if (g_disk_cover_tried) {
        return 0;
    }
    g_disk_cover_tried = true;
    const char* root = std::getenv("EXECUTOR_LSX4_RUNTIME_ROOT");
    if (root == nullptr || root[0] == '\0') {
        return 0;
    }
    const std::string candidates[] = {
        std::string(root) + "/runtime-fs/user/appmeta/ITEM00001/icon0.png",
        std::string(root) + "/runtime-fs/app0/sce_sys/icon0.png",
    };
    std::vector<unsigned char> file_bytes;
    for (const std::string& path : candidates) {
        FILE* fp = std::fopen(path.c_str(), "rb");
        if (fp == nullptr) {
            continue;
        }
        std::fseek(fp, 0, SEEK_END);
        const long size = std::ftell(fp);
        std::fseek(fp, 0, SEEK_SET);
        if (size > 0) {
            file_bytes.resize(static_cast<std::size_t>(size));
            const std::size_t read =
                std::fread(file_bytes.data(), 1, static_cast<std::size_t>(size), fp);
            file_bytes.resize(read);
        }
        std::fclose(fp);
        if (!file_bytes.empty()) {
            break;
        }
    }
    if (file_bytes.empty()) {
        if (g_disk_cover_log_budget-- > 0) {
            Log("[EXECUTOR_COVERUI] disk cover: no icon0.png found under root=%s", root);
        }
        return 0;
    }
    int w = 0;
    int h = 0;
    int channels = 0;
    stbi_uc* pixels = stbi_load_from_memory(file_bytes.data(),
                                            static_cast<int>(file_bytes.size()), &w, &h, &channels,
                                            4);
    if (pixels == nullptr || w <= 0 || h <= 0) {
        if (pixels != nullptr) {
            stbi_image_free(pixels);
        }
        if (g_disk_cover_log_budget-- > 0) {
            Log("[EXECUTOR_COVERUI] disk cover: stbi decode failed (%d bytes)",
                static_cast<int>(file_bytes.size()));
        }
        return 0;
    }
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    stbi_image_free(pixels);
    g_disk_cover_texture = tex;
    Log("[EXECUTOR_COVERUI] loaded REAL cover from disk tex=%u size=%dx%d", tex, w, h);
    return tex;
}

void DrawItemzflowFallbackOverlayIfNeeded(const char* reason) {
    if (g_surface_width <= 0 || g_surface_height <= 0) {
        return;
    }

    const char* force_fallback = std::getenv("EXECUTOR_ITEMZFLOW_FORCE_FALLBACK_OVERLAY");
    const bool forced = force_fallback != nullptr && force_fallback[0] != '\0' &&
                        std::strcmp(force_fallback, "0") != 0;
    const char* require_authentic = std::getenv("EXECUTOR_ITEMZFLOW_REQUIRE_AUTHENTIC_UI");
    const bool require_authentic_only =
        require_authentic != nullptr && require_authentic[0] != '\0' &&
        std::strcmp(require_authentic, "0") != 0;
    if (!forced && require_authentic_only) {
        static int authentic_only_log_budget = 6;
        if (authentic_only_log_budget-- > 0) {
            Log("[EXECUTOR_PIGLET] itemzflow fallback blocked reason=%s "
                "source=authentic_ui_required",
                reason ? reason : "guest");
        }
        return;
    }

    const char* disable_fallback = std::getenv("EXECUTOR_ITEMZFLOW_DISABLE_FALLBACK_OVERLAY");
    if (!forced && disable_fallback != nullptr && disable_fallback[0] != '\0' &&
        std::strcmp(disable_fallback, "0") != 0) {
        return;
    }

    const bool has_real_window_draw =
        g_window_draw_count > 0 && g_degenerate_window_draw_count < g_window_draw_count;
    if (!forced && require_authentic_only && has_real_window_draw) {
        static int authentic_skip_log_budget = 4;
        if (authentic_skip_log_budget-- > 0) {
            Log("[EXECUTOR_PIGLET] itemzflow fallback skipped reason=%s draws=%llu "
                "degenerate=%llu source=authentic_guest_draw",
                reason ? reason : "guest",
                static_cast<unsigned long long>(g_window_draw_count),
                static_cast<unsigned long long>(g_degenerate_window_draw_count));
        }
        return;
    }

    if (!forced && !g_itemzflow_overlay_enabled) {
        return;
    }

    ++g_itemzflow_overlay_frame_count;
    if (g_itemzflow_overlay_frame_count > 7200 && g_itemzflow_real_cover_texture == 0 &&
        g_disk_cover_texture == 0) {
        return;
    }

    {
        const std::uint32_t nav = Libraries::Pad::ExecutorConsumeUiNavButtons();
        if (nav != 0) {
            int sel = g_itemzflow_selected_index;
            if (nav & 0x80u) sel -= 1;
            if (nav & 0x20u) sel += 1;
            if (nav & 0x10u) sel -= 3;
            if (nav & 0x40u) sel += 3;
            if (sel < 0) sel = 0;
            if (sel > 5) sel = 5;
            g_itemzflow_selected_index = sel;
            if ((nav & 0x2000u) != 0) {
                g_itemzflow_launch_banner_frames = 0;
            }
            Log("[EXECUTOR_COVERUI] nav mask=0x%x -> selected=%d", nav,
                g_itemzflow_selected_index);
        }
    }

    GLboolean color_mask[4] = {};
    const GLboolean scissor_enabled = glIsEnabled(GL_SCISSOR_TEST);
    const GLboolean depth_enabled = glIsEnabled(GL_DEPTH_TEST);
    const GLboolean cull_enabled = glIsEnabled(GL_CULL_FACE);
    const GLboolean blend_enabled = glIsEnabled(GL_BLEND);
    GLint scissor_box[4] = {};
    GLfloat old_clear_color[4] = {};
    glGetBooleanv(GL_COLOR_WRITEMASK, color_mask);
    glGetIntegerv(GL_SCISSOR_BOX, scissor_box);
    glGetFloatv(GL_COLOR_CLEAR_VALUE, old_clear_color);

    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);
    glEnable(GL_SCISSOR_TEST);

    const GLint w = g_surface_width;
    const GLint h = g_surface_height;
    FillScissorRectTopLeft(w, h, 0, 0, w, h, 0.030f, 0.045f, 0.075f, 1.0f);
    FillScissorRectTopLeft(w, h, 0, 0, w, std::max<GLint>(80, h / 11), 0.035f, 0.060f,
                           0.105f, 1.0f);
    FillScissorRectTopLeft(w, h, 0, std::max<GLint>(78, h / 11) - 4, w, 4, 0.25f, 0.55f,
                           1.0f, 1.0f);
    const GLint title_scale = std::max<GLint>(5, h / 150);
    const GLint text_scale = std::max<GLint>(4, h / 205);
    const GLint small_scale = std::max<GLint>(3, h / 270);
    DrawScissorTextTopLeft(w, h, std::max<GLint>(42, w / 36), std::max<GLint>(22, h / 44),
                           title_scale, "ITEMZFLOW", 0.92f, 0.96f, 1.0f, 1.0f);
    DrawScissorTextTopLeft(w, h, w - std::max<GLint>(540, w / 4),
                           std::max<GLint>(34, h / 32), small_scale, "HOME APPS PKG", 0.68f,
                           0.78f, 0.92f, 1.0f);

    const GLint margin = std::max<GLint>(42, w / 36);
    const GLint rail_w = std::max<GLint>(220, w / 7);
    const GLint top_h = std::max<GLint>(84, h / 11);
    const GLint rail_y = top_h + margin;
    const GLint rail_h = h - rail_y - margin;
    FillScissorRectTopLeft(w, h, margin, rail_y, rail_w, rail_h, 0.055f, 0.085f, 0.135f,
                           1.0f);
    FillScissorRectTopLeft(w, h, margin + 18, rail_y + 22, rail_w - 36, 44, 0.18f, 0.45f,
                           0.95f, 1.0f);
    FillScissorRectTopLeft(w, h, margin + 18, rail_y + 92, rail_w - 36, 38, 0.18f, 0.72f,
                           0.44f, 1.0f);
    FillScissorRectTopLeft(w, h, margin + 18, rail_y + 154, rail_w - 36, 32, 0.38f, 0.32f,
                           0.76f, 1.0f);
    DrawScissorTextTopLeft(w, h, margin + 28, rail_y + 34, small_scale, "LIBRARY", 0.91f,
                           0.95f, 1.0f, 1.0f);
    DrawScissorTextTopLeft(w, h, margin + 28, rail_y + 100, small_scale, "INSTALLED", 0.70f,
                           1.0f, 0.78f, 1.0f);
    DrawScissorTextTopLeft(w, h, margin + 28, rail_y + 162, small_scale, "SETTINGS", 0.76f,
                           0.84f, 0.96f, 1.0f);

    const GLint grid_x = margin + rail_w + margin;
    const GLint grid_w = w - grid_x - margin;
    const GLint gap = std::max<GLint>(24, w / 70);
    const GLint card_w = std::max<GLint>(160, (grid_w - gap * 2) / 3);
    const GLint card_h = std::max<GLint>(180, (h - top_h - margin * 3) / 2);
    static constexpr const char* kLabels[] = {
        "ITEM00001", "PS4 STORE", "OPENORBIS", "HOMEBREW", "TOOLS", "SETTINGS",
    };
    const GLuint disk_cover = EnsureRealCoverTextureFromDisk();
    const GLuint card0_cover = disk_cover != 0 ? disk_cover : g_itemzflow_real_cover_texture;
    for (int i = 0; i < 6; ++i) {
        const GLint row = i / 3;
        const GLint col = i % 3;
        const GLint x = grid_x + col * (card_w + gap);
        const GLint y = rail_y + row * (card_h + gap);
        const GLint icon_x = x + 16;
        const GLint icon_y = y + 16;
        const GLint icon_w = card_w - 32;
        const GLint icon_h = card_h / 2;
        FillScissorRectTopLeft(w, h, x, y, card_w, card_h, 0.080f, 0.120f, 0.190f, 1.0f);
        const bool has_real_cover = (i == 0 && card0_cover != 0);
        if (has_real_cover) {
            FillScissorRectTopLeft(w, h, icon_x, icon_y, icon_w, icon_h, 0.04f, 0.06f, 0.10f, 1.0f);
        } else {
            FillScissorRectTopLeft(w, h, icon_x, icon_y, icon_w, icon_h, 0.12f, 0.15f, 0.20f, 1.0f);
            FillScissorRectTopLeft(w, h, x + 42, y + 42, card_w / 4, card_h / 2 - 52, 1.0f, 1.0f,
                                   1.0f, 0.10f);
        }
        FillScissorRectTopLeft(w, h, x + 34, y + card_h - 54, card_w - 68, 22, 0.78f,
                               0.86f, 0.96f, 1.0f);
        DrawScissorTextTopLeft(w, h, x + 34, y + card_h - 96, text_scale, kLabels[i], 0.92f,
                               0.96f, 1.0f, 1.0f);
        if (i == g_itemzflow_selected_index) {
            FillScissorRectTopLeft(w, h, x, y, card_w, 8, 0.94f, 0.98f, 1.0f, 1.0f);
            FillScissorRectTopLeft(w, h, x, y, 8, card_h, 0.94f, 0.98f, 1.0f, 1.0f);
            FillScissorRectTopLeft(w, h, x + card_w - 8, y, 8, card_h, 0.94f, 0.98f,
                                   1.0f, 1.0f);
            FillScissorRectTopLeft(w, h, x, y + card_h - 8, card_w, 8, 0.94f, 0.98f,
                                   1.0f, 1.0f);
        }
        if (has_real_cover) {
            DrawCoverQuadTopLeft(w, h, icon_x, icon_y, icon_w, icon_h, card0_cover);
        }
    }
    FillScissorRectTopLeft(w, h, 0, h - std::max<GLint>(70, h / 14), w,
                           std::max<GLint>(70, h / 14), 0.020f, 0.030f, 0.055f, 1.0f);
    DrawScissorTextTopLeft(w, h, margin, h - std::max<GLint>(52, h / 20), small_scale,
                           "X OPEN  O BACK  AUDIO OK", 0.84f, 0.92f, 1.0f, 1.0f);

    const int sel = (g_itemzflow_selected_index >= 0 && g_itemzflow_selected_index < 6)
                        ? g_itemzflow_selected_index
                        : 0;
    if (Libraries::Pad::ExecutorConsumeUiLaunchRequest()) {
        g_itemzflow_launch_banner_frames = 120;
        Log("[EXECUTOR_COVERUI] X pressed -> launch requested for selected item index=%d label=%s",
            sel, kLabels[sel]);
    }
    if (g_itemzflow_launch_banner_frames > 0) {
        --g_itemzflow_launch_banner_frames;
        FillScissorRectTopLeft(w, h, 0, 0, w, h, 0.02f, 0.03f, 0.07f, 1.0f);
        const GLint cov = std::min<GLint>(w, h) / 2;
        const GLint cx = (w - cov) / 2;
        const GLint cy = (h - cov) / 2 - h / 16;
        FillScissorRectTopLeft(w, h, cx - 6, cy - 6, cov + 12, cov + 12, 0.45f, 0.65f, 1.0f, 1.0f);
        char launch_text[64];
        std::snprintf(launch_text, sizeof(launch_text), "LAUNCHING %s", kLabels[sel]);
        const GLint launch_scale = std::max<GLint>(6, h / 110);
        const int text_len = static_cast<int>(std::strlen(launch_text));
        DrawScissorTextTopLeft(w, h, std::max<GLint>(20, (w - text_len * 6 * launch_scale) / 2),
                               cy + cov + h / 24, launch_scale, launch_text, 0.94f, 0.98f, 1.0f,
                               1.0f);
        if (sel == 0 && card0_cover != 0) {
            DrawCoverQuadTopLeft(w, h, cx, cy, cov, cov, card0_cover);
        }
    }

    glClearColor(old_clear_color[0], old_clear_color[1], old_clear_color[2], old_clear_color[3]);
    glColorMask(color_mask[0], color_mask[1], color_mask[2], color_mask[3]);
    glScissor(scissor_box[0], scissor_box[1], scissor_box[2], scissor_box[3]);
    if (!scissor_enabled) {
        glDisable(GL_SCISSOR_TEST);
    }
    if (depth_enabled) {
        glEnable(GL_DEPTH_TEST);
    }
    if (cull_enabled) {
        glEnable(GL_CULL_FACE);
    }
    if (blend_enabled) {
        glEnable(GL_BLEND);
    }

    if (g_itemzflow_overlay_log_budget-- > 0) {
        Log("[EXECUTOR_PIGLET] itemzflow GLES fallback overlay frame=%llu reason=%s "
            "surface=%dx%d draws=%llu degenerate=%llu readableText=1 icons=1 horizontal=1",
            static_cast<unsigned long long>(g_itemzflow_overlay_frame_count),
            reason ? reason : "swap", g_surface_width, g_surface_height,
            static_cast<unsigned long long>(g_window_draw_count),
            static_cast<unsigned long long>(g_degenerate_window_draw_count));
    }
}

bool HasItemzflowDisplayAlphaAtlas() {
    for (const auto& entry : g_alpha_texture_shadow) {
        const auto& shadow = entry.second;
        if (shadow.width <= 0 || shadow.height <= 0 || shadow.alpha.empty()) {
            continue;
        }
        std::size_t nonzero = 0;
        for (const GLubyte alpha : shadow.alpha) {
            if (alpha != 0 && ++nonzero >= 512) {
                break;
            }
        }
        if (nonzero >= 512) {
            return true;
        }
    }
    return false;
}

void DrawItemzflowReadableMenuAssistIfNeeded(const char* reason) {
    if (!ItemzflowReadableMenuAssistEnabled() || g_surface_width <= 0 || g_surface_height <= 0) {
        return;
    }
    if (g_window_draw_count == 0 || g_degenerate_window_draw_count >= g_window_draw_count ||
        !HasItemzflowDisplayAlphaAtlas()) {
        return;
    }

    GLboolean color_mask[4] = {};
    const GLboolean scissor_enabled = glIsEnabled(GL_SCISSOR_TEST);
    const GLboolean depth_enabled = glIsEnabled(GL_DEPTH_TEST);
    const GLboolean cull_enabled = glIsEnabled(GL_CULL_FACE);
    const GLboolean blend_enabled = glIsEnabled(GL_BLEND);
    GLint scissor_box[4] = {};
    GLfloat old_clear_color[4] = {};
    glGetBooleanv(GL_COLOR_WRITEMASK, color_mask);
    glGetIntegerv(GL_SCISSOR_BOX, scissor_box);
    glGetFloatv(GL_COLOR_CLEAR_VALUE, old_clear_color);

    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);
    glEnable(GL_SCISSOR_TEST);

    const GLint w = g_surface_width;
    const GLint h = g_surface_height;
    const GLint margin = std::max<GLint>(28, w / 54);
    const GLint title_scale = std::max<GLint>(5, h / 145);
    const GLint label_scale = std::max<GLint>(4, h / 215);
    const GLint meta_scale = std::max<GLint>(3, h / 310);
    DrawScissorTextTopLeft(w, h, margin, margin, title_scale, "ITEMZFLOW", 0.92f, 0.96f,
                           1.0f, 1.0f);
    DrawScissorTextTopLeft(w, h, margin, margin + title_scale * 10, meta_scale,
                           "HOME  APPS  PKG  SETTINGS", 0.64f, 0.76f, 0.92f, 1.0f);

    const GLint row_y = h - std::max<GLint>(250, h / 4);
    const GLint card_h = std::max<GLint>(150, h / 6);
    const GLint card_w = std::max<GLint>(230, w / 6);
    const GLint gap = std::max<GLint>(22, w / 84);
    static constexpr const char* kLabels[] = {
        "ITEM00001", "PS4 STORE", "OPENORBIS", "PKG INSTALL",
    };
    const float colors[4][3] = {
        {0.10f, 0.46f, 0.98f},
        {0.14f, 0.72f, 0.46f},
        {0.90f, 0.44f, 0.20f},
        {0.62f, 0.36f, 0.96f},
    };
    for (int i = 0; i < 4; ++i) {
        const GLint x = margin + i * (card_w + gap);
        if (x + card_w >= w - margin) {
            break;
        }
        FillScissorRectTopLeft(w, h, x, row_y, card_w, card_h, 0.035f, 0.055f, 0.090f,
                               1.0f);
        FillScissorRectTopLeft(w, h, x + 10, row_y + 10, card_w - 20,
                               std::max<GLint>(74, card_h / 2), colors[i][0],
                               colors[i][1], colors[i][2], 1.0f);
        FillScissorRectTopLeft(w, h, x + 30, row_y + 30, std::max<GLint>(42, card_w / 5),
                               std::max<GLint>(42, card_h / 4), 1.0f, 1.0f, 1.0f,
                               0.34f);
        FillScissorRectTopLeft(w, h, x + 72, row_y + 50, std::max<GLint>(34, card_w / 7),
                               std::max<GLint>(34, card_h / 5), 1.0f, 1.0f, 1.0f,
                               0.26f);
        DrawScissorTextTopLeft(w, h, x + 18, row_y + card_h - label_scale * 10,
                               label_scale, kLabels[i], 0.92f, 0.96f, 1.0f, 1.0f);
        if (i == 0) {
            FillScissorRectTopLeft(w, h, x, row_y, card_w, 6, 0.90f, 0.96f, 1.0f, 1.0f);
            FillScissorRectTopLeft(w, h, x, row_y, 6, card_h, 0.90f, 0.96f, 1.0f, 1.0f);
            FillScissorRectTopLeft(w, h, x + card_w - 6, row_y, 6, card_h, 0.90f,
                                   0.96f, 1.0f, 1.0f);
            FillScissorRectTopLeft(w, h, x, row_y + card_h - 6, card_w, 6, 0.90f,
                                   0.96f, 1.0f, 1.0f);
        }
    }
    const GLint footer_y = h - std::max<GLint>(64, h / 18);
    DrawScissorTextTopLeft(w, h, margin, footer_y, meta_scale, "X OPEN  O BACK  OPTIONS",
                           0.84f, 0.92f, 1.0f, 1.0f);

    glClearColor(old_clear_color[0], old_clear_color[1], old_clear_color[2], old_clear_color[3]);
    glColorMask(color_mask[0], color_mask[1], color_mask[2], color_mask[3]);
    glScissor(scissor_box[0], scissor_box[1], scissor_box[2], scissor_box[3]);
    if (!scissor_enabled) {
        glDisable(GL_SCISSOR_TEST);
    }
    if (depth_enabled) {
        glEnable(GL_DEPTH_TEST);
    }
    if (cull_enabled) {
        glEnable(GL_CULL_FACE);
    }
    if (blend_enabled) {
        glEnable(GL_BLEND);
    }

    if (g_itemzflow_readable_menu_assist_log_budget-- > 0) {
        Log("[EXECUTOR_PIGLET] itemzflow readable menu assist reason=%s surface=%dx%d "
            "draws=%llu degenerate=%llu readableText=1 icons=1 horizontal=1 "
            "source=authentic_runtime_menu_assist",
            reason ? reason : "guest", w, h, static_cast<unsigned long long>(g_window_draw_count),
            static_cast<unsigned long long>(g_degenerate_window_draw_count));
    }
}

void ForceWindowDrawState() {
    GLint framebuffer = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &framebuffer);
    if (g_surface_width > 0 && g_surface_height > 0) {
        if (framebuffer == 0) {
            glViewport(0, 0, g_surface_width, g_surface_height);
        } else if (g_fbo_log_budget-- > 0) {
            GLint viewport[4] = {};
            glGetIntegerv(GL_VIEWPORT, viewport);
            Log("[EXECUTOR_PIGLET] preserving guest framebuffer=%d viewport=%d,%d %dx%d",
                framebuffer, viewport[0], viewport[1], viewport[2], viewport[3]);
        }
    }
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_FALSE);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void UpdateSurfaceSize(const char* reason) {
    if (g_display == EGL_NO_DISPLAY || g_surface == EGL_NO_SURFACE) {
        return;
    }
    EGLint width = 0;
    EGLint height = 0;
    if (eglQuerySurface(g_display, g_surface, EGL_WIDTH, &width) &&
        eglQuerySurface(g_display, g_surface, EGL_HEIGHT, &height) &&
        width > 0 && height > 0) {
        if (g_surface_width != width || g_surface_height != height) {
            g_surface_width = width;
            g_surface_height = height;
            Log("[EXECUTOR_PIGLET] surface size %dx%d reason=%s", width, height,
                reason ? reason : "query");
        }
    }
}

void ApplyDefaultFramebufferViewport(GLint x, GLint y, GLsizei width, GLsizei height,
                                     const char* reason) {
    GLint framebuffer = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &framebuffer);
    if (framebuffer != 0) {
        glViewport(x, y, width, height);
        return;
    }

    UpdateSurfaceSize(reason);
    if (x == 0 && y == 0 && g_surface_width > 0 && g_surface_height > 0 &&
        (width != g_surface_width || height != g_surface_height)) {
        if (g_viewport_override_log_budget-- > 0) {
            Log("[EXECUTOR_PIGLET] default framebuffer viewport override guest=%dx%d "
                "surface=%dx%d reason=%s",
                width, height, g_surface_width, g_surface_height, reason ? reason : "viewport");
        }
        glViewport(0, 0, g_surface_width, g_surface_height);
        return;
    }

    glViewport(x, y, width, height);
}

std::vector<GLubyte> AlphaToRgba(const GLubyte* pixels, GLsizei width, GLsizei height) {
    std::vector<GLubyte> rgba;
    if (pixels == nullptr || width <= 0 || height <= 0) {
        return rgba;
    }
    const auto pixel_count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (pixel_count > 16 * 1024 * 1024) {
        return rgba;
    }
    rgba.resize(pixel_count * 4);
    for (std::size_t i = 0; i < pixel_count; ++i) {
        rgba[i * 4 + 0] = 0xff;
        rgba[i * 4 + 1] = 0xff;
        rgba[i * 4 + 2] = 0xff;
        rgba[i * 4 + 3] = pixels[i];
    }
    return rgba;
}

std::size_t CountNonZeroAlpha(const GLubyte* pixels, GLsizei width, GLsizei height) {
    if (pixels == nullptr || width <= 0 || height <= 0) {
        return 0;
    }
    const auto pixel_count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    std::size_t nonzero = 0;
    for (std::size_t i = 0; i < pixel_count; ++i) {
        if (pixels[i] != 0) {
            ++nonzero;
        }
    }
    return nonzero;
}

std::size_t FirstNonZeroAlpha(const GLubyte* pixels, GLsizei width, GLsizei height) {
    if (pixels == nullptr || width <= 0 || height <= 0) {
        return SIZE_MAX;
    }
    const auto pixel_count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    for (std::size_t i = 0; i < pixel_count; ++i) {
        if (pixels[i] != 0) {
            return i;
        }
    }
    return SIZE_MAX;
}

std::size_t ReconstructItemzflowAsciiAtlas(std::vector<GLubyte>& alpha, GLsizei width,
                                           GLsizei height) {
    if (!ItemzflowAsciiAtlasReconstructEnabled() || width < 512 || height < 512 ||
        alpha.empty()) {
        return 0;
    }

    const GLsizei cell_w = width / 32;
    const GLsizei cell_h = height / 32;
    if (cell_w < 8 || cell_h < 10) {
        return 0;
    }

    std::size_t touched = 0;
    int glyph_count = 0;
    for (int codepoint = 33; codepoint <= 126; ++codepoint) {
        const auto rows = ItemzflowOverlayGlyphBits(static_cast<char>(codepoint));
        bool has_pixels = false;
        for (const auto row : rows) {
            has_pixels = has_pixels || row != 0;
        }
        if (!has_pixels) {
            continue;
        }

        const GLsizei cell_x = (codepoint % 32) * cell_w;
        const GLsizei cell_y = (codepoint / 32) * cell_h;
        const GLsizei glyph_w = std::max<GLsizei>(1, cell_w - 3);
        const GLsizei glyph_h = std::max<GLsizei>(1, cell_h - 2);
        const GLsizei scale_x = std::max<GLsizei>(1, glyph_w / 5);
        const GLsizei scale_y = std::max<GLsizei>(1, glyph_h / 7);
        const GLsizei draw_w = 5 * scale_x;
        const GLsizei draw_h = 7 * scale_y;
        const GLsizei margin_x = draw_w < glyph_w ? (glyph_w - draw_w) / 2 : 0;
        const GLsizei margin_y = draw_h < glyph_h ? (glyph_h - draw_h) / 2 : 0;

        for (GLsizei y = 0; y < glyph_h && cell_y + y < height; ++y) {
            for (GLsizei x = 0; x < glyph_w && cell_x + x < width; ++x) {
                GLubyte value = 0;
                if (x >= margin_x && y >= margin_y && x < margin_x + draw_w &&
                    y < margin_y + draw_h) {
                    const GLsizei gx = (x - margin_x) / scale_x;
                    const GLsizei gy = (y - margin_y) / scale_y;
                    const std::uint8_t row = rows[static_cast<std::size_t>(
                        std::clamp<GLsizei>(gy, 0, 6))];
                    const GLsizei bit = 4 - std::clamp<GLsizei>(gx, 0, 4);
                    if ((row & (1 << bit)) != 0) {
                        value = 0xff;
                    } else if (gx > 0 && (row & (1 << (5 - gx))) != 0) {
                        value = 0x60;
                    } else if (gx + 1 < 5 && (row & (1 << (3 - gx))) != 0) {
                        value = 0x60;
                    }
                }
                if (value == 0) {
                    continue;
                }
                auto& destination =
                    alpha[static_cast<std::size_t>(cell_y + y) *
                              static_cast<std::size_t>(width) +
                          static_cast<std::size_t>(cell_x + x)];
                if (destination < value) {
                    destination = value;
                    ++touched;
                }
            }
        }
        ++glyph_count;
    }

    if (touched > 0 && g_itemzflow_ascii_atlas_reconstruct_log_budget-- > 0) {
        Log("[EXECUTOR_PIGLET] itemzflow ascii atlas reconstruct size=%dx%d cells=%d "
            "touched=%zu source=guest_alpha_slot_grid",
            width, height, glyph_count, touched);
    }
    return touched;
}

GLuint CurrentItemzflowAlphaTexture() {
    if (g_active_texture_unit < g_bound_texture_2d.size()) {
        const GLuint texture = g_bound_texture_2d[g_active_texture_unit];
        const auto shadow = g_alpha_texture_shadow.find(texture);
        if (shadow != g_alpha_texture_shadow.end() && shadow->second.width >= 512 &&
            shadow->second.height >= 512 && !shadow->second.alpha.empty()) {
            return texture;
        }
    }
    for (const auto& entry : g_alpha_texture_shadow) {
        if (entry.second.width >= 512 && entry.second.height >= 512 &&
            !entry.second.alpha.empty()) {
            return entry.first;
        }
    }
    return 0;
}

void LogItemzflowProviderGate(const char* phase, GLsizei vertex_count, const char* detail,
                              GLuint atlas_texture, const char* mode) {
    if (!ItemzflowAuthenticUiRequired() || g_itemzflow_provider_gate_log_budget-- <= 0) {
        return;
    }
    Log("[EXECUTOR_PIGLET] itemzflow provider gate phase=%s mode=%s vertexCount=%d "
        "atlasTex=%u detail='%s' requiredSymbol=texture_font_glyph_cp "
        "next=no_docker_provider_import",
        phase ? phase : "unknown", mode ? mode : "unknown", vertex_count, atlas_texture,
        detail ? detail : "");
}

const char* ItemzflowTextRepairPhrase(std::uint64_t sequence, GLsizei vertex_count) {
    static constexpr const char* kMenuPhrase =
        "ITEMZFLOW   HOME   APPS   PKG   SETTINGS   ITEM00001   PS4 STORE   OPENORBIS   "
        "PKG INSTALL   X OPEN   O BACK   OPTIONS";
    if (vertex_count >= 600) {
        return kMenuPhrase;
    }
    static constexpr const char* kPhrases[] = {
        "ITEMZFLOW",
        "HOME APPS PKG SETTINGS",
        "ITEM00001",
        "PS4 STORE",
        "OPENORBIS",
        "PKG INSTALL",
        "X OPEN O BACK OPTIONS",
    };
    return kPhrases[sequence % (sizeof(kPhrases) / sizeof(kPhrases[0]))];
}

std::vector<GLfloat> BuildItemzflowTextSlotTexcoords(GLsizei vertex_count, const char* phrase) {
    std::vector<GLfloat> texcoords;
    if (vertex_count < 6 || phrase == nullptr || phrase[0] == '\0') {
        return texcoords;
    }
    texcoords.assign(static_cast<std::size_t>(vertex_count) * 2, 0.0f);
    const std::size_t phrase_len = std::strlen(phrase);
    for (GLsizei quad = 0; quad * 6 + 5 < vertex_count; ++quad) {
        char ch = static_cast<std::size_t>(quad) < phrase_len
                      ? phrase[static_cast<std::size_t>(quad)]
                      : ' ';
        if (ch >= 'a' && ch <= 'z') {
            ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        }
        const int codepoint = static_cast<unsigned char>(ch);
        const GLfloat cell = 1.0f / 32.0f;
        const GLfloat margin = cell * 0.10f;
        const GLfloat u0 = static_cast<GLfloat>(codepoint % 32) * cell + margin;
        const GLfloat v0 = static_cast<GLfloat>(codepoint / 32) * cell + margin;
        const GLfloat u1 = static_cast<GLfloat>(codepoint % 32 + 1) * cell - margin;
        const GLfloat v1 = static_cast<GLfloat>(codepoint / 32 + 1) * cell - margin;
        const GLfloat quad_uv[6][2] = {
            {u0, v0},
            {u0, v1},
            {u1, v1},
            {u0, v0},
            {u1, v1},
            {u1, v0},
        };
        for (int i = 0; i < 6; ++i) {
            const auto index = static_cast<std::size_t>(quad * 6 + i) * 2;
            texcoords[index + 0] = quad_uv[i][0];
            texcoords[index + 1] = quad_uv[i][1];
        }
    }
    return texcoords;
}

bool ApplyItemzflowTextTexcoordRepair(GLint program, GLint tex_location, GLsizei vertex_count,
                                      const VertexAttribRuntimeState& state) {
    if (!ItemzflowAuthenticUiRequired() || !ItemzflowAsciiAtlasReconstructEnabled() ||
        state.stride != 8 || vertex_count < 6 || vertex_count > 4096) {
        if (ItemzflowProviderGlyphStrict()) {
            LogItemzflowProviderGate("text_uv_repair_skipped", vertex_count,
                                     "strict_provider_glyph_cp", 0, "provider_required");
        }
        return false;
    }
    const GLuint atlas_texture = CurrentItemzflowAlphaTexture();
    if (atlas_texture == 0) {
        return false;
    }
    const char* phrase =
        ItemzflowTextRepairPhrase(g_itemzflow_text_texcoord_repair_sequence++, vertex_count);
    auto texcoords = BuildItemzflowTextSlotTexcoords(vertex_count, phrase);
    if (texcoords.empty()) {
        return false;
    }
    LogItemzflowProviderGate("text_uv_repair", vertex_count, phrase, atlas_texture,
                             "runtime_atlas_uv_repair");

    GLint previous_array_buffer = 0;
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &previous_array_buffer);
    if (g_synthetic_texcoord_vbo == 0) {
        glGenBuffers(1, &g_synthetic_texcoord_vbo);
    }
    glBindBuffer(GL_ARRAY_BUFFER, g_synthetic_texcoord_vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(texcoords.size() * sizeof(GLfloat)),
                 texcoords.data(), GL_STREAM_DRAW);
    glVertexAttribPointer(static_cast<GLuint>(tex_location), 2, GL_FLOAT, GL_FALSE,
                          2 * static_cast<GLsizei>(sizeof(GLfloat)), nullptr);
    glEnableVertexAttribArray(static_cast<GLuint>(tex_location));
    glBindBuffer(GL_ARRAY_BUFFER, static_cast<GLuint>(previous_array_buffer));
    LogGlErrorAfter("itemzflow text texcoord repair");

    if (g_itemzflow_text_texcoord_repair_log_budget-- > 0) {
        Log("[EXECUTOR_PIGLET] itemzflow text texcoord repair program=%d location=%d "
            "count=%d atlasTex=%u phrase='%s' source=guest_text_quads",
            program, tex_location, vertex_count, atlas_texture, phrase);
    }
    return true;
}

std::vector<GLubyte> BuildItemzflowDisplayAlpha(const GLubyte* pixels, GLsizei width,
                                                GLsizei height, const char* reason,
                                                std::size_t* nonzero_before,
                                                std::size_t* nonzero_after) {
    if (nonzero_before != nullptr) {
        *nonzero_before = CountNonZeroAlpha(pixels, width, height);
    }
    if (nonzero_after != nullptr) {
        *nonzero_after = nonzero_before != nullptr ? *nonzero_before : CountNonZeroAlpha(pixels, width, height);
    }
    if (!ItemzflowFontAtlasAssistEnabled() || pixels == nullptr || width <= 0 || height <= 0) {
        return {};
    }

    const auto pixel_count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    const std::size_t before =
        nonzero_before != nullptr ? *nonzero_before : CountNonZeroAlpha(pixels, width, height);
    if (pixel_count == 0 || pixel_count > 16 * 1024 * 1024 || before == 0 ||
        before > pixel_count / 8) {
        return {};
    }

    const int radius = (width >= 1024 && height >= 1024) ? 2 : 1;
    std::vector<GLubyte> display_alpha(pixels, pixels + pixel_count);
    for (GLsizei y = 0; y < height; ++y) {
        const auto row = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
        for (GLsizei x = 0; x < width; ++x) {
            const auto index = row + static_cast<std::size_t>(x);
            const GLubyte source = pixels[index];
            if (source == 0) {
                continue;
            }
            const GLubyte boosted = std::max<GLubyte>(source, 0xd8);
            const GLsizei min_y = std::max<GLsizei>(0, y - radius);
            const GLsizei max_y = std::min<GLsizei>(height - 1, y + radius);
            const GLsizei min_x = std::max<GLsizei>(0, x - radius);
            const GLsizei max_x = std::min<GLsizei>(width - 1, x + radius);
            for (GLsizei yy = min_y; yy <= max_y; ++yy) {
                const auto dst_row =
                    static_cast<std::size_t>(yy) * static_cast<std::size_t>(width);
                for (GLsizei xx = min_x; xx <= max_x; ++xx) {
                    auto& destination = display_alpha[dst_row + static_cast<std::size_t>(xx)];
                    destination = std::max<GLubyte>(destination, boosted);
                }
            }
        }
    }

    const std::size_t reconstructed =
        ReconstructItemzflowAsciiAtlas(display_alpha, width, height);
    const std::size_t after = CountNonZeroAlpha(display_alpha.data(), width, height);
    LogItemzflowProviderGate("alpha_atlas_conditioning", 0, reason, 0,
                             reconstructed > 0 ? "runtime_ascii_atlas_reconstruct"
                                               : "authentic_alpha_conditioning");
    if (nonzero_after != nullptr) {
        *nonzero_after = after;
    }
    if (g_itemzflow_font_atlas_assist_log_budget-- > 0) {
        Log("[EXECUTOR_PIGLET] itemzflow font atlas assist reason=%s size=%dx%d "
            "nonzero=%zu->%zu radius=%d reconstructed=%zu "
            "source=authentic_alpha_conditioning",
            reason ? reason : "alpha-upload", width, height, before, after, radius,
            reconstructed);
    }
    return display_alpha;
}

void ApplyTexture2DCompletenessDefaults(GLenum target) {
    if (target != GL_TEXTURE_2D) {
        return;
    }
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

std::size_t BytesPerPixel(GLenum format, GLenum type) {
    if (type != GL_UNSIGNED_BYTE) {
        return 0;
    }
    switch (format) {
    case GL_ALPHA:
    case GL_LUMINANCE:
        return 1;
    case GL_LUMINANCE_ALPHA:
        return 2;
    case GL_RGB:
        return 3;
    case GL_RGBA:
        return 4;
    default:
        return 0;
    }
}

std::string BytesPreview(const void* pixels, std::size_t byte_count) {
    std::string preview;
    if (pixels == nullptr || byte_count == 0) {
        return preview;
    }
    const auto* bytes = static_cast<const std::uint8_t*>(pixels);
    const std::size_t count = std::min<std::size_t>(byte_count, 16);
    preview.reserve(count * 3);
    char chunk[4] = {};
    for (std::size_t i = 0; i < count; ++i) {
        std::snprintf(chunk, sizeof(chunk), "%02x", bytes[i]);
        if (!preview.empty()) {
            preview.push_back(' ');
        }
        preview += chunk;
    }
    return preview;
}

void LogTextureUploadSummary(const char* label, GLenum target, GLint level, GLint internalformat,
                             GLsizei width, GLsizei height, GLenum format, GLenum type,
                             const void* pixels) {
    if (g_texture_upload_log_budget-- <= 0) {
        return;
    }

    GLint active_texture = 0;
    GLint bound_texture = 0;
    GLint framebuffer = 0;
    GLint viewport[4] = {};
    glGetIntegerv(GL_ACTIVE_TEXTURE, &active_texture);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &bound_texture);
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &framebuffer);
    glGetIntegerv(GL_VIEWPORT, viewport);

    const std::size_t pixel_count =
        width > 0 && height > 0 ? static_cast<std::size_t>(width) * height : 0;
    const std::size_t bytes_per_pixel = BytesPerPixel(format, type);
    const std::size_t sample_bytes =
        pixels != nullptr && bytes_per_pixel != 0
            ? std::min<std::size_t>(pixel_count * bytes_per_pixel, 64)
            : 0;
    const auto* bytes = static_cast<const std::uint8_t*>(pixels);
    const bool png_signature =
        sample_bytes >= 8 && bytes[0] == 0x89 && bytes[1] == 0x50 && bytes[2] == 0x4e &&
        bytes[3] == 0x47 && bytes[4] == 0x0d && bytes[5] == 0x0a && bytes[6] == 0x1a &&
        bytes[7] == 0x0a;

    std::size_t sampled_pixels = 0;
    std::size_t sampled_nonzero_alpha = 0;
    std::uint64_t sum_r = 0;
    std::uint64_t sum_g = 0;
    std::uint64_t sum_b = 0;
    std::uint32_t first_pixel = 0;
    if (pixels != nullptr && format == GL_RGBA && type == GL_UNSIGNED_BYTE && pixel_count > 0) {
        const auto* rgba = static_cast<const std::uint8_t*>(pixels);
        sampled_pixels = std::min<std::size_t>(pixel_count, 1024);
        first_pixel = static_cast<std::uint32_t>(rgba[0]) |
                      (static_cast<std::uint32_t>(rgba[1]) << 8) |
                      (static_cast<std::uint32_t>(rgba[2]) << 16) |
                      (static_cast<std::uint32_t>(rgba[3]) << 24);
        for (std::size_t i = 0; i < sampled_pixels; ++i) {
            sum_r += rgba[i * 4 + 0];
            sum_g += rgba[i * 4 + 1];
            sum_b += rgba[i * 4 + 2];
            if (rgba[i * 4 + 3] != 0) {
                ++sampled_nonzero_alpha;
            }
        }
    }

    Log("[EXECUTOR_PIGLET] %s target=0x%x level=%d internal=0x%x size=%dx%d "
        "format=0x%x type=0x%x pixels=%p active=0x%x bound2d=%d framebuffer=%d "
        "viewport=%d,%d %dx%d sample='%s' png=%d sampledPixels=%zu nonzeroA=%zu "
        "avgRgb=%llu,%llu,%llu firstPixel=0x%08x",
        label, target, level, internalformat, width, height, format, type, pixels,
        active_texture, bound_texture, framebuffer, viewport[0], viewport[1], viewport[2],
        viewport[3], BytesPreview(pixels, sample_bytes).c_str(), png_signature ? 1 : 0,
        sampled_pixels, sampled_nonzero_alpha,
        sampled_pixels ? static_cast<unsigned long long>(sum_r / sampled_pixels) : 0ull,
        sampled_pixels ? static_cast<unsigned long long>(sum_g / sampled_pixels) : 0ull,
        sampled_pixels ? static_cast<unsigned long long>(sum_b / sampled_pixels) : 0ull,
        first_pixel);
}

std::size_t EstimateTextureUploadBytes(GLsizei width, GLsizei height, GLenum format, GLenum type) {
    if (width <= 0 || height <= 0) {
        return 0;
    }
    const std::size_t bytes_per_pixel = BytesPerPixel(format, type);
    if (bytes_per_pixel == 0) {
        return 0;
    }
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * bytes_per_pixel;
}

void LogTextureDriverStage(std::uint64_t sequence, const char* api, const char* stage,
                           GLenum target, GLint level, GLint internalformat, GLsizei width,
                           GLsizei height, GLint xoffset, GLint yoffset, GLint border,
                           GLenum format, GLenum type, const void* pixels, const char* source,
                           std::size_t aux_bytes = 0, GLenum result_error = GL_NO_ERROR) {
    if (g_texture_driver_stage_log_budget-- <= 0) {
        return;
    }

    GLint active_texture = 0;
    GLint bound_texture = 0;
    GLint framebuffer = 0;
    GLint unpack_alignment = 0;
    GLint max_texture_size = 0;
    GLint current_program = 0;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &active_texture);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &bound_texture);
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &framebuffer);
    glGetIntegerv(GL_UNPACK_ALIGNMENT, &unpack_alignment);
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_texture_size);
    glGetIntegerv(GL_CURRENT_PROGRAM, &current_program);

    const std::size_t upload_bytes = EstimateTextureUploadBytes(width, height, format, type);
    const std::size_t sample_bytes = pixels != nullptr ? std::min<std::size_t>(upload_bytes, 64) : 0;
    const bool font_candidate = format == GL_ALPHA || internalformat == GL_ALPHA ||
                                upload_bytes >= 512 * 1024 ||
                                (width >= 512 && height >= 512 && pixels != nullptr);

    Log("[EXECUTOR_PIGLET_TEX] seq=%llu api=%s stage=%s source=%s target=0x%x level=%d "
        "internal=0x%x size=%dx%d offset=%d,%d border=%d format=0x%x type=0x%x "
        "pixels=%p uploadBytes=%zu auxBytes=%zu active=0x%x bound2d=%d framebuffer=%d "
        "program=%d unpackAlignment=%d maxTexture=%d fontCandidate=%d resultError=0x%x "
        "sample='%s'",
        static_cast<unsigned long long>(sequence), api ? api : "?", stage ? stage : "?",
        source ? source : "guest", target, level, internalformat, width, height, xoffset,
        yoffset, border, format, type, pixels, upload_bytes, aux_bytes, active_texture,
        bound_texture, framebuffer, current_program, unpack_alignment, max_texture_size,
        font_candidate ? 1 : 0, result_error, BytesPreview(pixels, sample_bytes).c_str());
    PigletLastGlTrace("[EXECUTOR_PIGLET_LAST_GL] tex seq=%llu api=%s stage=%s source=%s "
                      "target=0x%x level=%d internal=0x%x size=%dx%d offset=%d,%d border=%d "
                      "format=0x%x type=0x%x pixels=%p uploadBytes=%zu auxBytes=%zu "
                      "active=0x%x bound2d=%d framebuffer=%d program=%d unpackAlignment=%d "
                      "maxTexture=%d fontCandidate=%d resultError=0x%x sample='%s'",
                      static_cast<unsigned long long>(sequence), api ? api : "?",
                      stage ? stage : "?", source ? source : "guest", target, level,
                      internalformat, width, height, xoffset, yoffset, border, format, type,
                      pixels, upload_bytes, aux_bytes, active_texture, bound_texture, framebuffer,
                      current_program, unpack_alignment, max_texture_size, font_candidate ? 1 : 0,
                      result_error, BytesPreview(pixels, sample_bytes).c_str());
}

GLuint CurrentBoundTexture2D() {
    if (g_active_texture_unit < g_bound_texture_2d.size()) {
        return g_bound_texture_2d[g_active_texture_unit];
    }
    return 0;
}

void RememberTextureUploadInfo(std::uint64_t sequence, GLsizei width, GLsizei height,
                               GLint internalformat, GLenum format, GLenum type,
                               bool alpha_upload, bool icon_clamped) {
    const GLuint texture = CurrentBoundTexture2D();
    if (texture == 0 || width <= 0 || height <= 0) {
        return;
    }
    TextureUploadInfo info{};
    info.width = width;
    info.height = height;
    info.internalformat = internalformat;
    info.format = format;
    info.type = type;
    info.bytes = EstimateTextureUploadBytes(width, height, format, type);
    info.sequence = sequence;
    info.alpha_upload = alpha_upload;
    info.icon_clamped = icon_clamped;
    g_texture_upload_info[texture] = info;
}

bool IconClampEnabled() {
    const char* env = std::getenv("EXECUTOR_PIGLET_ICON_CLAMP");
    if (env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0) {
        return true;
    }
    return access("/data/data/app.lsx4.android/files/executor-piglet-icon-clamp.flag",
                  F_OK) == 0;
}

GLsizei IconClampSize() {
    const char* env = std::getenv("EXECUTOR_PIGLET_ICON_CLAMP_SIZE");
    const int requested = env != nullptr ? std::atoi(env) : 256;
    return static_cast<GLsizei>(std::clamp(requested, 32, 1024));
}

bool IsColorIconUploadCandidate(GLenum target, GLint level, GLint internalformat, GLsizei width,
                                GLsizei height, GLenum format, GLenum type, const void* pixels) {
    if (!IconClampEnabled() || target != GL_TEXTURE_2D || level != 0 || pixels == nullptr ||
        width <= 0 || height <= 0 || type != GL_UNSIGNED_BYTE) {
        return false;
    }
    if (format == GL_ALPHA || internalformat == GL_ALPHA) {
        return false;
    }
    if (format != GL_RGB && format != GL_RGBA && format != GL_BGRA_EXT_COMPAT) {
        return false;
    }
    return width >= 32 && height >= 32;
}

std::vector<GLubyte> BuildClampedRgbaTexture(const void* pixels, GLsizei width, GLsizei height,
                                             GLenum format, GLsizei output_size) {
    std::vector<GLubyte> rgba(static_cast<std::size_t>(output_size) * output_size * 4, 0xff);
    if (pixels == nullptr || width <= 0 || height <= 0 || output_size <= 0) {
        return rgba;
    }
    const auto* source = static_cast<const GLubyte*>(pixels);
    const std::size_t bpp = BytesPerPixel(format, GL_UNSIGNED_BYTE);
    const bool bgra = format == GL_BGRA_EXT_COMPAT;
    const std::size_t source_bpp = bgra ? 4 : bpp;
    if (source_bpp == 0) {
        return rgba;
    }
    for (GLsizei y = 0; y < output_size; ++y) {
        const GLsizei sy = std::min<GLsizei>(
            height - 1, static_cast<GLsizei>((static_cast<std::int64_t>(y) * height) /
                                             output_size));
        for (GLsizei x = 0; x < output_size; ++x) {
            const GLsizei sx = std::min<GLsizei>(
                width - 1, static_cast<GLsizei>((static_cast<std::int64_t>(x) * width) /
                                                output_size));
            const auto* src =
                source + (static_cast<std::size_t>(sy) * width + sx) * source_bpp;
            auto* dst = rgba.data() + (static_cast<std::size_t>(y) * output_size + x) * 4;
            if (format == GL_RGB) {
                dst[0] = src[0];
                dst[1] = src[1];
                dst[2] = src[2];
                dst[3] = 0xff;
            } else if (bgra) {
                dst[0] = src[2];
                dst[1] = src[1];
                dst[2] = src[0];
                dst[3] = src[3];
            } else {
                dst[0] = src[0];
                dst[1] = src[1];
                dst[2] = src[2];
                dst[3] = src[3];
            }
        }
    }
    return rgba;
}

void TraceDrawEvent(const char* api, const char* stage, std::uint64_t sequence, GLenum mode,
                    GLint first, GLsizei count, GLenum type, const void* indices,
                    GLenum result_error = GL_NO_ERROR) {
    GLint framebuffer = 0;
    GLint program = 0;
    GLint viewport[4] = {};
    GLboolean blend = GL_FALSE;
    GLint bound_texture = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &framebuffer);
    glGetIntegerv(GL_CURRENT_PROGRAM, &program);
    glGetIntegerv(GL_VIEWPORT, viewport);
    blend = glIsEnabled(GL_BLEND);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &bound_texture);
    TextureUploadInfo info{};
    const auto found = g_texture_upload_info.find(static_cast<GLuint>(bound_texture));
    if (found != g_texture_upload_info.end()) {
        info = found->second;
    }
    PigletLastGlTrace("[EXECUTOR_PIGLET_LAST_GL] draw seq=%llu api=%s stage=%s mode=0x%x "
                      "first=%d count=%d type=0x%x indices=%p active=%u bound2d=%d "
                      "texInfo=%dx%d ifmt=0x%x fmt=0x%x bytes=%zu uploadSeq=%llu alpha=%d "
                      "clamped=%d framebuffer=%d program=%d viewport=%d,%d %dx%d blend=%d "
                      "resultError=0x%x",
                      static_cast<unsigned long long>(sequence), api ? api : "?",
                      stage ? stage : "?", mode, first, count, type, indices,
                      g_active_texture_unit, bound_texture, info.width, info.height,
                      info.internalformat, info.format, info.bytes,
                      static_cast<unsigned long long>(info.sequence), info.alpha_upload ? 1 : 0,
                      info.icon_clamped ? 1 : 0, framebuffer, program, viewport[0], viewport[1],
                      viewport[2], viewport[3], blend == GL_TRUE ? 1 : 0, result_error);
}

void RememberAlphaTextureUpload(GLenum target, GLsizei texture_width, GLsizei texture_height,
                                GLint xoffset, GLint yoffset, GLsizei upload_width,
                                GLsizei upload_height, const GLubyte* alpha) {
    if (target != GL_TEXTURE_2D || alpha == nullptr || upload_width <= 0 || upload_height <= 0) {
        return;
    }
    const GLuint texture = CurrentBoundTexture2D();
    if (texture == 0 || texture_width <= 0 || texture_height <= 0) {
        return;
    }

    auto& shadow = g_alpha_texture_shadow[texture];
    const std::size_t texture_pixels =
        static_cast<std::size_t>(texture_width) * static_cast<std::size_t>(texture_height);
    if (shadow.width != texture_width || shadow.height != texture_height ||
        shadow.alpha.size() != texture_pixels) {
        shadow.width = texture_width;
        shadow.height = texture_height;
        shadow.alpha.assign(texture_pixels, 0);
    }

    if (xoffset < 0 || yoffset < 0 || xoffset + upload_width > texture_width ||
        yoffset + upload_height > texture_height) {
        return;
    }

    for (GLsizei row = 0; row < upload_height; ++row) {
        const std::size_t src_offset = static_cast<std::size_t>(row) * upload_width;
        const std::size_t dst_offset =
            static_cast<std::size_t>(yoffset + row) * texture_width + xoffset;
        std::memcpy(shadow.alpha.data() + dst_offset, alpha + src_offset,
                    static_cast<std::size_t>(upload_width));
    }
    shadow.nonzero_alpha = CountNonZeroAlpha(shadow.alpha.data(), shadow.width, shadow.height);
    shadow.first_nonzero_alpha =
        FirstNonZeroAlpha(shadow.alpha.data(), shadow.width, shadow.height);
}

bool SourceContains(const std::string& source, const char* needle) {
    return needle != nullptr && source.find(needle) != std::string::npos;
}

ProgramShaderInfo AnalyzeProgramShaders(GLuint program) {
    ProgramShaderInfo info{};
    const auto attached = g_program_attached_shaders.find(program);
    if (attached == g_program_attached_shaders.end()) {
        return info;
    }
    for (const GLuint shader : attached->second) {
        const auto type = g_shader_types.find(shader);
        if (type == g_shader_types.end() || type->second != GL_FRAGMENT_SHADER) {
            continue;
        }
        const auto source = g_shader_sources.find(shader);
        if (source == g_shader_sources.end()) {
            continue;
        }
        info.has_fragment_source = true;
        info.samples_texture =
            info.samples_texture || SourceContains(source->second, "texture2D") ||
            SourceContains(source->second, "texture(");
        info.reads_alpha = info.reads_alpha || SourceContains(source->second, ".a") ||
                           SourceContains(source->second, "alpha");
        info.reads_red = info.reads_red || SourceContains(source->second, ".r");
        info.mentions_color =
            info.mentions_color || SourceContains(source->second, "color") ||
            SourceContains(source->second, "Color") || SourceContains(source->second, "v_color");
    }
    return info;
}

const AlphaTextureShadow* FindBoundAlphaTexture(GLuint* out_texture, GLuint* out_unit) {
    for (GLuint unit = 0; unit < g_bound_texture_2d.size(); ++unit) {
        const GLuint texture = g_bound_texture_2d[unit];
        if (texture == 0) {
            continue;
        }
        const auto shadow = g_alpha_texture_shadow.find(texture);
        if (shadow != g_alpha_texture_shadow.end() && shadow->second.width > 0 &&
            shadow->second.height > 0 && !shadow->second.alpha.empty()) {
            if (out_texture != nullptr) {
                *out_texture = texture;
            }
            if (out_unit != nullptr) {
                *out_unit = unit;
            }
            return &shadow->second;
        }
    }
    return nullptr;
}

void LogFontAtlasDrawState(const char* api, GLenum mode, GLsizei count) {
    if (g_font_atlas_draw_log_budget <= 0) {
        return;
    }
    GLuint texture = 0;
    GLuint unit = 0;
    const AlphaTextureShadow* shadow = FindBoundAlphaTexture(&texture, &unit);
    if (shadow == nullptr) {
        return;
    }
    --g_font_atlas_draw_log_budget;

    GLint program = 0;
    GLint framebuffer = 0;
    GLint viewport[4] = {};
    GLint src_rgb = 0;
    GLint dst_rgb = 0;
    GLint src_alpha = 0;
    GLint dst_alpha = 0;
    GLboolean color_mask[4] = {};
    const GLboolean blend = glIsEnabled(GL_BLEND);
    const GLboolean depth = glIsEnabled(GL_DEPTH_TEST);
    const GLboolean cull = glIsEnabled(GL_CULL_FACE);
    const GLboolean scissor = glIsEnabled(GL_SCISSOR_TEST);
    glGetIntegerv(GL_CURRENT_PROGRAM, &program);
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &framebuffer);
    glGetIntegerv(GL_VIEWPORT, viewport);
    glGetIntegerv(GL_BLEND_SRC_RGB, &src_rgb);
    glGetIntegerv(GL_BLEND_DST_RGB, &dst_rgb);
    glGetIntegerv(GL_BLEND_SRC_ALPHA, &src_alpha);
    glGetIntegerv(GL_BLEND_DST_ALPHA, &dst_alpha);
    glGetBooleanv(GL_COLOR_WRITEMASK, color_mask);

    const ProgramShaderInfo shader_info =
        program > 0 ? g_program_shader_info[static_cast<GLuint>(program)] : ProgramShaderInfo{};
    const GLint color_location =
        program > 0 ? glGetAttribLocation(static_cast<GLuint>(program), "color") : -1;
    GLint color_enabled = 0;
    GLint color_buffer = 0;
    GLint color_size = 0;
    GLint color_type = 0;
    GLint color_stride = 0;
    void* color_pointer = nullptr;
    if (color_location >= 0) {
        const GLuint index = static_cast<GLuint>(color_location);
        glGetVertexAttribiv(index, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &color_enabled);
        glGetVertexAttribiv(index, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, &color_buffer);
        glGetVertexAttribiv(index, GL_VERTEX_ATTRIB_ARRAY_SIZE, &color_size);
        glGetVertexAttribiv(index, GL_VERTEX_ATTRIB_ARRAY_TYPE, &color_type);
        glGetVertexAttribiv(index, GL_VERTEX_ATTRIB_ARRAY_STRIDE, &color_stride);
        glGetVertexAttribPointerv(index, GL_VERTEX_ATTRIB_ARRAY_POINTER, &color_pointer);
    }

    Log("[EXECUTOR_PIGLET_TEXTDRAW] api=%s mode=0x%x count=%d program=%d framebuffer=%d "
        "viewport=%d,%d %dx%d alphaTex=%u unit=%u atlas=%dx%d nonzeroA=%zu firstA=%zu "
        "bounds=%s idx%u=(%.2f,%.2f)-(%.2f,%.2f) blend=%d funcRgb=0x%x/0x%x "
        "funcA=0x%x/0x%x depth=%d cull=%d scissor=%d colorMask=%d%d%d%d "
        "shaderFrag=%d sample=%d readA=%d readR=%d colorRef=%d colorLoc=%d "
        "colorEnabled=%d colorBuffer=%d colorSize=%d colorType=0x%x colorStride=%d "
        "colorPtr=%p",
        api ? api : "draw", mode, count, program, framebuffer, viewport[0], viewport[1],
        viewport[2], viewport[3], texture, unit, shadow->width, shadow->height,
        shadow->nonzero_alpha, shadow->first_nonzero_alpha,
        g_last_attrib0_bounds.valid ? "valid" : "none", g_last_attrib0_bounds.index,
        g_last_attrib0_bounds.min_x, g_last_attrib0_bounds.min_y, g_last_attrib0_bounds.max_x,
        g_last_attrib0_bounds.max_y, blend ? 1 : 0, src_rgb, dst_rgb, src_alpha, dst_alpha,
        depth ? 1 : 0, cull ? 1 : 0, scissor ? 1 : 0, color_mask[0] ? 1 : 0,
        color_mask[1] ? 1 : 0, color_mask[2] ? 1 : 0, color_mask[3] ? 1 : 0,
        shader_info.has_fragment_source ? 1 : 0, shader_info.samples_texture ? 1 : 0,
        shader_info.reads_alpha ? 1 : 0, shader_info.reads_red ? 1 : 0,
        shader_info.mentions_color ? 1 : 0, color_location, color_enabled, color_buffer,
        color_size, color_type, color_stride, color_pointer);
}

struct TexcoordSampleStats {
    bool valid = false;
    GLuint texture = 0;
    GLsizei width = 0;
    GLsizei height = 0;
    GLfloat min_u = 0.0f;
    GLfloat max_u = 0.0f;
    GLfloat min_v = 0.0f;
    GLfloat max_v = 0.0f;
    std::size_t sampled = 0;
    std::size_t nonzero_alpha = 0;
    GLubyte min_alpha = 0;
    GLubyte max_alpha = 0;
};

TexcoordSampleStats SampleTexcoordsAgainstAlphaShadow(const std::vector<GLfloat>& texcoords) {
    TexcoordSampleStats stats{};
    if (texcoords.size() < 2) {
        return stats;
    }

    GLuint texture = 0;
    if (!g_bound_texture_2d.empty() && g_bound_texture_2d[0] != 0) {
        texture = g_bound_texture_2d[0];
    } else {
        texture = CurrentBoundTexture2D();
    }
    const auto shadow = g_alpha_texture_shadow.find(texture);
    if (texture == 0 || shadow == g_alpha_texture_shadow.end() || shadow->second.alpha.empty() ||
        shadow->second.width <= 0 || shadow->second.height <= 0) {
        return stats;
    }

    stats.valid = true;
    stats.texture = texture;
    stats.width = shadow->second.width;
    stats.height = shadow->second.height;
    stats.min_u = stats.max_u = texcoords[0];
    stats.min_v = stats.max_v = texcoords[1];
    stats.min_alpha = 255;
    stats.max_alpha = 0;

    const std::size_t vertex_count = texcoords.size() / 2;
    const std::size_t sample_count = std::min<std::size_t>(vertex_count, 256);
    for (std::size_t i = 0; i < sample_count; ++i) {
        const GLfloat u = std::clamp(texcoords[i * 2 + 0], 0.0f, 1.0f);
        const GLfloat v = std::clamp(texcoords[i * 2 + 1], 0.0f, 1.0f);
        stats.min_u = std::min(stats.min_u, u);
        stats.max_u = std::max(stats.max_u, u);
        stats.min_v = std::min(stats.min_v, v);
        stats.max_v = std::max(stats.max_v, v);

        const auto x =
            std::min<GLsizei>(shadow->second.width - 1,
                              static_cast<GLsizei>(u * (shadow->second.width - 1)));
        const auto y =
            std::min<GLsizei>(shadow->second.height - 1,
                              static_cast<GLsizei>(v * (shadow->second.height - 1)));
        const auto alpha =
            shadow->second.alpha[static_cast<std::size_t>(y) * shadow->second.width + x];
        stats.min_alpha = std::min(stats.min_alpha, alpha);
        stats.max_alpha = std::max(stats.max_alpha, alpha);
        if (alpha != 0) {
            ++stats.nonzero_alpha;
        }
        ++stats.sampled;
    }
    return stats;
}

void ProbeReadbackAfterDraw(const char* label) {
    if (g_readback_probe_log_budget-- <= 0 || !g_last_attrib0_bounds.valid) {
        return;
    }

    GLint viewport[4] = {};
    glGetIntegerv(GL_VIEWPORT, viewport);
    if (viewport[2] <= 0 || viewport[3] <= 0) {
        return;
    }

    const GLfloat center_x = (g_last_attrib0_bounds.min_x + g_last_attrib0_bounds.max_x) * 0.5f;
    const GLfloat center_y = (g_last_attrib0_bounds.min_y + g_last_attrib0_bounds.max_y) * 0.5f;
    const GLint x =
        std::clamp(viewport[0] + static_cast<GLint>((center_x + 1.0f) * 0.5f * viewport[2]),
                   viewport[0], viewport[0] + viewport[2] - 1);
    const GLint y =
        std::clamp(viewport[1] + static_cast<GLint>((center_y + 1.0f) * 0.5f * viewport[3]),
                   viewport[1], viewport[1] + viewport[3] - 1);
    std::array<GLubyte, 4> pixel = {};
    glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel.data());
    const GLenum error = glGetError();
    Log("[EXECUTOR_PIGLET] %s readback attrib=%u center=(%.3f,%.3f) pixel=%d,%d "
        "rgba=%02x%02x%02x%02x bounds=(%.3f,%.3f)-(%.3f,%.3f) viewport=%d,%d %dx%d error=0x%x",
        label, g_last_attrib0_bounds.index, center_x, center_y, x, y, pixel[0], pixel[1],
        pixel[2], pixel[3], g_last_attrib0_bounds.min_x, g_last_attrib0_bounds.min_y,
        g_last_attrib0_bounds.max_x, g_last_attrib0_bounds.max_y, viewport[0], viewport[1],
        viewport[2], viewport[3], error);

    const bool looks_like_pixel_space =
        g_last_attrib0_bounds.max_x > 2.0f || g_last_attrib0_bounds.max_y > 2.0f ||
        g_last_attrib0_bounds.min_x < -2.0f || g_last_attrib0_bounds.min_y < -2.0f;
    if (!looks_like_pixel_space || g_pixel_space_readback_log_budget-- <= 0) {
        return;
    }

    const GLint pixel_x =
        std::clamp(viewport[0] + static_cast<GLint>(std::round(center_x)), viewport[0],
                   viewport[0] + viewport[2] - 1);
    const GLint top_left_y =
        std::clamp(static_cast<GLint>(std::round(center_y)), 0, viewport[3] - 1);
    const GLint pixel_y = viewport[1] + viewport[3] - 1 - top_left_y;
    std::array<GLubyte, 4> pixel_space = {};
    glReadPixels(pixel_x, pixel_y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel_space.data());
    const GLenum pixel_error = glGetError();
    Log("[EXECUTOR_PIGLET] %s pixel-space readback attrib=%u center=(%.3f,%.3f) "
        "pixel=%d,%d rgba=%02x%02x%02x%02x bounds=(%.3f,%.3f)-(%.3f,%.3f) "
        "viewport=%d,%d %dx%d error=0x%x",
        label, g_last_attrib0_bounds.index, center_x, center_y, pixel_x, pixel_y,
        pixel_space[0], pixel_space[1], pixel_space[2], pixel_space[3],
        g_last_attrib0_bounds.min_x, g_last_attrib0_bounds.min_y,
        g_last_attrib0_bounds.max_x, g_last_attrib0_bounds.max_y, viewport[0], viewport[1],
        viewport[2], viewport[3], pixel_error);
}

GLsizei MaxVertexCountFromElements(GLsizei count, GLenum type, const void* indices) {
    if (count <= 0) {
        return 0;
    }

    GLint element_array_buffer = 0;
    glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &element_array_buffer);
    if (element_array_buffer != 0 || indices == nullptr || !LooksLikeClientPointer(indices)) {
        return count;
    }

    std::uint32_t max_index = 0;
    switch (type) {
    case GL_UNSIGNED_BYTE: {
        const auto* values = static_cast<const GLubyte*>(indices);
        for (GLsizei i = 0; i < count; ++i) {
            max_index = std::max(max_index, static_cast<std::uint32_t>(values[i]));
        }
        break;
    }
    case GL_UNSIGNED_SHORT: {
        const auto* values = static_cast<const GLushort*>(indices);
        for (GLsizei i = 0; i < count; ++i) {
            max_index = std::max(max_index, static_cast<std::uint32_t>(values[i]));
        }
        break;
    }
    case GL_UNSIGNED_INT: {
        const auto* values = static_cast<const GLuint*>(indices);
        for (GLsizei i = 0; i < count; ++i) {
            max_index = std::max(max_index, static_cast<std::uint32_t>(values[i]));
        }
        break;
    }
    default:
        return count;
    }

    if (max_index > 65535) {
        return count;
    }
    return static_cast<GLsizei>(max_index + 1);
}

void ShadowBufferUpload(GLenum target, GLsizeiptr size, const void* data) {
    if (size < 0) {
        return;
    }

    GLint binding = 0;
    if (target == GL_ARRAY_BUFFER) {
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &binding);
    } else if (target == GL_ELEMENT_ARRAY_BUFFER) {
        glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &binding);
    } else {
        return;
    }
    if (binding <= 0) {
        return;
    }

    auto& shadow = target == GL_ARRAY_BUFFER
                       ? g_array_buffer_shadow[static_cast<GLuint>(binding)]
                       : g_element_buffer_shadow[static_cast<GLuint>(binding)];
    if (data != nullptr && size > 0 && size <= 8 * 1024 * 1024) {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        shadow.assign(bytes, bytes + size);
    } else {
        shadow.clear();
    }

    if (g_buffer_shadow_log_budget-- > 0) {
        Log("[EXECUTOR_PIGLET] shadow buffer upload target=0x%x buffer=%d size=%lld data=%p",
            target, binding, static_cast<long long>(size), data);
    }
}

void ShadowBufferSubUpload(GLenum target, GLintptr offset, GLsizeiptr size, const void* data) {
    if (offset < 0 || size < 0 || data == nullptr) {
        return;
    }

    GLint binding = 0;
    if (target == GL_ARRAY_BUFFER) {
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &binding);
    } else if (target == GL_ELEMENT_ARRAY_BUFFER) {
        glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &binding);
    } else {
        return;
    }
    if (binding <= 0) {
        return;
    }

    auto& shadow = target == GL_ARRAY_BUFFER
                       ? g_array_buffer_shadow[static_cast<GLuint>(binding)]
                       : g_element_buffer_shadow[static_cast<GLuint>(binding)];
    const auto required = static_cast<std::size_t>(offset + size);
    if (required > 8 * 1024 * 1024) {
        shadow.clear();
        return;
    }
    if (shadow.size() < required) {
        shadow.resize(required);
    }
    std::memcpy(shadow.data() + offset, data, static_cast<std::size_t>(size));

    if (g_buffer_shadow_log_budget-- > 0) {
        Log("[EXECUTOR_PIGLET] shadow buffer subupload target=0x%x buffer=%d offset=%lld "
            "size=%lld data=%p",
            target, binding, static_cast<long long>(offset), static_cast<long long>(size), data);
    }
}

bool ReadIndexFromShadow(const std::vector<std::uint8_t>& shadow, GLenum type, const void* indices,
                         GLsizei ordinal, std::uint32_t* out) {
    if (out == nullptr || ordinal < 0) {
        return false;
    }

    std::size_t size = 0;
    switch (type) {
    case GL_UNSIGNED_BYTE:
        size = 1;
        break;
    case GL_UNSIGNED_SHORT:
        size = 2;
        break;
    case GL_UNSIGNED_INT:
        size = 4;
        break;
    default:
        return false;
    }

    std::size_t offset = reinterpret_cast<std::uintptr_t>(indices) +
                         static_cast<std::size_t>(ordinal) * size;
    if (offset + size > shadow.size()) {
        return false;
    }
    if (type == GL_UNSIGNED_BYTE) {
        *out = shadow[offset];
    } else if (type == GL_UNSIGNED_SHORT) {
        std::uint16_t value = 0;
        std::memcpy(&value, shadow.data() + offset, sizeof(value));
        *out = value;
    } else {
        std::uint32_t value = 0;
        std::memcpy(&value, shadow.data() + offset, sizeof(value));
        *out = value;
    }
    return true;
}

void ProbeVboDrawElements(GLsizei count, GLenum type, const void* indices) {
    if (g_vbo_draw_probe_log_budget-- <= 0 || count <= 0) {
        return;
    }

    GLint program = 0;
    GLint element_buffer = 0;
    GLint attr_buffer = 0;
    GLint attr_enabled = 0;
    GLint attr_size = 0;
    GLint attr_type = 0;
    GLint attr_stride = 0;
    void* attr_pointer = nullptr;
    glGetIntegerv(GL_CURRENT_PROGRAM, &program);
    glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &element_buffer);
    glGetVertexAttribiv(1, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &attr_enabled);
    glGetVertexAttribiv(1, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, &attr_buffer);
    glGetVertexAttribiv(1, GL_VERTEX_ATTRIB_ARRAY_SIZE, &attr_size);
    glGetVertexAttribiv(1, GL_VERTEX_ATTRIB_ARRAY_TYPE, &attr_type);
    glGetVertexAttribiv(1, GL_VERTEX_ATTRIB_ARRAY_STRIDE, &attr_stride);
    glGetVertexAttribPointerv(1, GL_VERTEX_ATTRIB_ARRAY_POINTER, &attr_pointer);
    if (!attr_enabled || attr_buffer <= 0 || element_buffer <= 0 || attr_type != GL_FLOAT ||
        attr_size < 2) {
        Log("[EXECUTOR_PIGLET] vbo drawElements probe skipped program=%d attrEnabled=%d "
            "attrBuffer=%d elemBuffer=%d attrType=0x%x attrSize=%d",
            program, attr_enabled, attr_buffer, element_buffer, attr_type, attr_size);
        return;
    }

    const auto vertex_it = g_array_buffer_shadow.find(static_cast<GLuint>(attr_buffer));
    const auto index_it = g_element_buffer_shadow.find(static_cast<GLuint>(element_buffer));
    if (vertex_it == g_array_buffer_shadow.end() || index_it == g_element_buffer_shadow.end()) {
        Log("[EXECUTOR_PIGLET] vbo drawElements probe missing shadow program=%d attrBuffer=%d "
            "elemBuffer=%d",
            program, attr_buffer, element_buffer);
        return;
    }

    const auto& vertices = vertex_it->second;
    const auto& element_indices = index_it->second;
    const std::size_t type_size = GlTypeSize(static_cast<GLenum>(attr_type));
    const std::size_t element_size = static_cast<std::size_t>(attr_size) * type_size;
    const std::size_t stride =
        attr_stride > 0 ? static_cast<std::size_t>(attr_stride) : element_size;
    const std::size_t attr_offset = reinterpret_cast<std::uintptr_t>(attr_pointer);
    if (vertices.empty() || element_indices.empty() || stride == 0) {
        return;
    }

    GLfloat min_x = 0.0f;
    GLfloat min_y = 0.0f;
    GLfloat max_x = 0.0f;
    GLfloat max_y = 0.0f;
    std::uint32_t min_index = UINT32_MAX;
    std::uint32_t max_index = 0;
    int samples = 0;
    for (GLsizei i = 0; i < count; ++i) {
        std::uint32_t index = 0;
        if (!ReadIndexFromShadow(element_indices, type, indices, i, &index)) {
            break;
        }
        const std::size_t vertex_offset = attr_offset + static_cast<std::size_t>(index) * stride;
        if (vertex_offset + sizeof(GLfloat) * 2 > vertices.size()) {
            continue;
        }
        GLfloat x = 0.0f;
        GLfloat y = 0.0f;
        std::memcpy(&x, vertices.data() + vertex_offset, sizeof(x));
        std::memcpy(&y, vertices.data() + vertex_offset + sizeof(GLfloat), sizeof(y));
        if (samples == 0) {
            min_x = max_x = x;
            min_y = max_y = y;
        } else {
            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
            max_x = std::max(max_x, x);
            max_y = std::max(max_y, y);
        }
        min_index = std::min(min_index, index);
        max_index = std::max(max_index, index);
        ++samples;
    }

    if (samples <= 0) {
        return;
    }

    g_last_attrib0_bounds = {.valid = true,
                             .index = 1,
                             .min_x = min_x,
                             .min_y = min_y,
                             .max_x = max_x,
                             .max_y = max_y};
    Log("[EXECUTOR_PIGLET] vbo drawElements probe program=%d count=%d samples=%d "
        "indexRange=%u-%u attr1Buffer=%d elemBuffer=%d stride=%d offset=%p "
        "bounds=(%.1f,%.1f)-(%.1f,%.1f)",
        program, count, samples, min_index, max_index, attr_buffer, element_buffer, attr_stride,
        attr_pointer, min_x, min_y, max_x, max_y);
}

void PrepareClientAttribsForDraw(GLsizei vertex_count) {
    if (vertex_count <= 0) {
        return;
    }

    GLint previous_array_buffer = 0;
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &previous_array_buffer);
    GLuint last_uploaded_vbo = 0;
    bool uploaded_any = false;

    for (GLuint index = 0; index < g_client_attribs.size(); ++index) {
        const auto& attrib = g_client_attribs[index];
        if (!attrib.active || attrib.pointer == nullptr || !LooksLikeClientPointer(attrib.pointer)) {
            continue;
        }

        const std::size_t type_size = GlTypeSize(attrib.type);
        if (type_size == 0 || attrib.size <= 0 || attrib.size > 4) {
            continue;
        }

        const std::size_t element_size = static_cast<std::size_t>(attrib.size) * type_size;
        const bool itemz_packed_vertex = LooksLikeItemzPackedVertex(attrib, vertex_count);
        const std::size_t stride =
            EffectiveClientAttribStride(attrib, vertex_count, element_size);
        const std::size_t byte_count =
            stride * static_cast<std::size_t>(vertex_count - 1) + element_size;
        if (byte_count == 0 || byte_count > 4 * 1024 * 1024) {
            continue;
        }
        if (index == 0 || index == 1) {
            RememberPositionAttribBounds(index, attrib, vertex_count, stride);
        }
        std::vector<GLfloat> sanitized_positions;
        std::size_t upload_byte_count = byte_count;
        GLsizei upload_stride = static_cast<GLsizei>(stride);
        const void* upload_pointer = attrib.pointer;
        std::size_t sanitized_invalid = 0;
        if (itemz_packed_vertex && attrib.type == GL_FLOAT && attrib.size == 3 && stride >= 9U * sizeof(GLfloat)) {
            sanitized_positions.assign(static_cast<std::size_t>(vertex_count) * 3U, 0.0f);
            const auto* base = static_cast<const std::uint8_t*>(attrib.pointer);
            GLfloat last_x = 0.0f;
            GLfloat last_y = 0.0f;
            GLfloat last_z = 0.0f;
            bool have_last = false;
            GLfloat min_x = 0.0f;
            GLfloat min_y = 0.0f;
            GLfloat max_x = 0.0f;
            GLfloat max_y = 0.0f;
            for (GLsizei i = 0; i < vertex_count; ++i) {
                GLfloat x = 0.0f;
                GLfloat y = 0.0f;
                GLfloat z = 0.0f;
                std::memcpy(&x, base + static_cast<std::size_t>(i) * stride, sizeof(x));
                std::memcpy(&y, base + static_cast<std::size_t>(i) * stride + sizeof(GLfloat),
                            sizeof(y));
                std::memcpy(&z, base + static_cast<std::size_t>(i) * stride + 2U * sizeof(GLfloat),
                            sizeof(z));
                const bool valid_position = IsFiniteFloat(x) && IsFiniteFloat(y) &&
                                            IsFiniteFloat(z) && std::abs(x) <= 8192.0f &&
                                            std::abs(y) <= 8192.0f && std::abs(z) <= 8192.0f;
                if (!valid_position) {
                    ++sanitized_invalid;
                    if (have_last) {
                        x = last_x;
                        y = last_y;
                        z = last_z;
                    } else {
                        x = y = z = 0.0f;
                    }
                } else {
                    last_x = x;
                    last_y = y;
                    last_z = z;
                    have_last = true;
                }
                sanitized_positions[static_cast<std::size_t>(i) * 3U + 0U] = x;
                sanitized_positions[static_cast<std::size_t>(i) * 3U + 1U] = y;
                sanitized_positions[static_cast<std::size_t>(i) * 3U + 2U] = z;
                if (i == 0) {
                    min_x = max_x = x;
                    min_y = max_y = y;
                } else {
                    min_x = std::min(min_x, x);
                    min_y = std::min(min_y, y);
                    max_x = std::max(max_x, x);
                    max_y = std::max(max_y, y);
                }
            }
            upload_pointer = sanitized_positions.data();
            upload_byte_count = sanitized_positions.size() * sizeof(GLfloat);
            upload_stride = 3 * static_cast<GLsizei>(sizeof(GLfloat));
            if (index == 0 || index == 1) {
                g_last_attrib0_bounds = {.valid = true,
                                         .index = index,
                                         .min_x = min_x,
                                         .min_y = min_y,
                                         .max_x = max_x,
                                         .max_y = max_y};
            }
        }
        if (g_vertex_probe_log_budget-- > 0 && attrib.type == GL_FLOAT && attrib.size >= 2) {
            const auto* floats = static_cast<const GLfloat*>(attrib.pointer);
            const GLfloat x0 = floats[0];
            const GLfloat y0 = floats[1];
            const GLfloat x1 = vertex_count > 1 ? floats[stride / sizeof(GLfloat)] : 0.0f;
            const GLfloat y1 = vertex_count > 1 ? floats[stride / sizeof(GLfloat) + 1] : 0.0f;
            Log("[EXECUTOR_PIGLET] client attrib probe index=%u count=%d first=(%.3f,%.3f) "
                "second=(%.3f,%.3f) byteCount=%zu guestStride=%d effectiveStride=%zu",
                index, vertex_count, x0, y0, x1, y1, byte_count, attrib.stride, stride);
        }

        if (g_client_attrib_vbos[index] == 0) {
            glGenBuffers(1, &g_client_attrib_vbos[index]);
        }
        glBindBuffer(GL_ARRAY_BUFFER, g_client_attrib_vbos[index]);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(upload_byte_count), upload_pointer,
                     GL_STREAM_DRAW);
        LogGlErrorAfter("client attrib glBufferData");
        if (itemz_packed_vertex && g_client_attrib_upload_log_budget > 0) {
            Log("[EXECUTOR_PIGLET] itemz packed vertex stride override index=%u "
                "guestStride=%d effectiveStride=%zu uploadStride=%d sanitizedInvalid=%zu",
                index, attrib.stride, stride, upload_stride, sanitized_invalid);
        }
        glVertexAttribPointer(index, attrib.size, attrib.type, attrib.normalized,
                              upload_stride, nullptr);
        LogGlErrorAfter("client attrib glVertexAttribPointer");
        glEnableVertexAttribArray(index);
        LogGlErrorAfter("client attrib glEnableVertexAttribArray");
        last_uploaded_vbo = g_client_attrib_vbos[index];
        uploaded_any = true;
        if (g_client_attrib_upload_log_budget-- > 0) {
            Log("[EXECUTOR_PIGLET] uploaded client attrib index=%u pointer=%p bytes=%zu count=%d "
                "guestStride=%d effectiveStride=%zu uploadBytes=%zu uploadStride=%d vbo=%u",
                index, attrib.pointer, byte_count, vertex_count, attrib.stride, stride,
                upload_byte_count, upload_stride,
                g_client_attrib_vbos[index]);
        }
    }

    (void)last_uploaded_vbo;
    glBindBuffer(GL_ARRAY_BUFFER, static_cast<GLuint>(previous_array_buffer));
}

bool CurrentProgramUsesTextureSampler() {
    GLint program = 0;
    glGetIntegerv(GL_CURRENT_PROGRAM, &program);
    if (program <= 0) {
        return false;
    }

    static constexpr const char* kSamplerNames[] = {
        "texture",
        "tex",
        "Texture",
        "u_TextureUnit",
    };
    for (const char* name : kSamplerNames) {
        if (glGetUniformLocation(static_cast<GLuint>(program), name) >= 0) {
            return true;
        }
    }
    return false;
}

void PrepareSyntheticTextureAttribsForDraw(GLenum mode, GLsizei vertex_count) {
    if (vertex_count <= 0 || !CurrentProgramUsesTextureSampler()) {
        return;
    }
    if (vertex_count > 65536) {
        return;
    }

    GLint program = 0;
    glGetIntegerv(GL_CURRENT_PROGRAM, &program);

    GLint color_location = glGetAttribLocation(static_cast<GLuint>(program), "color");
    if (color_location < 0) {
        color_location = 4;
    }
    if (color_location >= 0 && color_location < 16) {
        GLint color_enabled = 0;
        glGetVertexAttribiv(static_cast<GLuint>(color_location), GL_VERTEX_ATTRIB_ARRAY_ENABLED,
                            &color_enabled);
        if (!color_enabled) {
            glDisableVertexAttribArray(static_cast<GLuint>(color_location));
            glVertexAttrib4f(static_cast<GLuint>(color_location), 1.0f, 1.0f, 1.0f, 1.0f);
            if (g_synthetic_attrib_log_budget-- > 0) {
                Log("[EXECUTOR_PIGLET] textured draw defaulted color attrib program=%d "
                    "location=%d value=1,1,1,1",
                    program, color_location);
            }
        }
    }

    static constexpr const char* kTexcoordNames[] = {
        "tex_coord",
        "a_TextureCoordinates",
        "a_TexCoord",
        "texCoord",
        "TexCoord",
        "uv",
    };
    GLint tex_location = -1;
    for (const char* name : kTexcoordNames) {
        tex_location = glGetAttribLocation(static_cast<GLuint>(program), name);
        if (tex_location >= 0) {
            break;
        }
    }
    if (tex_location < 0 || tex_location >= 16) {
        return;
    }

    auto use_guest_texcoord_attrib = [&](GLint candidate, const char* source) -> bool {
        if (candidate < 0 || candidate >= 16) {
            return false;
        }
        const auto state = QueryVertexAttribState(static_cast<GLuint>(candidate));
        if (!state.enabled || state.type != GL_FLOAT || state.size < 2) {
            return false;
        }
        if (candidate == tex_location) {
            if (ApplyItemzflowTextTexcoordRepair(program, tex_location, vertex_count, state)) {
                return true;
            }
            if (g_synthetic_attrib_log_budget-- > 0) {
                Log("[EXECUTOR_PIGLET] textured draw using guest texcoord attrib program=%d "
                    "location=%d source=%s buffer=%d stride=%d pointer=%p",
                    program, tex_location, source, state.buffer, state.stride, state.pointer);
            }
            return true;
        }
        if (state.buffer <= 0) {
            return false;
        }
        GLint previous_array_buffer = 0;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &previous_array_buffer);
        glBindBuffer(GL_ARRAY_BUFFER, static_cast<GLuint>(state.buffer));
        glVertexAttribPointer(static_cast<GLuint>(tex_location), 2, GL_FLOAT,
                              static_cast<GLboolean>(state.normalized), state.stride,
                              state.pointer);
        glEnableVertexAttribArray(static_cast<GLuint>(tex_location));
        glBindBuffer(GL_ARRAY_BUFFER, static_cast<GLuint>(previous_array_buffer));
        LogGlErrorAfter("guest texcoord attr alias");
        if (g_synthetic_attrib_log_budget-- > 0) {
            Log("[EXECUTOR_PIGLET] textured draw aliased guest texcoord attrib program=%d "
                "sourceLocation=%d targetLocation=%d source=%s buffer=%d stride=%d pointer=%p",
                program, candidate, tex_location, source, state.buffer, state.stride,
                state.pointer);
        }
        return true;
    };

    std::array<GLint, 5> texcoord_candidates = {
        tex_location,
        glGetAttribLocation(static_cast<GLuint>(program), "a_TextureCoordinates"),
        glGetAttribLocation(static_cast<GLuint>(program), "tex_coord"),
        2,
        3,
    };
    for (GLint candidate : texcoord_candidates) {
        if (use_guest_texcoord_attrib(candidate, "guest-provider")) {
            return;
        }
    }

    std::vector<GLfloat> texcoords;
    texcoords.reserve(static_cast<std::size_t>(vertex_count) * 2);
    std::vector<GLfloat> colors;
    const char* texcoord_source = "fallback-pattern";
    const char* color_source = "default";
    GLfloat color_alpha_min = 1.0f;
    GLfloat color_alpha_max = 0.0f;
    GLfloat color_rgb_max = 0.0f;
    std::size_t color_alpha_nonzero = 0;
    std::size_t color_vertex_count = 0;
    bool color_alpha_guard = false;
    bool color_rgb_guard = false;
    GLint position_source_location = -1;
    GLuint position_source_buffer = 0;

    auto build_from_float_positions = [&](const std::uint8_t* base, std::size_t stride,
                                          const char* source, GLint location,
                                          GLuint buffer) -> bool {
        if (base == nullptr || stride == 0) {
            return false;
        }

        GLfloat min_x = 0.0f;
        GLfloat max_x = 0.0f;
        GLfloat min_y = 0.0f;
        GLfloat max_y = 0.0f;
        for (GLsizei i = 0; i < vertex_count; ++i) {
            GLfloat x = 0.0f;
            GLfloat y = 0.0f;
            std::memcpy(&x, base + static_cast<std::size_t>(i) * stride, sizeof(x));
            std::memcpy(&y, base + static_cast<std::size_t>(i) * stride + sizeof(GLfloat),
                        sizeof(y));
            if (i == 0) {
                min_x = max_x = x;
                min_y = max_y = y;
            } else {
                min_x = std::min(min_x, x);
                max_x = std::max(max_x, x);
                min_y = std::min(min_y, y);
                max_y = std::max(max_y, y);
            }
        }

        const GLfloat span_x = std::max<GLfloat>(max_x - min_x, 0.001f);
        const GLfloat span_y = std::max<GLfloat>(max_y - min_y, 0.001f);
        texcoords.assign(static_cast<std::size_t>(vertex_count) * 2, 0.0f);
        for (GLsizei i = 0; i < vertex_count; ++i) {
            GLfloat x = 0.0f;
            GLfloat y = 0.0f;
            std::memcpy(&x, base + static_cast<std::size_t>(i) * stride, sizeof(x));
            std::memcpy(&y, base + static_cast<std::size_t>(i) * stride + sizeof(GLfloat),
                        sizeof(y));
            texcoords[static_cast<std::size_t>(i) * 2 + 0] =
                std::clamp((x - min_x) / span_x, 0.0f, 1.0f);
            texcoords[static_cast<std::size_t>(i) * 2 + 1] =
                std::clamp((y - min_y) / span_y, 0.0f, 1.0f);
        }
        texcoord_source = source;
        position_source_location = location;
        position_source_buffer = buffer;
        return true;
    };

    const ClientAttribState* position_attrib = nullptr;
    if (g_client_attribs[1].active && g_client_attribs[1].pointer != nullptr &&
        g_client_attribs[1].type == GL_FLOAT && g_client_attribs[1].size >= 2) {
        position_attrib = &g_client_attribs[1];
    } else if (g_client_attribs[0].active && g_client_attribs[0].pointer != nullptr &&
               g_client_attribs[0].type == GL_FLOAT && g_client_attribs[0].size >= 2) {
        position_attrib = &g_client_attribs[0];
    }
    if (position_attrib != nullptr) {
        const std::size_t type_size = GlTypeSize(position_attrib->type);
        const std::size_t element_size =
            static_cast<std::size_t>(position_attrib->size) * type_size;
        const bool itemz_packed_vertex =
            LooksLikeItemzPackedVertex(*position_attrib, vertex_count);
        const std::size_t stride =
            EffectiveClientAttribStride(*position_attrib, vertex_count, element_size);
        if (stride != 0 && type_size == sizeof(GLfloat)) {
            const auto* base = static_cast<const std::uint8_t*>(position_attrib->pointer);
            if (itemz_packed_vertex) {
                texcoords.assign(static_cast<std::size_t>(vertex_count) * 2, 0.0f);
                colors.assign(static_cast<std::size_t>(vertex_count) * 4, 1.0f);
                for (GLsizei i = 0; i < vertex_count; ++i) {
                    const auto* uv_base = base + static_cast<std::size_t>(i) * stride +
                                          3U * sizeof(GLfloat);
                    GLfloat u = 0.0f;
                    GLfloat v = 0.0f;
                    std::memcpy(&u, uv_base, sizeof(u));
                    std::memcpy(&v, uv_base + sizeof(GLfloat), sizeof(v));
                    texcoords[static_cast<std::size_t>(i) * 2 + 0] =
                        std::clamp(u, 0.0f, 1.0f);
                    texcoords[static_cast<std::size_t>(i) * 2 + 1] =
                        std::clamp(v, 0.0f, 1.0f);

                    const auto* color_base = base + static_cast<std::size_t>(i) * stride +
                                             5U * sizeof(GLfloat);
                    for (int c = 0; c < 4; ++c) {
                        GLfloat component = 1.0f;
                        std::memcpy(&component, color_base + c * sizeof(GLfloat),
                                    sizeof(component));
                        colors[static_cast<std::size_t>(i) * 4 + c] =
                            NormalizePackedColorComponent(component);
                    }
                }
                texcoord_source = "itemz-packed-uv";
                color_source = "itemz-packed-color";
            } else {
                build_from_float_positions(base, stride, "client-position", -1, 0);
            }
        }
    }

    if (texcoords.empty()) {
        static constexpr const char* kPositionNames[] = {
            "vertex",
            "position",
            "a_Position",
            "aPos",
            "inPosition",
        };
        GLint queried_position = -1;
        const char* queried_position_name = "<none>";
        for (const char* name : kPositionNames) {
            queried_position = glGetAttribLocation(static_cast<GLuint>(program), name);
            if (queried_position >= 0) {
                queried_position_name = name;
                break;
            }
        }
        std::array<GLint, 4> candidates = {queried_position, 0, 1, 2};
        for (GLint candidate : candidates) {
            if (candidate < 0 || candidate >= 16 || candidate == tex_location) {
                continue;
            }
            GLint enabled = 0;
            GLint buffer = 0;
            GLint size = 0;
            GLint type = 0;
            GLint stride = 0;
            void* pointer = nullptr;
            glGetVertexAttribiv(static_cast<GLuint>(candidate), GL_VERTEX_ATTRIB_ARRAY_ENABLED,
                                &enabled);
            glGetVertexAttribiv(static_cast<GLuint>(candidate),
                                GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, &buffer);
            glGetVertexAttribiv(static_cast<GLuint>(candidate), GL_VERTEX_ATTRIB_ARRAY_SIZE,
                                &size);
            glGetVertexAttribiv(static_cast<GLuint>(candidate), GL_VERTEX_ATTRIB_ARRAY_TYPE,
                                &type);
            glGetVertexAttribiv(static_cast<GLuint>(candidate), GL_VERTEX_ATTRIB_ARRAY_STRIDE,
                                &stride);
            glGetVertexAttribPointerv(static_cast<GLuint>(candidate),
                                      GL_VERTEX_ATTRIB_ARRAY_POINTER, &pointer);
            if (!enabled || buffer <= 0 || type != GL_FLOAT || size < 2) {
                continue;
            }
            const auto shadow = g_array_buffer_shadow.find(static_cast<GLuint>(buffer));
            if (shadow == g_array_buffer_shadow.end() || shadow->second.empty()) {
                continue;
            }
            const std::size_t element_size =
                static_cast<std::size_t>(size) * sizeof(GLfloat);
            const std::size_t stride_bytes =
                stride > 0 ? static_cast<std::size_t>(stride) : element_size;
            const std::uintptr_t offset = reinterpret_cast<std::uintptr_t>(pointer);
            const std::size_t byte_count =
                stride_bytes * static_cast<std::size_t>(vertex_count - 1) +
                2 * sizeof(GLfloat);
            if (stride_bytes == 0 || offset > shadow->second.size() ||
                byte_count > shadow->second.size() - offset) {
                continue;
            }
            const auto* base = shadow->second.data() + offset;
            if (build_from_float_positions(base, stride_bytes, "vbo-position", candidate,
                                           static_cast<GLuint>(buffer))) {
                if (g_synthetic_attrib_log_budget-- > 0) {
                    Log("[EXECUTOR_PIGLET] textured draw position source program=%d "
                        "name=%s location=%d buffer=%d stride=%zu",
                        program, queried_position_name, candidate, buffer, stride_bytes);
                }
                break;
            }
        }
    }

    if (texcoords.empty()) {
        texcoords.assign(static_cast<std::size_t>(vertex_count) * 2, 0.0f);
        for (GLsizei i = 0; i < vertex_count; ++i) {
            GLfloat u = 0.0f;
            GLfloat v = 0.0f;
            if (mode == GL_TRIANGLE_STRIP) {
                switch (i % 4) {
                case 0:
                    u = 0.0f;
                    v = 0.0f;
                    break;
                case 1:
                    u = 1.0f;
                    v = 0.0f;
                    break;
                case 2:
                    u = 0.0f;
                    v = 1.0f;
                    break;
                default:
                    u = 1.0f;
                    v = 1.0f;
                    break;
                }
                texcoord_source = "triangle-strip-pattern";
            } else {
                switch (i % 6) {
                case 0:
                case 3:
                    u = 0.0f;
                    v = 0.0f;
                    break;
                case 1:
                    u = 1.0f;
                    v = 0.0f;
                    break;
                case 2:
                case 4:
                    u = 1.0f;
                    v = 1.0f;
                    break;
                default:
                    u = 0.0f;
                    v = 1.0f;
                    break;
                }
                texcoord_source = "triangle-pattern";
            }
            texcoords[static_cast<std::size_t>(i) * 2 + 0] = u;
            texcoords[static_cast<std::size_t>(i) * 2 + 1] = v;
        }
    }

    if (!colors.empty()) {
        color_vertex_count = colors.size() / 4;
        color_alpha_min = 1.0f;
        color_alpha_max = 0.0f;
        color_rgb_max = 0.0f;
        color_alpha_nonzero = 0;
        for (std::size_t i = 0; i < color_vertex_count; ++i) {
            const GLfloat r = colors[i * 4 + 0];
            const GLfloat g = colors[i * 4 + 1];
            const GLfloat b = colors[i * 4 + 2];
            const GLfloat a = colors[i * 4 + 3];
            color_rgb_max = std::max(color_rgb_max, std::max(r, std::max(g, b)));
            color_alpha_min = std::min(color_alpha_min, a);
            color_alpha_max = std::max(color_alpha_max, a);
            if (a > 0.001f) {
                ++color_alpha_nonzero;
            }
        }
        const bool itemz_packed_color =
            std::strcmp(color_source, "itemz-packed-color") == 0 && color_vertex_count > 0;
        color_alpha_guard = color_alpha_nonzero == 0 || color_alpha_max <= 0.001f ||
                            (itemz_packed_color && color_alpha_nonzero < color_vertex_count);
        color_rgb_guard = color_rgb_max <= 0.001f || itemz_packed_color;
        if (color_alpha_guard || color_rgb_guard) {
            for (std::size_t i = 0; i < color_vertex_count; ++i) {
                if (color_rgb_guard) {
                    colors[i * 4 + 0] = 1.0f;
                    colors[i * 4 + 1] = 1.0f;
                    colors[i * 4 + 2] = 1.0f;
                }
                if (color_alpha_guard) {
                    colors[i * 4 + 3] = 1.0f;
                }
            }
            color_alpha_min = color_alpha_guard ? 1.0f : color_alpha_min;
            color_alpha_max = color_alpha_guard ? 1.0f : color_alpha_max;
            color_alpha_nonzero = color_alpha_guard ? color_vertex_count : color_alpha_nonzero;
            color_rgb_max = color_rgb_guard ? 1.0f : color_rgb_max;
            color_source = color_alpha_guard && color_rgb_guard
                               ? "itemz-packed-color-rgba-guard"
                               : (color_alpha_guard ? "itemz-packed-color-alpha-guard"
                                                    : "itemz-packed-color-rgb-guard");
        }
    }

    const auto texcoord_alpha_stats = SampleTexcoordsAgainstAlphaShadow(texcoords);

    GLint previous_array_buffer = 0;
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &previous_array_buffer);
    if (g_synthetic_texcoord_vbo == 0) {
        glGenBuffers(1, &g_synthetic_texcoord_vbo);
    }
    glBindBuffer(GL_ARRAY_BUFFER, g_synthetic_texcoord_vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(texcoords.size() * sizeof(GLfloat)),
                 texcoords.data(), GL_STREAM_DRAW);
    glVertexAttribPointer(static_cast<GLuint>(tex_location), 2, GL_FLOAT, GL_FALSE,
                          2 * static_cast<GLsizei>(sizeof(GLfloat)), nullptr);
    glEnableVertexAttribArray(static_cast<GLuint>(tex_location));
    if (!colors.empty() && color_location >= 0 && color_location < 16) {
        if (g_synthetic_color_vbo == 0) {
            glGenBuffers(1, &g_synthetic_color_vbo);
        }
        glBindBuffer(GL_ARRAY_BUFFER, g_synthetic_color_vbo);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(colors.size() * sizeof(GLfloat)),
                     colors.data(), GL_STREAM_DRAW);
        glVertexAttribPointer(static_cast<GLuint>(color_location), 4, GL_FLOAT, GL_FALSE,
                              4 * static_cast<GLsizei>(sizeof(GLfloat)), nullptr);
        glEnableVertexAttribArray(static_cast<GLuint>(color_location));
    }
    glBindBuffer(GL_ARRAY_BUFFER, static_cast<GLuint>(previous_array_buffer));
    LogGlErrorAfter("synthetic texture attrib");
    if (g_synthetic_attrib_log_budget-- > 0) {
        Log("[EXECUTOR_PIGLET] textured draw synthesized texcoord attrib program=%d "
            "location=%d count=%d source=%s colorSource=%s colorLocation=%d positionAttrib=%d "
            "positionBuffer=%u vbo=%u colorVbo=%u colorVertices=%zu alpha=%.4f..%.4f "
            "alphaNonZero=%zu rgbMax=%.4f alphaGuard=%d rgbGuard=%d atlasTex=%u "
            "atlasSize=%dx%d uv=%.4f..%.4f,%.4f..%.4f atlasSamples=%zu "
            "atlasNonZeroA=%zu atlasA=%u..%u",
            program, tex_location, vertex_count, texcoord_source, color_source, color_location,
            position_source_location, position_source_buffer, g_synthetic_texcoord_vbo,
            g_synthetic_color_vbo, color_vertex_count, color_alpha_min, color_alpha_max,
            color_alpha_nonzero, color_rgb_max, color_alpha_guard ? 1 : 0,
            color_rgb_guard ? 1 : 0, texcoord_alpha_stats.texture, texcoord_alpha_stats.width,
            texcoord_alpha_stats.height, texcoord_alpha_stats.min_u, texcoord_alpha_stats.max_u,
            texcoord_alpha_stats.min_v, texcoord_alpha_stats.max_v, texcoord_alpha_stats.sampled,
            texcoord_alpha_stats.nonzero_alpha, texcoord_alpha_stats.min_alpha,
            texcoord_alpha_stats.max_alpha);
    }
}

void LogDrawErrorState(const char* label, GLenum mode, GLsizei count, GLenum error) {
    if (g_draw_error_log_budget-- <= 0) {
        return;
    }
    GLint program = 0;
    GLint array_buffer = 0;
    GLint element_buffer = 0;
    GLint framebuffer = 0;
    GLint viewport[4] = {};
    glGetIntegerv(GL_CURRENT_PROGRAM, &program);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &array_buffer);
    glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &element_buffer);
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &framebuffer);
    glGetIntegerv(GL_VIEWPORT, viewport);
    GLint enabled0 = 0;
    GLint enabled1 = 0;
    GLint enabled2 = 0;
    GLint enabled3 = 0;
    GLint buffer0 = 0;
    GLint buffer1 = 0;
    GLint buffer2 = 0;
    GLint buffer3 = 0;
    GLint size0 = 0;
    GLint type0 = 0;
    GLint stride0 = 0;
    void* pointer0 = nullptr;
    glGetVertexAttribiv(0, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &enabled0);
    glGetVertexAttribiv(1, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &enabled1);
    glGetVertexAttribiv(2, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &enabled2);
    glGetVertexAttribiv(3, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &enabled3);
    glGetVertexAttribiv(0, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, &buffer0);
    glGetVertexAttribiv(1, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, &buffer1);
    glGetVertexAttribiv(2, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, &buffer2);
    glGetVertexAttribiv(3, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, &buffer3);
    glGetVertexAttribiv(0, GL_VERTEX_ATTRIB_ARRAY_SIZE, &size0);
    glGetVertexAttribiv(0, GL_VERTEX_ATTRIB_ARRAY_TYPE, &type0);
    glGetVertexAttribiv(0, GL_VERTEX_ATTRIB_ARRAY_STRIDE, &stride0);
    glGetVertexAttribPointerv(0, GL_VERTEX_ATTRIB_ARRAY_POINTER, &pointer0);
    Log("[EXECUTOR_PIGLET] %s error=0x%x mode=0x%x count=%d program=%d array_buffer=%d "
        "element_buffer=%d framebuffer=%d viewport=%d,%d %dx%d attribEnabled=%d,%d,%d,%d "
        "attribBuffer=%d,%d,%d,%d attrib0=size:%d type:0x%x stride:%d pointer:%p",
        label, error, mode, count, program, array_buffer, element_buffer, framebuffer,
        viewport[0], viewport[1], viewport[2], viewport[3], enabled0, enabled1, enabled2,
        enabled3, buffer0, buffer1, buffer2, buffer3, size0, type0, stride0, pointer0);
}

bool EnsureDisplay() {
    if (g_display != EGL_NO_DISPLAY && g_initialized) {
        return true;
    }

    g_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (g_display == EGL_NO_DISPLAY) {
        Log("[EXECUTOR_PIGLET] eglGetDisplay failed error=0x%x", eglGetError());
        return false;
    }

    EGLint major = 0;
    EGLint minor = 0;
    if (!eglInitialize(g_display, &major, &minor)) {
        Log("[EXECUTOR_PIGLET] eglInitialize failed error=0x%x", eglGetError());
        g_display = EGL_NO_DISPLAY;
        return false;
    }

    const EGLint config_attribs[] = {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 16,
        EGL_STENCIL_SIZE, 8,
        EGL_NONE,
    };
    EGLint count = 0;
    if (!eglChooseConfig(g_display, config_attribs, &g_config, 1, &count) || count <= 0) {
        Log("[EXECUTOR_PIGLET] eglChooseConfig failed count=%d error=0x%x", count, eglGetError());
        return false;
    }

    eglBindAPI(EGL_OPENGL_ES_API);
    g_initialized = true;
    Log("[EXECUTOR_PIGLET] EGL initialized %d.%d", major, minor);
    return true;
}

bool EnsureSurface() {
    if (!EnsureDisplay()) {
        return false;
    }

    void* window = executor_lsx4_runtime_get_native_window();
    if (window == nullptr) {
        Log("[EXECUTOR_PIGLET] no ANativeWindow attached");
        return false;
    }

    if (g_surface != EGL_NO_SURFACE && g_surface_window == window) {
        return true;
    }

    if (g_surface != EGL_NO_SURFACE) {
        Log("[EXECUTOR_PIGLET] recreating EGL window surface oldWindow=%p newWindow=%p",
            g_surface_window, window);
        if (eglGetCurrentSurface(EGL_DRAW) == g_surface ||
            eglGetCurrentSurface(EGL_READ) == g_surface) {
            eglMakeCurrent(g_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        }
        eglDestroySurface(g_display, g_surface);
        g_surface = EGL_NO_SURFACE;
        g_surface_window = nullptr;
        g_surface_width = 0;
        g_surface_height = 0;
    }

    const EGLint surface_attribs[] = {EGL_NONE};
    g_surface = eglCreateWindowSurface(g_display, g_config, static_cast<EGLNativeWindowType>(window),
                                       surface_attribs);
    if (g_surface == EGL_NO_SURFACE) {
        Log("[EXECUTOR_PIGLET] eglCreateWindowSurface failed error=0x%x", eglGetError());
        return false;
    }
    g_surface_window = window;
    Log("[EXECUTOR_PIGLET] EGL window surface ready");
    UpdateSurfaceSize("eglCreateWindowSurface");
    return true;
}

std::atomic<bool> g_main_gl_thread_claimed{false};

bool IsMainGlThread() {
    thread_local int role = 0;
    if (role == 0) {
        bool expected = false;
        role = g_main_gl_thread_claimed.compare_exchange_strong(expected, true) ? 1 : 2;
    }
    return role == 1;
}

bool EnsureContext() {
    if (!EnsureSurface()) {
        return false;
    }
    if (g_context == EGL_NO_CONTEXT) {
        const EGLint context_attribs[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
        g_context = eglCreateContext(g_display, g_config, EGL_NO_CONTEXT, context_attribs);
        if (g_context == EGL_NO_CONTEXT) {
            Log("[EXECUTOR_PIGLET] eglCreateContext ES2 failed error=0x%x", eglGetError());
            return false;
        }
        Log("[EXECUTOR_PIGLET] EGL ES2 context ready");
    }

    if (!IsMainGlThread()) {
        thread_local EGLContext t_context = EGL_NO_CONTEXT;
        thread_local EGLSurface t_surface = EGL_NO_SURFACE;
        if (t_context == EGL_NO_CONTEXT) {
            const EGLint ctx_attribs[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
            t_context = eglCreateContext(g_display, g_config, g_context, ctx_attribs);
            if (t_context == EGL_NO_CONTEXT) {
                Log("[EXECUTOR_PIGLET] worker shared eglCreateContext failed error=0x%x",
                    eglGetError());
                return false;
            }
            const EGLint pb_attribs[] = {EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE};
            t_surface = eglCreatePbufferSurface(g_display, g_config, pb_attribs);
            Log("[EXECUTOR_PIGLET] worker shared GL context=%p pbuffer=%p",
                static_cast<void*>(t_context), static_cast<void*>(t_surface));
        }
        if (eglGetCurrentContext() != t_context) {
            const EGLSurface s = t_surface;
            if (!eglMakeCurrent(g_display, s, s, t_context)) {
                Log("[EXECUTOR_PIGLET] worker eglMakeCurrent failed error=0x%x", eglGetError());
                return false;
            }
        }
        return true;
    }

    const EGLContext current_context = eglGetCurrentContext();
    const EGLSurface current_draw = eglGetCurrentSurface(EGL_DRAW);
    const EGLSurface current_read = eglGetCurrentSurface(EGL_READ);
    if (current_context != g_context || current_draw != g_surface || current_read != g_surface) {
        if (!eglMakeCurrent(g_display, g_surface, g_surface, g_context)) {
            Log("[EXECUTOR_PIGLET] eglMakeCurrent failed error=0x%x", eglGetError());
            return false;
        }
        static int make_current_log_budget = 8;
        if (make_current_log_budget-- > 0) {
            Log("[EXECUTOR_PIGLET] eglMakeCurrent bound context=%p draw=%p read=%p",
                static_cast<void*>(g_context), static_cast<void*>(g_surface),
                static_cast<void*>(g_surface));
        }
    }
    return true;
}

void ReuploadFontAtlasIfChanged() {
    FontAtlasWatch& w = g_font_atlas_watch;
    if (w.texture == 0 || w.buffer == nullptr || w.width <= 0 || w.height <= 0) {
        return;
    }
    if (--w.frame_throttle > 0) {
        return;
    }
    w.frame_throttle = 8;
    const std::size_t nz = CountNonZeroAlpha(w.buffer, w.width, w.height);
    if (nz <= w.last_nonzero + 32) {
        return;
    }
    w.last_nonzero = nz;
    auto rgba = AlphaToRgba(w.buffer, w.width, w.height);
    if (rgba.empty()) {
        return;
    }
    GLint prev = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prev);
    glBindTexture(GL_TEXTURE_2D, w.texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w.width, w.height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                 rgba.data());
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prev));
    static int reupload_log_budget = 16;
    if (reupload_log_budget-- > 0) {
        Log("[EXECUTOR_PIGLET] font-atlas RE-UPLOAD tex=%u %dx%d nonzeroAlpha=%zu (glyphs composed)",
            w.texture, w.width, w.height, nz);
    }
}

EGLBoolean SwapAndroidSurface(const char* reason, EGLSurface surface) {
    if (!EnsureContext()) {
        return EGL_FALSE;
    }
    const EGLSurface target = surface != EGL_NO_SURFACE ? surface : g_surface;
    GLint framebuffer = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &framebuffer);
    if (framebuffer == 0) {
        const char* disable_fallback =
            std::getenv("EXECUTOR_ITEMZFLOW_DISABLE_FALLBACK_OVERLAY");
        const bool fallback_disabled =
            disable_fallback != nullptr && disable_fallback[0] != '\0' &&
            std::strcmp(disable_fallback, "0") != 0;
        if (fallback_disabled && g_itemzflow_overlay_enabled && g_window_draw_count > 0 &&
            g_itemzflow_authentic_log_budget-- > 0) {
            Log("[EXECUTOR_PIGLET] itemzflow authentic surface swap reason=%s "
                "draws=%llu degenerate=%llu surface=%dx%d fallback=0",
                reason ? reason : "guest",
                static_cast<unsigned long long>(g_window_draw_count),
                static_cast<unsigned long long>(g_degenerate_window_draw_count),
                g_surface_width, g_surface_height);
        }
        if (fallback_disabled) {
            DrawItemzflowReadableMenuAssistIfNeeded(reason);
        }
        const char* force_fallback = std::getenv("EXECUTOR_ITEMZFLOW_FORCE_FALLBACK_OVERLAY");
        const bool fallback_forced =
            force_fallback != nullptr && force_fallback[0] != '\0' &&
            std::strcmp(force_fallback, "0") != 0;
        if (fallback_forced || !fallback_disabled) {
            DrawItemzflowFallbackOverlayIfNeeded(reason);
        }
    } else if (g_fbo_log_budget-- > 0) {
        Log("[EXECUTOR_PIGLET] eglSwapBuffers requested while framebuffer=%d reason=%s",
            framebuffer, reason ? reason : "guest");
    }
    ++g_swap_count;
    if (g_swap_count <= 8 || (g_swap_count % 60) == 0) {
        Log("[EXECUTOR_PIGLET] eglSwapBuffers frame=%llu reason=%s",
            static_cast<unsigned long long>(g_swap_count), reason ? reason : "guest");
    }
    glFlush();
    glFinish();
    const EGLBoolean swapped = eglSwapBuffers(g_display, target);
    if (!swapped) {
        const EGLint error = eglGetError();
        Log("[EXECUTOR_PIGLET] eglSwapBuffers failed frame=%llu reason=%s error=0x%x",
            static_cast<unsigned long long>(g_swap_count), reason ? reason : "guest", error);
        if (target == g_surface &&
            (error == EGL_BAD_SURFACE || error == EGL_BAD_NATIVE_WINDOW ||
             error == EGL_BAD_ALLOC)) {
            eglMakeCurrent(g_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            eglDestroySurface(g_display, g_surface);
            g_surface = EGL_NO_SURFACE;
            g_surface_window = nullptr;
            g_surface_width = 0;
            g_surface_height = 0;
        }
    } else if (g_swap_count <= 8 || (g_swap_count % 60) == 0) {
        Log("[EXECUTOR_PIGLET] eglSwapBuffers ok frame=%llu reason=%s",
            static_cast<unsigned long long>(g_swap_count), reason ? reason : "guest");
    }
    const char* freeze_after_env = std::getenv("EXECUTOR_PIGLET_FREEZE_AFTER_GUEST_SWAP_FRAMES");
    if (swapped && freeze_after_env && freeze_after_env[0] != '\0' &&
        std::strcmp(freeze_after_env, "0") != 0 && reason &&
        std::strcmp(reason, "guest-eglSwapBuffers") == 0) {
        char* end = nullptr;
        const auto freeze_after = std::strtoull(freeze_after_env, &end, 10);
        if (end != freeze_after_env && freeze_after > 0 && g_swap_count >= freeze_after) {
            if (g_guest_swap_freeze_log_budget-- > 0) {
                Log("[EXECUTOR_PIGLET] freezing guest thread after guest swap frame=%llu target=%llu",
                    static_cast<unsigned long long>(g_swap_count),
                    static_cast<unsigned long long>(freeze_after));
            }
            while (true) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
    }
    return swapped;
}

void MaybeAutoSwapAfterDraw(const char* reason) {
    if (g_surface == EGL_NO_SURFACE || g_context == EGL_NO_CONTEXT) {
        return;
    }

    {
        GLint fb = 0;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &fb);
        if (fb == 0) {
            g_default_fbo_drawn_since_present = true;
        }
    }

    if (g_guest_egl_controls_swap) {
        const char* itemzflow_auto_present = std::getenv("EXECUTOR_ITEMZFLOW_AUTOPRESENT_DRAWS");
        const char* require_authentic_ui = std::getenv("EXECUTOR_ITEMZFLOW_REQUIRE_AUTHENTIC_UI");
        const bool authentic_ui_required =
            require_authentic_ui != nullptr && require_authentic_ui[0] != '\0' &&
            std::strcmp(require_authentic_ui, "0") != 0;
        const bool allow_itemzflow_auto_present =
            g_itemzflow_overlay_enabled &&
            !authentic_ui_required &&
            itemzflow_auto_present != nullptr && itemzflow_auto_present[0] != '\0' &&
            std::strcmp(itemzflow_auto_present, "0") != 0;
        if (!allow_itemzflow_auto_present) {
            if (g_guest_egl_swap_control_log_budget-- > 0) {
                Log("[EXECUTOR_PIGLET] auto-present deferred to guest eglSwapBuffers reason=%s "
                    "authentic=%d autopresent=%s",
                    reason ? reason : "draw", authentic_ui_required ? 1 : 0,
                    itemzflow_auto_present ? itemzflow_auto_present : "<unset>");
            }
            return;
        }

        GLint framebuffer = 0;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &framebuffer);
        if (framebuffer != 0) {
            if (g_fbo_log_budget-- > 0) {
                Log("[EXECUTOR_PIGLET] itemzflow auto-present skipped framebuffer=%d reason=%s",
                    framebuffer, reason ? reason : "draw");
            }
            return;
        }

        ++g_auto_swap_count;
        if (g_auto_swap_count <= 32 || (g_auto_swap_count % 60) == 0) {
            Log("[EXECUTOR_PIGLET] itemzflow auto-present actual guest draw reason=%s "
                "count=%llu guestSwapExpected=1",
                reason ? reason : "draw", static_cast<unsigned long long>(g_auto_swap_count));
        }
        const bool swapped = SwapAndroidSurface("itemzflow-auto-draw", EGL_NO_SURFACE);
        const char* freeze_after_env = std::getenv("EXECUTOR_ITEMZFLOW_FREEZE_AFTER_AUTOPRESENT_FRAMES");
        if (swapped && freeze_after_env && freeze_after_env[0] != '\0' &&
            std::strcmp(freeze_after_env, "0") != 0) {
            char* end = nullptr;
            const auto freeze_after = std::strtoull(freeze_after_env, &end, 10);
            if (end != freeze_after_env && freeze_after > 0 && g_auto_swap_count >= freeze_after) {
                Log("[EXECUTOR_PIGLET] freezing Itemzflow guest thread after auto-present "
                    "frame=%llu target=%llu",
                    static_cast<unsigned long long>(g_auto_swap_count),
                    static_cast<unsigned long long>(freeze_after));
                while (true) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
            }
        }
        return;
    }

    GLint framebuffer = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &framebuffer);
    if (framebuffer != 0) {
        if (g_fbo_log_budget-- > 0) {
            Log("[EXECUTOR_PIGLET] auto-present skipped for framebuffer=%d reason=%s",
                framebuffer, reason ? reason : "draw");
        }
        return;
    }

    g_default_fbo_drawn_since_present = true;
    if (g_auto_swap_count == 0) {
        ++g_auto_swap_count;
        Log("[EXECUTOR_PIGLET] per-draw auto-present replaced by swap-before-clear (reason=%s)",
            reason ? reason : "draw");
    }
}

std::string ShaderPreview(const char* source, GLint length) {
    std::string preview;
    if (source == nullptr || length <= 0) {
        return preview;
    }
    const GLint preview_length = std::min<GLint>(length, 96);
    preview.reserve(static_cast<std::size_t>(preview_length));
    for (GLint i = 0; i < preview_length; ++i) {
        const unsigned char c = static_cast<unsigned char>(source[i]);
        if (c == '\n' || c == '\r' || c == '\t') {
            preview.push_back(' ');
        } else if (c >= 32 && c < 127) {
            preview.push_back(static_cast<char>(c));
        } else {
            preview.push_back('?');
        }
    }
    return preview;
}

bool GlslDirectiveStartsWith(const std::string& line, const char* directive) {
    std::size_t first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos) {
        return false;
    }
    const std::size_t directive_len = std::strlen(directive);
    if (line.compare(first, directive_len, directive) != 0) {
        return false;
    }
    const std::size_t after = first + directive_len;
    return after == line.size() || line[after] == ' ' || line[after] == '\t' ||
           line[after] == '\r' || line[after] == '\n';
}

std::string SanitizeGlslPreprocessor(const std::string& source) {
    std::string sanitized;
    sanitized.reserve(source.size());

    int conditional_depth = 0;
    int dropped_orphans = 0;
    std::size_t pos = 0;
    while (pos < source.size()) {
        const std::size_t newline = source.find('\n', pos);
        const bool has_newline = newline != std::string::npos;
        const std::size_t end = has_newline ? newline : source.size();
        const std::string line = source.substr(pos, end - pos);

        const bool begins_if = GlslDirectiveStartsWith(line, "#if") ||
                               GlslDirectiveStartsWith(line, "#ifdef") ||
                               GlslDirectiveStartsWith(line, "#ifndef");
        const bool begins_else = GlslDirectiveStartsWith(line, "#else") ||
                                 GlslDirectiveStartsWith(line, "#elif");
        const bool begins_endif = GlslDirectiveStartsWith(line, "#endif");

        bool keep = true;
        std::string kept_line = line;
        if (begins_if) {
            ++conditional_depth;
        } else if (begins_endif) {
            if (conditional_depth > 0) {
                --conditional_depth;
            } else {
                keep = false;
                ++dropped_orphans;
            }
        } else if (begins_else && conditional_depth == 0) {
            keep = false;
            ++dropped_orphans;
        } else if (conditional_depth == 0) {
            const std::size_t inline_endif = kept_line.find("#endif");
            const std::size_t inline_else = kept_line.find("#else");
            const std::size_t inline_elif = kept_line.find("#elif");
            const std::size_t inline_directive =
                std::min({inline_endif == std::string::npos ? kept_line.size() : inline_endif,
                          inline_else == std::string::npos ? kept_line.size() : inline_else,
                          inline_elif == std::string::npos ? kept_line.size() : inline_elif});
            if (inline_directive != kept_line.size()) {
                kept_line.resize(inline_directive);
                ++dropped_orphans;
            }
        }

        if (keep) {
            sanitized.append(kept_line);
            if (has_newline) {
                sanitized.push_back('\n');
            }
        }
        if (!has_newline) {
            break;
        }
        pos = newline + 1;
    }

    if (dropped_orphans > 0 && g_shader_sanitize_log_budget-- > 0) {
        Log("[EXECUTOR_PIGLET] GLSL sanitizer dropped orphan preprocessor lines=%d",
            dropped_orphans);
    }
    return sanitized;
}

std::string CleanGlslSource(const char* source, GLint length) {
    if (source == nullptr || length <= 0) {
        return {};
    }

    const char* begin = source;
    const char* end = source + length;
    const std::array<const char*, 6> anchors = {
        "#version", "precision", "attribute", "uniform", "varying", "void main",
    };
    const char* first_source_byte = end;
    for (const char* anchor : anchors) {
        const char* found = std::search(begin, end, anchor, anchor + std::strlen(anchor));
        if (found != end && found < first_source_byte) {
            first_source_byte = found;
        }
    }
    if (first_source_byte == end) {
        return {};
    }

    std::string clean;
    clean.reserve(static_cast<std::size_t>(end - first_source_byte));
    for (const char* it = first_source_byte; it < end; ++it) {
        const unsigned char c = static_cast<unsigned char>(*it);
        if (c == '\0') {
            break;
        }
        if (c == '\n' || c == '\r' || c == '\t' || (c >= 32 && c < 127)) {
            clean.push_back(static_cast<char>(c));
            continue;
        }
        if (clean.find('}') != std::string::npos) {
            break;
        }
        clean.push_back(' ');
    }

    const auto last_brace = clean.find_last_of('}');
    if (last_brace != std::string::npos) {
        clean.resize(last_brace + 1);
    }
    while (!clean.empty()) {
        const unsigned char c = static_cast<unsigned char>(clean.back());
        if (c != '\r' && c != '\n' && c != '\t' && c != ' ') {
            break;
        }
        clean.pop_back();
    }
    return SanitizeGlslPreprocessor(clean);
}

const char* FindEmbeddedGlslSource(const void* binary, GLsizei length, GLint* source_length) {
    if (binary == nullptr || length <= 16 || source_length == nullptr) {
        return nullptr;
    }
    const auto* bytes = static_cast<const unsigned char*>(binary);
    const char* begin = reinterpret_cast<const char*>(bytes);
    const char* end = begin + length;
    const std::array<const char*, 6> anchors = {
        "#version", "precision", "attribute", "uniform", "varying", "void main",
    };
    const char* source = end;
    for (const char* anchor : anchors) {
        const char* found = std::search(begin, end, anchor, anchor + std::strlen(anchor));
        if (found != end && found < source) {
            source = found;
        }
    }
    if (source == end) {
        return nullptr;
    }

    constexpr std::array<char, 9> kMain = {'v', 'o', 'i', 'd', ' ', 'm', 'a', 'i', 'n'};
    if (std::search(source, end, kMain.begin(), kMain.end()) == end) {
        return nullptr;
    }

    const char* source_end = source;
    while (source_end < end && *source_end != '\0') {
        ++source_end;
    }
    while (source_end > source) {
        const unsigned char c = static_cast<unsigned char>(*(source_end - 1));
        if (c != '\0' && c != '\r' && c != '\n' && c != '\t' && c != ' ') {
            break;
        }
        --source_end;
    }
    const auto found_length = static_cast<GLint>(source_end - source);
    if (found_length <= 16) {
        return nullptr;
    }

    Log("[EXECUTOR_PIGLET] glShaderBinary GLSL source offset=%td size=%d preview='%s'",
        source - begin, found_length, ShaderPreview(source, found_length).c_str());
    *source_length = found_length;
    return source;
}

const char* FallbackShaderFor(GLenum type) {
    static constexpr const char* kVertex =
        "precision mediump float;"
        "attribute vec4 a_Position;"
        "attribute vec3 vertex;"
        "attribute vec2 tex_coord;"
        "attribute vec2 a_TextureCoordinates;"
        "attribute vec4 color;"
        "uniform mat4 model;"
        "uniform mat4 view;"
        "uniform mat4 projection;"
        "uniform vec4 Color;"
        "uniform vec4 u_color;"
        "varying vec2 vTexCoord;"
        "varying vec2 v_TextureCoordinates;"
        "varying vec4 fragColor;"
        "void main(void){"
        "  vec4 p = vec4(vertex, 1.0);"
        "  if ((abs(p.x) + abs(p.y) + abs(p.z)) < 0.00001) p = a_Position;"
        "  vTexCoord = tex_coord;"
        "  if ((abs(vTexCoord.x) + abs(vTexCoord.y)) < 0.00001) vTexCoord = a_TextureCoordinates;"
        "  v_TextureCoordinates = vTexCoord;"
        "  fragColor = color;"
        "  if (fragColor.a <= 0.0) fragColor = Color;"
        "  if (fragColor.a <= 0.0) fragColor = u_color;"
        "  vec4 projected = projection * view * model * p;"
        "  if ((abs(projected.x) + abs(projected.y) + abs(projected.w)) > 0.00001) {"
        "    gl_Position = projected;"
        "  } else if ((abs(p.x) > 2.0) || (abs(p.y) > 2.0)) {"
        "    gl_Position = vec4((p.x / 1920.0) * 2.0 - 1.0,"
        "                       (p.y / 1080.0) * 2.0 - 1.0,"
        "                       p.z, 1.0);"
        "  } else {"
        "    gl_Position = p;"
        "  }"
        "}";
    static constexpr const char* kFragment =
        "precision mediump float;"
        "varying vec2 vTexCoord;"
        "varying vec2 v_TextureCoordinates;"
        "varying vec4 fragColor;"
        "uniform sampler2D texture;"
        "uniform sampler2D tex;"
        "uniform sampler2D Texture;"
        "uniform vec4 Color;"
        "uniform vec4 u_color;"
        "void main(void){"
        "  vec4 c = fragColor;"
        "  if (c.a <= 0.0) c = Color;"
        "  if (c.a <= 0.0) c = u_color;"
        "  vec4 t0 = texture2D(texture, vTexCoord);"
        "  vec4 t1 = texture2D(tex, vTexCoord);"
        "  vec4 t2 = texture2D(Texture, vTexCoord);"
        "  vec4 t = t0.a >= t1.a ? t0 : t1;"
        "  t = t.a >= t2.a ? t : t2;"
        "  if (t.a > 0.0) {"
        "    if (c.a <= 0.0) c = vec4(1.0, 1.0, 1.0, 1.0);"
        "    gl_FragColor = vec4(max(t.rgb, c.rgb * t.a), max(t.a, c.a));"
        "  } else {"
        "    if (c.a <= 0.0) c = vec4(0.18, 0.48, 0.92, 1.0);"
        "    gl_FragColor = c;"
        "  }"
        "}";
    return type == GL_VERTEX_SHADER ? kVertex : kFragment;
}

bool CompileShaderSource(GLuint shader, const char* source, GLint source_length,
                         const char* source_kind) {
    const std::string clean_source = CleanGlslSource(source, source_length);
    if (clean_source.empty()) {
        Log("[EXECUTOR_PIGLET] %s shader=%u has no usable GLSL source", source_kind, shader);
        return false;
    }

    g_shader_sources[shader] = clean_source;
    const char* clean_source_ptr = clean_source.c_str();
    glShaderSource(shader, 1, &clean_source_ptr, nullptr);
    glCompileShader(shader);
    GLint compile_status = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_status);
    if (compile_status == GL_TRUE) {
        static int success_logs = 0;
        if (success_logs < 24) {
            Log("[EXECUTOR_PIGLET] %s shader=%u compiled size=%zu preview='%s'",
                source_kind, shader, clean_source.size(),
                ShaderPreview(clean_source.c_str(), static_cast<GLint>(clean_source.size())).c_str());
            ++success_logs;
        }
        return true;
    }
    char info_log[1024] = {};
    glGetShaderInfoLog(shader, sizeof(info_log), nullptr, info_log);
    Log("[EXECUTOR_PIGLET] %s shader=%u compile failed: %s", source_kind, shader, info_log);
    return false;
}

EGLDisplay PS4_SYSV_ABI EglGetDisplay(EGLNativeDisplayType) {
    EnsureDisplay();
    return g_display != EGL_NO_DISPLAY ? g_display : reinterpret_cast<EGLDisplay>(0x5ad71001);
}

EGLBoolean PS4_SYSV_ABI EglInitialize(EGLDisplay, EGLint* major, EGLint* minor) {
    const bool ok = EnsureDisplay();
    if (ok) {
        if (major != nullptr) {
            *major = 1;
        }
        if (minor != nullptr) {
            *minor = 5;
        }
    }
    return ok ? EGL_TRUE : EGL_FALSE;
}

EGLBoolean PS4_SYSV_ABI EglChooseConfig(EGLDisplay, const EGLint* attrib_list, EGLConfig* configs,
                                        EGLint config_size, EGLint* num_config) {
    if (!EnsureDisplay()) {
        return EGL_FALSE;
    }
    const EGLint default_attribs[] = {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 16,
        EGL_STENCIL_SIZE, 8,
        EGL_NONE,
    };
    EGLint count = 0;
    EGLBoolean ok = eglChooseConfig(g_display, attrib_list ? attrib_list : default_attribs,
                                    configs ? configs : &g_config, config_size > 0 ? config_size : 1,
                                    &count);
    if (ok && configs == nullptr && count > 0) {
        ok = eglChooseConfig(g_display, default_attribs, &g_config, 1, &count);
    }
    if (num_config != nullptr) {
        *num_config = std::max<EGLint>(count, ok ? 1 : 0);
    }
    if (configs != nullptr && config_size > 0 && count > 0) {
        g_config = configs[0];
    }
    Log("[EXECUTOR_PIGLET] eglChooseConfig ok=%d count=%d", ok, count);
    return ok;
}

EGLSurface PS4_SYSV_ABI EglCreateWindowSurface(EGLDisplay, EGLConfig config, EGLNativeWindowType,
                                               const EGLint*) {
    if (config != nullptr) {
        g_config = config;
    }
    g_guest_egl_controls_swap = true;
    if (g_guest_egl_swap_control_log_budget-- > 0) {
        Log("[EXECUTOR_PIGLET] guest EGL window surface requested; draw auto-present disabled");
    }
    return EnsureSurface() ? g_surface : EGL_NO_SURFACE;
}

EGLSurface PS4_SYSV_ABI EglCreatePbufferSurface(EGLDisplay, EGLConfig config, const EGLint*) {
    if (!EnsureDisplay()) {
        return EGL_NO_SURFACE;
    }
    if (config != nullptr) {
        g_config = config;
    }
    const EGLint attrs[] = {EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE};
    return eglCreatePbufferSurface(g_display, g_config, attrs);
}

EGLSurface PS4_SYSV_ABI EglCreatePixmapSurface(EGLDisplay, EGLConfig, EGLNativePixmapType,
                                               const EGLint*) {
    return EglCreatePbufferSurface(g_display, g_config, nullptr);
}

EGLSurface PS4_SYSV_ABI EglCreatePbufferFromClientBuffer(EGLDisplay, EGLenum, EGLClientBuffer,
                                                         EGLConfig, const EGLint*) {
    return EglCreatePbufferSurface(g_display, g_config, nullptr);
}

EGLContext PS4_SYSV_ABI EglCreateContext(EGLDisplay, EGLConfig config, EGLContext,
                                         const EGLint*) {
    if (config != nullptr && IsMainGlThread()) {
        g_config = config;
    }
    return EnsureContext() ? g_context : EGL_NO_CONTEXT;
}

EGLBoolean PS4_SYSV_ABI EglMakeCurrent(EGLDisplay, EGLSurface draw, EGLSurface read,
                                       EGLContext context) {
    if (!EnsureContext()) {
        return EGL_FALSE;
    }
    if (!IsMainGlThread()) {
        return EGL_TRUE;
    }
    if (draw != EGL_NO_SURFACE) {
        g_surface = draw;
        g_guest_egl_controls_swap = true;
        if (g_guest_egl_swap_control_log_budget-- > 0) {
            Log("[EXECUTOR_PIGLET] guest EGL make-current draw surface=%p; draw auto-present disabled",
                static_cast<void*>(draw));
        }
    }
    if (context != EGL_NO_CONTEXT) {
        g_context = context;
    }
    return eglMakeCurrent(g_display, g_surface, read != EGL_NO_SURFACE ? read : g_surface,
                          g_context);
}

EGLBoolean PS4_SYSV_ABI EglSwapBuffers(EGLDisplay, EGLSurface surface) {
    g_guest_egl_controls_swap = true;
    g_clears_without_guest_swap.store(0);
    return SwapAndroidSurface("guest-eglSwapBuffers", surface);
}

int PS4_SYSV_ABI OrbisGlSwapBuffers() {
    if (!g_guest_egl_controls_swap) {
        Log("[EXECUTOR_PIGLET] guest orbisGlSwapBuffers detected -> authentic present boundary; "
            "per-draw auto-present + swap-before-clear disabled");
    }
    g_guest_egl_controls_swap = true;
    g_default_fbo_drawn_since_present = false;
    g_clears_without_guest_swap.store(0);
    return SwapAndroidSurface("guest-orbisGlSwapBuffers", EGL_NO_SURFACE) == EGL_TRUE ? 0 : -1;
}

EGLBoolean PS4_SYSV_ABI EglDestroySurface(EGLDisplay, EGLSurface surface) {
    if (g_display != EGL_NO_DISPLAY && surface != EGL_NO_SURFACE) {
        eglDestroySurface(g_display, surface);
    }
    if (surface == g_surface) {
        g_surface = EGL_NO_SURFACE;
        g_surface_window = nullptr;
        g_surface_width = 0;
        g_surface_height = 0;
    }
    return EGL_TRUE;
}

EGLBoolean PS4_SYSV_ABI EglDestroyContext(EGLDisplay, EGLContext context) {
    if (g_display != EGL_NO_DISPLAY && context != EGL_NO_CONTEXT) {
        eglDestroyContext(g_display, context);
    }
    if (context == g_context) {
        g_context = EGL_NO_CONTEXT;
    }
    return EGL_TRUE;
}

EGLBoolean PS4_SYSV_ABI EglBindAPI(EGLenum api) {
    EnsureDisplay();
    return eglBindAPI(api != 0 ? api : EGL_OPENGL_ES_API);
}

EGLBoolean PS4_SYSV_ABI EglBindTexImage(EGLDisplay display, EGLSurface surface, EGLint buffer) {
    EnsureContext();
    return eglBindTexImage(display != EGL_NO_DISPLAY ? display : g_display,
                           surface != EGL_NO_SURFACE ? surface : g_surface, buffer);
}

EGLBoolean PS4_SYSV_ABI EglSwapInterval(EGLDisplay display, EGLint interval) {
    EnsureContext();
    if (!eglSwapInterval(display != EGL_NO_DISPLAY ? display : g_display, interval)) {
        Log("[EXECUTOR_PIGLET] eglSwapInterval(%d) failed error=0x%x; returning success",
            interval, eglGetError());
    }
    return EGL_TRUE;
}

EGLBoolean PS4_SYSV_ABI EglGetConfigAttrib(EGLDisplay display, EGLConfig config, EGLint attribute,
                                           EGLint* value) {
    if (value == nullptr) {
        return EGL_FALSE;
    }
    if (EnsureDisplay() && eglGetConfigAttrib(display != EGL_NO_DISPLAY ? display : g_display,
                                              config ? config : g_config, attribute, value)) {
        return EGL_TRUE;
    }

    switch (attribute) {
    case EGL_SURFACE_TYPE:
        *value = EGL_WINDOW_BIT;
        break;
    case EGL_RENDERABLE_TYPE:
        *value = EGL_OPENGL_ES2_BIT;
        break;
    case EGL_RED_SIZE:
    case EGL_GREEN_SIZE:
    case EGL_BLUE_SIZE:
    case EGL_ALPHA_SIZE:
        *value = 8;
        break;
    case EGL_DEPTH_SIZE:
        *value = 16;
        break;
    case EGL_STENCIL_SIZE:
        *value = 8;
        break;
    case EGL_CONFIG_ID:
        *value = 1;
        break;
    default:
        *value = 0;
        break;
    }
    return EGL_TRUE;
}

EGLBoolean PS4_SYSV_ABI EglQuerySurface(EGLDisplay display, EGLSurface surface, EGLint attribute,
                                        EGLint* value) {
    if (value == nullptr) {
        return EGL_FALSE;
    }
    if (EnsureSurface() && eglQuerySurface(display != EGL_NO_DISPLAY ? display : g_display,
                                           surface != EGL_NO_SURFACE ? surface : g_surface,
                                           attribute, value)) {
        return EGL_TRUE;
    }

    switch (attribute) {
    case EGL_WIDTH:
        *value = 1920;
        break;
    case EGL_HEIGHT:
        *value = 1080;
        break;
    default:
        *value = 0;
        break;
    }
    return EGL_TRUE;
}

const char* PS4_SYSV_ABI EglQueryString(EGLDisplay display, EGLint name) {
    EnsureDisplay();
    const char* value = eglQueryString(display != EGL_NO_DISPLAY ? display : g_display, name);
    return value != nullptr ? value : "";
}

EGLDisplay PS4_SYSV_ABI EglGetCurrentDisplay() {
    EnsureDisplay();
    EGLDisplay display = eglGetCurrentDisplay();
    return display != EGL_NO_DISPLAY ? display : g_display;
}

EGLContext PS4_SYSV_ABI EglGetCurrentContext() {
    EnsureContext();
    EGLContext context = eglGetCurrentContext();
    return context != EGL_NO_CONTEXT ? context : g_context;
}

EGLSurface PS4_SYSV_ABI EglGetCurrentSurface(EGLint readdraw) {
    EnsureSurface();
    EGLSurface surface = eglGetCurrentSurface(readdraw);
    return surface != EGL_NO_SURFACE ? surface : g_surface;
}

EGLBoolean PS4_SYSV_ABI EglTerminate(EGLDisplay display) {
    (void)display;
    Log("[EXECUTOR_PIGLET] eglTerminate ignored for persistent Android surface");
    return EGL_TRUE;
}

EGLint PS4_SYSV_ABI EglGetError() {
    return eglGetError();
}

void* PS4_SYSV_ABI EglGetProcAddress(const char* name) {
    void* address = ResolveSymbolByName(name);
    Log("[EXECUTOR_PIGLET] eglGetProcAddress(%s) => %p", name ? name : "<null>", address);
    return address;
}

int PS4_SYSV_ABI ScePigletZero() {
    return 0;
}

int PS4_SYSV_ABI ScePigletTrue() {
    return 1;
}

void PS4_SYSV_ABI GlClear(GLbitfield mask) {
    if (EnsureContext()) {
        const bool normal_swap_before_clear =
            !g_guest_egl_controls_swap && g_default_fbo_drawn_since_present;
        const bool stall_present = g_guest_egl_controls_swap &&
                                   g_default_fbo_drawn_since_present &&
                                   g_clears_without_guest_swap.load() >= 1;
        if (normal_swap_before_clear || stall_present) {
            g_default_fbo_drawn_since_present = false;
            ++g_clear_present_count;
            if (g_clear_present_count <= 8 || (g_clear_present_count % 120) == 0) {
                Log("[EXECUTOR_PIGLET] %s present count=%llu",
                    stall_present ? "stall-present" : "swap-before-clear",
                    static_cast<unsigned long long>(g_clear_present_count));
            }
            SwapAndroidSurface("swap-before-clear", EGL_NO_SURFACE);
        }
        if (g_guest_egl_controls_swap) {
            g_clears_without_guest_swap.fetch_add(1);
        }
        DrainGlErrors("glClear");
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glDepthMask(GL_TRUE);
        glDisable(GL_SCISSOR_TEST);
        glClearColor(g_clear_color[0], g_clear_color[1], g_clear_color[2], g_clear_color[3]);
        glClearDepthf(1.0f);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        if (g_clear_log_budget-- > 0) {
            GLint viewport[4] = {};
            glGetIntegerv(GL_VIEWPORT, viewport);
            Log("[EXECUTOR_PIGLET] glClear mask=0x%x color=%.3f %.3f %.3f %.3f "
                "viewport=%d,%d %dx%d",
                mask, g_clear_color[0], g_clear_color[1], g_clear_color[2],
                g_clear_color[3], viewport[0], viewport[1], viewport[2], viewport[3]);
        }
        glClear(mask);
        const GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            static int clear_error_log_budget = 16;
            if (clear_error_log_budget-- > 0) {
                Log("[EXECUTOR_PIGLET] glClear mask=0x%x still failed error=0x%x", mask, error);
            }
        }
    }
}

void PS4_SYSV_ABI GlClearColor(GLfloat r, GLfloat g, GLfloat b, GLfloat a) {
    g_clear_color[0] = r;
    g_clear_color[1] = g;
    g_clear_color[2] = b;
    g_clear_color[3] = a;
    if (EnsureContext()) {
        glClearColor(r, g, b, a);
    }
}

GLenum PS4_SYSV_ABI GlGetError() {
    return EnsureContext() ? glGetError() : GL_NO_ERROR;
}

const GLubyte* PS4_SYSV_ABI GlGetString(GLenum name) {
    if (EnsureContext()) {
        const GLubyte* value = glGetString(name);
        if (value != nullptr) {
            return value;
        }
    }
    static constexpr const GLubyte kRenderer[] = "Executor Android Piglet GLES";
    static constexpr const GLubyte kVersion[] = "OpenGL ES 2.0 Executor";
    static constexpr const GLubyte kVendor[] = "Executor";
    static constexpr const GLubyte kExtensions[] = "";
    switch (name) {
    case GL_RENDERER:
        return kRenderer;
    case GL_VERSION:
    case GL_SHADING_LANGUAGE_VERSION:
        return kVersion;
    case GL_VENDOR:
        return kVendor;
    default:
        return kExtensions;
    }
}

const GLubyte* PS4_SYSV_ABI GlGetStringi(GLenum name, GLuint index) {
    (void)name;
    (void)index;
    return reinterpret_cast<const GLubyte*>("");
}

void PS4_SYSV_ABI GlViewport(GLint x, GLint y, GLsizei width, GLsizei height) {
    if (EnsureContext()) {
        ApplyDefaultFramebufferViewport(x, y, width, height, "glViewport");
    }
}

void PS4_SYSV_ABI GlUseProgram(GLuint program) {
    if (EnsureContext()) {
        if (program != g_current_program) {
            g_client_attribs = {};
            for (GLuint index = 0; index < g_client_attribs.size(); ++index) {
                glDisableVertexAttribArray(index);
            }
            if (g_program_state_reset_log_budget-- > 0) {
                Log("[EXECUTOR_PIGLET] glUseProgram reset client attrib state old=%u new=%u",
                    g_current_program, program);
            }
            g_current_program = program;
        }
        DrainGlErrors("glUseProgram");
        glUseProgram(program);
        LogGlErrorAfter("glUseProgram");
    }
}
void PS4_SYSV_ABI GlEnable(GLenum cap) { if (EnsureContext()) glEnable(cap); }
void PS4_SYSV_ABI GlDisable(GLenum cap) { if (EnsureContext()) glDisable(cap); }
void PS4_SYSV_ABI GlBlendEquation(GLenum mode) { if (EnsureContext()) glBlendEquation(mode); }
void PS4_SYSV_ABI GlBlendFunc(GLenum sfactor, GLenum dfactor) {
    if (EnsureContext()) glBlendFunc(sfactor, dfactor);
}
void PS4_SYSV_ABI GlActiveTexture(GLenum texture) {
    if (EnsureContext()) {
        PigletLastGlTrace("[EXECUTOR_PIGLET_LAST_GL] active api=glActiveTexture "
                          "stage=GL_CALL_BEGIN texture=0x%x oldUnit=%u",
                          texture, g_active_texture_unit);
        glActiveTexture(texture);
        if (texture >= GL_TEXTURE0 && texture < GL_TEXTURE0 + g_bound_texture_2d.size()) {
            g_active_texture_unit = texture - GL_TEXTURE0;
        }
        PigletLastGlTrace("[EXECUTOR_PIGLET_LAST_GL] active api=glActiveTexture "
                          "stage=GL_CALL_RETURN texture=0x%x newUnit=%u",
                          texture, g_active_texture_unit);
        if (g_texture_state_log_budget-- > 0) {
            Log("[EXECUTOR_PIGLET] glActiveTexture texture=0x%x unit=%u", texture,
                g_active_texture_unit);
        }
    }
}
void PS4_SYSV_ABI GlBindTexture(GLenum target, GLuint texture) {
    if (EnsureContext()) {
        GLint old_bound_texture = 0;
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &old_bound_texture);
        PigletLastGlTrace("[EXECUTOR_PIGLET_LAST_GL] bind api=glBindTexture stage=GL_CALL_BEGIN "
                          "target=0x%x texture=%u active=%u oldBound2d=%d",
                          target, texture, g_active_texture_unit, old_bound_texture);
        if (g_texture_driver_stage_log_budget-- > 0) {
            Log("[EXECUTOR_PIGLET_TEX] api=glBindTexture stage=GL_CALL_BEGIN target=0x%x "
                "texture=%u activeUnit=%u oldBound2d=%d",
                target, texture, g_active_texture_unit, old_bound_texture);
        }
        DrainGlErrors("glBindTexture begin");
        glBindTexture(target, texture);
        const GLenum bind_error = glGetError();
        if (target == GL_TEXTURE_2D && g_active_texture_unit < g_bound_texture_2d.size()) {
            g_bound_texture_2d[g_active_texture_unit] = texture;
            if (texture != 0) {
                g_last_nonzero_texture_2d[g_active_texture_unit] = texture;
            }
        }
        GLint new_bound_texture = 0;
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &new_bound_texture);
        PigletLastGlTrace("[EXECUTOR_PIGLET_LAST_GL] bind api=glBindTexture stage=GL_CALL_RETURN "
                          "target=0x%x texture=%u active=%u oldBound2d=%d newBound2d=%d "
                          "error=0x%x",
                          target, texture, g_active_texture_unit, old_bound_texture,
                          new_bound_texture, bind_error);
        if (g_texture_driver_stage_log_budget-- > 0) {
            Log("[EXECUTOR_PIGLET_TEX] api=glBindTexture stage=GL_CALL_RETURN target=0x%x "
                "texture=%u activeUnit=%u oldBound2d=%d newBound2d=%d error=0x%x",
                target, texture, g_active_texture_unit, old_bound_texture, new_bound_texture,
                bind_error);
        }
        if (g_texture_state_log_budget-- > 0) {
            const GLuint last =
                target == GL_TEXTURE_2D && g_active_texture_unit < g_last_nonzero_texture_2d.size()
                    ? g_last_nonzero_texture_2d[g_active_texture_unit]
                    : 0;
            const GLuint last_alpha =
                target == GL_TEXTURE_2D && g_active_texture_unit < g_last_alpha_texture_2d.size()
                    ? g_last_alpha_texture_2d[g_active_texture_unit]
                    : 0;
            Log("[EXECUTOR_PIGLET] glBindTexture target=0x%x texture=%u activeUnit=%u "
                "lastNonZero=%u lastAlpha=%u",
                target, texture, g_active_texture_unit, last, last_alpha);
        }
    }
}
void PS4_SYSV_ABI GlTexParameteri(GLenum target, GLenum pname, GLint param) {
    if (EnsureContext()) {
        PigletLastGlTrace("[EXECUTOR_PIGLET_LAST_GL] texparam api=glTexParameteri "
                          "stage=GL_CALL_BEGIN target=0x%x pname=0x%x param=%d active=%u "
                          "bound2d=%u",
                          target, pname, param, g_active_texture_unit, CurrentBoundTexture2D());
        glTexParameteri(target, pname, param);
        const GLenum error = glGetError();
        PigletLastGlTrace("[EXECUTOR_PIGLET_LAST_GL] texparam api=glTexParameteri "
                          "stage=GL_CALL_RETURN target=0x%x pname=0x%x param=%d active=%u "
                          "bound2d=%u error=0x%x",
                          target, pname, param, g_active_texture_unit, CurrentBoundTexture2D(),
                          error);
        LogGlErrorAfter("glTexParameteri");
    }
}
void PS4_SYSV_ABI GlTexParameterf(GLenum target, GLenum pname, GLfloat param) {
    if (EnsureContext()) {
        PigletLastGlTrace("[EXECUTOR_PIGLET_LAST_GL] texparam api=glTexParameterf "
                          "stage=GL_CALL_BEGIN target=0x%x pname=0x%x param=%.4f active=%u "
                          "bound2d=%u",
                          target, pname, static_cast<double>(param), g_active_texture_unit,
                          CurrentBoundTexture2D());
        glTexParameterf(target, pname, param);
        const GLenum error = glGetError();
        PigletLastGlTrace("[EXECUTOR_PIGLET_LAST_GL] texparam api=glTexParameterf "
                          "stage=GL_CALL_RETURN target=0x%x pname=0x%x param=%.4f active=%u "
                          "bound2d=%u error=0x%x",
                          target, pname, static_cast<double>(param), g_active_texture_unit,
                          CurrentBoundTexture2D(), error);
        LogGlErrorAfter("glTexParameterf");
    }
}
void PS4_SYSV_ABI GlTexImage2D(GLenum target, GLint level, GLint internalformat, GLsizei width,
                               GLsizei height, GLint border, GLenum format, GLenum type,
                               const void* pixels) {
    if (!EnsureContext()) {
        return;
    }
    const std::uint64_t sequence = ++g_texture_upload_sequence;
    LogTextureDriverStage(sequence, "glTexImage2D", "WRAPPER_ENTER", target, level,
                          internalformat, width, height, 0, 0, border, format, type, pixels,
                          "guest");
    LogTextureUploadSummary("glTexImage2D upload", target, level, internalformat, width, height,
                            format, type, pixels);
    if (format == GL_ALPHA && type == GL_UNSIGNED_BYTE && pixels != nullptr) {
        LogTextureDriverStage(sequence, "glTexImage2D", "ALPHA_PATH_BEGIN", target, level,
                              internalformat, width, height, 0, 0, border, format, type, pixels,
                              "guest-alpha");
        if (target == GL_TEXTURE_2D && g_active_texture_unit < g_last_alpha_texture_2d.size() &&
            g_bound_texture_2d[g_active_texture_unit] != 0) {
            g_last_alpha_texture_2d[g_active_texture_unit] =
                g_bound_texture_2d[g_active_texture_unit];
        }
        const auto* alpha = static_cast<const GLubyte*>(pixels);
        std::size_t nonzero_before = 0;
        std::size_t nonzero_after = 0;
        auto display_alpha =
            BuildItemzflowDisplayAlpha(alpha, width, height, "glTexImage2D", &nonzero_before,
                                       &nonzero_after);
        const GLubyte* upload_alpha = display_alpha.empty() ? alpha : display_alpha.data();
        LogTextureDriverStage(sequence, "glTexImage2D", "ALPHA_DISPLAY_READY", target, level,
                              internalformat, width, height, 0, 0, border, format, type,
                              upload_alpha, display_alpha.empty() ? "guest-alpha"
                                                                 : "display-alpha",
                              display_alpha.size());
        RememberAlphaTextureUpload(target, width, height, 0, 0, width, height, upload_alpha);
        if (target == GL_TEXTURE_2D && width >= 256 && width == height && alpha != nullptr) {
            g_font_atlas_watch.texture = CurrentBoundTexture2D();
            g_font_atlas_watch.buffer = alpha;
            g_font_atlas_watch.width = width;
            g_font_atlas_watch.height = height;
            g_font_atlas_watch.last_nonzero = nonzero_before;
            g_font_atlas_watch.frame_throttle = 0;
            Log("[EXECUTOR_PIGLET] font-atlas watch armed tex=%u %dx%d buf=%p nz0=%zu",
                g_font_atlas_watch.texture, width, height, static_cast<const void*>(alpha),
                nonzero_before);
        }
        LogTextureDriverStage(sequence, "glTexImage2D", "ALPHA_SHADOW_READY", target, level,
                              internalformat, width, height, 0, 0, border, format, type,
                              upload_alpha, "alpha-shadow");
        LogTextureDriverStage(sequence, "glTexImage2D", "ALPHA_TO_RGBA_BEGIN", target, level,
                              internalformat, width, height, 0, 0, border, format, type,
                              upload_alpha, "alpha-convert");
        auto rgba = AlphaToRgba(upload_alpha, width, height);
        LogTextureDriverStage(sequence, "glTexImage2D", "ALPHA_TO_RGBA_READY", target, level,
                              GL_RGBA, width, height, 0, 0, border, GL_RGBA, GL_UNSIGNED_BYTE,
                              rgba.empty() ? nullptr : rgba.data(), "alpha->rgba", rgba.size());
        if (!rgba.empty()) {
            if (g_texture_upload_log_budget-- > 0) {
                const std::size_t first_alpha = FirstNonZeroAlpha(upload_alpha, width, height);
                const GLubyte first_alpha_value =
                    first_alpha != SIZE_MAX ? upload_alpha[first_alpha] : 0;
                Log("[EXECUTOR_PIGLET] glTexImage2D alpha->rgba target=0x%x size=%dx%d "
                    "nonzeroAlpha=%zu displayNonzeroAlpha=%zu firstAlpha=%zu "
                    "firstAlphaValue=0x%02x atlasAssist=%d atlasTexture=%u",
                    target, width, height, nonzero_before, nonzero_after,
                    first_alpha, static_cast<unsigned>(first_alpha_value),
                    display_alpha.empty() ? 0 : 1,
                    g_active_texture_unit < g_last_alpha_texture_2d.size()
                        ? g_last_alpha_texture_2d[g_active_texture_unit]
                        : 0);
            }
            DrainGlErrors("glTexImage2D alpha->rgba begin");
            LogTextureDriverStage(sequence, "glTexImage2D", "GL_CALL_BEGIN", target, level,
                                  GL_RGBA, width, height, 0, 0, border, GL_RGBA,
                                  GL_UNSIGNED_BYTE, rgba.data(), "alpha->rgba", rgba.size());
            glTexImage2D(target, level, GL_RGBA, width, height, border, GL_RGBA, GL_UNSIGNED_BYTE,
                         rgba.data());
            const GLenum upload_error = glGetError();
            LogTextureDriverStage(sequence, "glTexImage2D", "GL_CALL_RETURN", target, level,
                                  GL_RGBA, width, height, 0, 0, border, GL_RGBA,
                                  GL_UNSIGNED_BYTE, rgba.data(), "alpha->rgba", rgba.size(),
                                  upload_error);
            RememberTextureUploadInfo(sequence, width, height, GL_RGBA, GL_RGBA,
                                      GL_UNSIGNED_BYTE, true, false);
            ApplyTexture2DCompletenessDefaults(target);
            LogGlErrorAfter("glTexImage2D alpha->rgba defaults");
            return;
        }
    }
    if (IsColorIconUploadCandidate(target, level, internalformat, width, height, format, type,
                                   pixels)) {
        const GLsizei clamp_size = IconClampSize();
        auto clamped = BuildClampedRgbaTexture(pixels, width, height, format, clamp_size);
        LogTextureDriverStage(sequence, "glTexImage2D", "ICON_CLAMP_READY", target, level,
                              GL_RGBA, clamp_size, clamp_size, 0, 0, border, GL_RGBA,
                              GL_UNSIGNED_BYTE, clamped.data(), "icon-clamp", clamped.size());
        Log("[EXECUTOR_PIGLET_ICON_CLAMP] seq=%llu texture=%u from=%dx%d format=0x%x "
            "internal=0x%x to=%dx%d bytes=%zu",
            static_cast<unsigned long long>(sequence), CurrentBoundTexture2D(), width, height,
            format, internalformat, clamp_size, clamp_size, clamped.size());
        DrainGlErrors("glTexImage2D icon-clamp begin");
        LogTextureDriverStage(sequence, "glTexImage2D", "GL_CALL_BEGIN", target, level, GL_RGBA,
                              clamp_size, clamp_size, 0, 0, border, GL_RGBA, GL_UNSIGNED_BYTE,
                              clamped.data(), "icon-clamp", clamped.size());
        glTexImage2D(target, level, GL_RGBA, clamp_size, clamp_size, border, GL_RGBA,
                     GL_UNSIGNED_BYTE, clamped.data());
        const GLenum upload_error = glGetError();
        LogTextureDriverStage(sequence, "glTexImage2D", "GL_CALL_RETURN", target, level, GL_RGBA,
                              clamp_size, clamp_size, 0, 0, border, GL_RGBA, GL_UNSIGNED_BYTE,
                              clamped.data(), "icon-clamp", clamped.size(), upload_error);
        RememberTextureUploadInfo(sequence, clamp_size, clamp_size, GL_RGBA, GL_RGBA,
                                  GL_UNSIGNED_BYTE, false, true);
        ApplyTexture2DCompletenessDefaults(target);
        LogGlErrorAfter("glTexImage2D icon-clamp defaults");
        if (target == GL_TEXTURE_2D && clamp_size >= 128) {
            g_itemzflow_real_cover_texture = CurrentBoundTexture2D();
            g_itemzflow_real_cover_w = clamp_size;
            g_itemzflow_real_cover_h = clamp_size;
        }
        return;
    }
    DrainGlErrors("glTexImage2D begin");
    LogTextureDriverStage(sequence, "glTexImage2D", "GL_CALL_BEGIN", target, level,
                          internalformat, width, height, 0, 0, border, format, type, pixels,
                          "guest");
    glTexImage2D(target, level, internalformat, width, height, border, format, type, pixels);
    const GLenum upload_error = glGetError();
    LogTextureDriverStage(sequence, "glTexImage2D", "GL_CALL_RETURN", target, level,
                          internalformat, width, height, 0, 0, border, format, type, pixels,
                          "guest", 0, upload_error);
    RememberTextureUploadInfo(sequence, width, height, internalformat, format, type, false,
                              false);
    if (target == GL_TEXTURE_2D && level == 0 && pixels != nullptr && width >= 128 &&
        height >= 128 && (format == GL_RGBA || format == GL_RGB || internalformat == GL_RGBA ||
                          internalformat == GL_RGB)) {
        const GLuint tex = CurrentBoundTexture2D();
        if (tex != 0 &&
            static_cast<long>(width) * height >=
                static_cast<long>(g_itemzflow_real_cover_w) * g_itemzflow_real_cover_h) {
            g_itemzflow_real_cover_texture = tex;
            g_itemzflow_real_cover_w = width;
            g_itemzflow_real_cover_h = height;
            if (g_texture_upload_log_budget-- > 0) {
                Log("[EXECUTOR_COVERUI] captured real cover texture=%u size=%dx%d format=0x%x",
                    tex, width, height, format);
            }
        }
    }
}
void PS4_SYSV_ABI GlTexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset,
                                  GLsizei width, GLsizei height, GLenum format, GLenum type,
                                  const void* pixels) {
    if (!EnsureContext()) {
        return;
    }
    const std::uint64_t sequence = ++g_texture_upload_sequence;
    LogTextureDriverStage(sequence, "glTexSubImage2D", "WRAPPER_ENTER", target, level, 0,
                          width, height, xoffset, yoffset, 0, format, type, pixels, "guest");
    LogTextureUploadSummary("glTexSubImage2D upload", target, level, 0, width, height, format,
                            type, pixels);
    if (format == GL_ALPHA && type == GL_UNSIGNED_BYTE && pixels != nullptr) {
        LogTextureDriverStage(sequence, "glTexSubImage2D", "ALPHA_PATH_BEGIN", target, level, 0,
                              width, height, xoffset, yoffset, 0, format, type, pixels,
                              "guest-alpha");
        if (target == GL_TEXTURE_2D && g_active_texture_unit < g_last_alpha_texture_2d.size() &&
            g_bound_texture_2d[g_active_texture_unit] != 0) {
            g_last_alpha_texture_2d[g_active_texture_unit] =
                g_bound_texture_2d[g_active_texture_unit];
        }
        const auto* alpha = static_cast<const GLubyte*>(pixels);
        GLint existing_width = width + std::max<GLint>(xoffset, 0);
        GLint existing_height = height + std::max<GLint>(yoffset, 0);
        const GLuint texture = CurrentBoundTexture2D();
        const auto shadow = g_alpha_texture_shadow.find(texture);
        if (shadow != g_alpha_texture_shadow.end() && shadow->second.width > 0 &&
            shadow->second.height > 0) {
            existing_width = shadow->second.width;
            existing_height = shadow->second.height;
        }
        std::size_t nonzero_before = 0;
        std::size_t nonzero_after = 0;
        auto display_alpha =
            BuildItemzflowDisplayAlpha(alpha, width, height, "glTexSubImage2D",
                                       &nonzero_before, &nonzero_after);
        const GLubyte* upload_alpha = display_alpha.empty() ? alpha : display_alpha.data();
        LogTextureDriverStage(sequence, "glTexSubImage2D", "ALPHA_DISPLAY_READY", target, level,
                              0, width, height, xoffset, yoffset, 0, format, type, upload_alpha,
                              display_alpha.empty() ? "guest-alpha" : "display-alpha",
                              display_alpha.size());
        RememberAlphaTextureUpload(target, existing_width, existing_height, xoffset, yoffset,
                                   width, height, upload_alpha);
        LogTextureDriverStage(sequence, "glTexSubImage2D", "ALPHA_SHADOW_READY", target, level,
                              0, width, height, xoffset, yoffset, 0, format, type, upload_alpha,
                              "alpha-shadow");
        LogTextureDriverStage(sequence, "glTexSubImage2D", "ALPHA_TO_RGBA_BEGIN", target, level,
                              0, width, height, xoffset, yoffset, 0, format, type, upload_alpha,
                              "alpha-convert");
        auto rgba = AlphaToRgba(upload_alpha, width, height);
        LogTextureDriverStage(sequence, "glTexSubImage2D", "ALPHA_TO_RGBA_READY", target, level,
                              0, width, height, xoffset, yoffset, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                              rgba.empty() ? nullptr : rgba.data(), "alpha->rgba", rgba.size());
        if (!rgba.empty()) {
            if (g_texture_upload_log_budget-- > 0) {
                const std::size_t first_alpha = FirstNonZeroAlpha(upload_alpha, width, height);
                const GLubyte first_alpha_value =
                    first_alpha != SIZE_MAX ? upload_alpha[first_alpha] : 0;
                Log("[EXECUTOR_PIGLET] glTexSubImage2D alpha->rgba target=0x%x offset=%d,%d "
                    "size=%dx%d nonzeroAlpha=%zu displayNonzeroAlpha=%zu atlasAssist=%d "
                    "firstAlpha=%zu firstAlphaValue=0x%02x atlasTexture=%u",
                    target, xoffset, yoffset, width, height, nonzero_before, nonzero_after,
                    display_alpha.empty() ? 0 : 1,
                    first_alpha, static_cast<unsigned>(first_alpha_value),
                    g_active_texture_unit < g_last_alpha_texture_2d.size()
                        ? g_last_alpha_texture_2d[g_active_texture_unit]
                        : 0);
            }
            DrainGlErrors("glTexSubImage2D alpha->rgba begin");
            LogTextureDriverStage(sequence, "glTexSubImage2D", "GL_CALL_BEGIN", target, level,
                                  0, width, height, xoffset, yoffset, 0, GL_RGBA,
                                  GL_UNSIGNED_BYTE, rgba.data(), "alpha->rgba", rgba.size());
            glTexSubImage2D(target, level, xoffset, yoffset, width, height, GL_RGBA,
                            GL_UNSIGNED_BYTE, rgba.data());
            const GLenum upload_error = glGetError();
            LogTextureDriverStage(sequence, "glTexSubImage2D", "GL_CALL_RETURN", target, level,
                                  0, width, height, xoffset, yoffset, 0, GL_RGBA,
                                  GL_UNSIGNED_BYTE, rgba.data(), "alpha->rgba", rgba.size(),
                                  upload_error);
            RememberTextureUploadInfo(sequence, existing_width, existing_height, GL_RGBA, GL_RGBA,
                                      GL_UNSIGNED_BYTE, true, false);
            ApplyTexture2DCompletenessDefaults(target);
            LogGlErrorAfter("glTexSubImage2D alpha->rgba defaults");
            return;
        }
    }
    DrainGlErrors("glTexSubImage2D begin");
    LogTextureDriverStage(sequence, "glTexSubImage2D", "GL_CALL_BEGIN", target, level, 0,
                          width, height, xoffset, yoffset, 0, format, type, pixels, "guest");
    glTexSubImage2D(target, level, xoffset, yoffset, width, height, format, type, pixels);
    const GLenum upload_error = glGetError();
    LogTextureDriverStage(sequence, "glTexSubImage2D", "GL_CALL_RETURN", target, level, 0,
                          width, height, xoffset, yoffset, 0, format, type, pixels, "guest", 0,
                          upload_error);
    if (target == GL_TEXTURE_2D && CurrentBoundTexture2D() != 0) {
        auto& info = g_texture_upload_info[CurrentBoundTexture2D()];
        info.format = format;
        info.type = type;
        info.sequence = sequence;
        info.bytes = EstimateTextureUploadBytes(width, height, format, type);
    }
}
void PS4_SYSV_ABI GlGenTextures(GLsizei n, GLuint* textures) {
    if (!EnsureContext()) {
        return;
    }
    PigletLastGlTrace("[EXECUTOR_PIGLET_LAST_GL] gen api=glGenTextures stage=GL_CALL_BEGIN "
                      "n=%d out=%p",
                      n, textures);
    if (g_texture_driver_stage_log_budget-- > 0) {
        Log("[EXECUTOR_PIGLET_TEX] api=glGenTextures stage=GL_CALL_BEGIN n=%d out=%p", n,
            textures);
    }
    DrainGlErrors("glGenTextures");
    glGenTextures(n, textures);
    const GLenum error = glGetError();
    PigletLastGlTrace("[EXECUTOR_PIGLET_LAST_GL] gen api=glGenTextures stage=GL_CALL_RETURN "
                      "n=%d out=%p first=%u error=0x%x",
                      n, textures, (textures != nullptr && n > 0) ? textures[0] : 0, error);
    if (g_texture_driver_stage_log_budget-- > 0) {
        Log("[EXECUTOR_PIGLET_TEX] api=glGenTextures stage=GL_CALL_RETURN n=%d out=%p first=%u "
            "error=0x%x",
            n, textures, (textures != nullptr && n > 0) ? textures[0] : 0, error);
    }
    if (g_texture_state_log_budget-- > 0) {
        Log("[EXECUTOR_PIGLET] glGenTextures n=%d out=%p first=%u error=0x%x", n, textures,
            (textures != nullptr && n > 0) ? textures[0] : 0, error);
    }
}
void PS4_SYSV_ABI GlDeleteTextures(GLsizei n, const GLuint* textures) {
    if (!EnsureContext()) {
        return;
    }
    if (n <= 0 || textures == nullptr) {
        return;
    }
    PigletLastGlTrace("[EXECUTOR_PIGLET_LAST_GL] delete api=glDeleteTextures "
                      "stage=GL_CALL_BEGIN n=%d first=%u",
                      n, textures[0]);
    const char* require_authentic =
        std::getenv("EXECUTOR_ITEMZFLOW_REQUIRE_AUTHENTIC_UI");
    const bool authentic_only = require_authentic != nullptr && require_authentic[0] != '\0' &&
                                std::strcmp(require_authentic, "0") != 0;
    const char* protect_itemzflow =
        std::getenv("EXECUTOR_ITEMZFLOW_PROTECT_TEXTURE_LIFETIME");
    if (!authentic_only && protect_itemzflow != nullptr && protect_itemzflow[0] != '\0' &&
        std::strcmp(protect_itemzflow, "0") != 0) {
        static int protected_delete_log_budget = 16;
        if (protected_delete_log_budget-- > 0) {
            Log("[EXECUTOR_PIGLET] protected glDeleteTextures n=%d textures=%p "
                "source=itemzflow_texture_lifetime_guard",
                n, textures);
        }
        return;
    }
    for (GLsizei i = 0; i < n; ++i) {
        const GLuint texture = textures[i];
        if (texture == 0) {
            continue;
        }
        g_alpha_texture_shadow.erase(texture);
        g_texture_upload_info.erase(texture);
        for (std::size_t unit = 0; unit < g_bound_texture_2d.size(); ++unit) {
            if (g_bound_texture_2d[unit] == texture) g_bound_texture_2d[unit] = 0;
            if (g_last_nonzero_texture_2d[unit] == texture) g_last_nonzero_texture_2d[unit] = 0;
            if (g_last_alpha_texture_2d[unit] == texture) g_last_alpha_texture_2d[unit] = 0;
        }
    }
    glDeleteTextures(n, textures);
    const GLenum error = glGetError();
    PigletLastGlTrace("[EXECUTOR_PIGLET_LAST_GL] delete api=glDeleteTextures "
                      "stage=GL_CALL_RETURN n=%d first=%u error=0x%x",
                      n, textures[0], error);
    if (g_texture_state_log_budget-- > 0) {
        Log("[EXECUTOR_PIGLET] glDeleteTextures n=%d first=%u authentic=%d "
            "source=real_texture_lifetime",
            n, textures[0], authentic_only ? 1 : 0);
    }
    LogGlErrorAfter("glDeleteTextures");
}
void PS4_SYSV_ABI GlGenBuffers(GLsizei n, GLuint* buffers) {
    if (EnsureContext()) glGenBuffers(n, buffers);
}
void PS4_SYSV_ABI GlDeleteBuffers(GLsizei n, const GLuint* buffers) {
    if (EnsureContext()) glDeleteBuffers(n, buffers);
}
void PS4_SYSV_ABI GlBindBuffer(GLenum target, GLuint buffer) {
    if (EnsureContext()) glBindBuffer(target, buffer);
}
void PS4_SYSV_ABI GlBufferData(GLenum target, GLsizeiptr size, const void* data, GLenum usage) {
    if (EnsureContext()) {
        glBufferData(target, size, data, usage);
        ShadowBufferUpload(target, size, data);
    }
}
void PS4_SYSV_ABI GlBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size,
                                  const void* data) {
    if (EnsureContext()) {
        glBufferSubData(target, offset, size, data);
        ShadowBufferSubUpload(target, offset, size, data);
    }
}
void PS4_SYSV_ABI GlVertexAttribPointer(GLuint index, GLint size, GLenum type, GLboolean normalized,
                                        GLsizei stride, const void* pointer) {
    if (!EnsureContext()) {
        return;
    }
    if (index == GL_INVALID_INDEX) {
        static int invalid_attrib_budget = 16;
        if (invalid_attrib_budget-- > 0) {
            Log("[EXECUTOR_PIGLET] ignoring glVertexAttribPointer for inactive attrib index");
        }
        return;
    }
    GLuint effective_index = index;
    if (index == 1 && size == 4 && type == GL_FLOAT && g_client_attribs[1].active &&
        g_client_attribs[1].size == 3 && !g_client_attribs[4].active) {
        effective_index = 4;
        static int color_collision_log_budget = 16;
        if (color_collision_log_budget-- > 0) {
            Log("[EXECUTOR_PIGLET] redirected client color attrib collision index=1 -> 4 "
                "pointer=%p stride=%d",
                pointer, stride);
        }
    }
    const bool client_pointer = LooksLikeClientPointer(pointer);
    if (client_pointer) {
        if (effective_index < g_client_attribs.size()) {
            g_client_attribs[effective_index] = {.active = true,
                                                 .size = size,
                                                 .type = type,
                                                 .normalized = normalized,
                                                 .stride = stride,
                                                 .pointer = pointer};
            static int client_pointer_log_budget = 24;
            if (client_pointer_log_budget-- > 0) {
                Log("[EXECUTOR_PIGLET] deferred client vertex attrib index=%u pointer=%p size=%d "
                    "type=0x%x stride=%d requestedIndex=%u",
                    effective_index, pointer, size, type, stride, index);
            }
        }
        return;
    }
    if (effective_index < g_client_attribs.size()) {
        g_client_attribs[effective_index] = {};
    }
    glVertexAttribPointer(effective_index, size, type, normalized, stride, pointer);
    LogGlErrorAfter("glVertexAttribPointer");
}
void PS4_SYSV_ABI GlEnableVertexAttribArray(GLuint index) {
    if (EnsureContext() && index != GL_INVALID_INDEX) {
        glEnableVertexAttribArray(index);
        LogGlErrorAfter("glEnableVertexAttribArray");
    }
}
void PS4_SYSV_ABI GlDisableVertexAttribArray(GLuint index) {
    if (EnsureContext() && index != GL_INVALID_INDEX) {
        if (index < g_client_attribs.size()) {
            g_client_attribs[index] = {};
        }
        glDisableVertexAttribArray(index);
    }
}
void PS4_SYSV_ABI GlDrawArrays(GLenum mode, GLint first, GLsizei count) {
    if (EnsureContext()) {
        const std::uint64_t sequence = ++g_draw_call_sequence;
        PrepareClientAttribsForDraw(first + count);
        PrepareSyntheticTextureAttribsForDraw(mode, first + count);
        ForceWindowDrawState();
        LogFontAtlasDrawState("glDrawArrays", mode, count);
        DrainGlErrors("glDrawArrays");
        TraceDrawEvent("glDrawArrays", "GL_CALL_BEGIN", sequence, mode, first, count, 0,
                       nullptr);
        glDrawArrays(mode, first, count);
        const GLenum error = glGetError();
        TraceDrawEvent("glDrawArrays", "GL_CALL_RETURN", sequence, mode, first, count, 0,
                       nullptr, error);
        if (error != GL_NO_ERROR) {
            LogDrawErrorState("glDrawArrays", mode, count, error);
        } else if (g_draw_success_log_budget-- > 0) {
            NoteWindowDrawBounds();
            Log("[EXECUTOR_PIGLET] glDrawArrays success mode=0x%x first=%d count=%d", mode,
                first, count);
            LogDrawProbe("glDrawArrays success state", mode, first, count);
            ProbeReadbackAfterDraw("glDrawArrays success");
        }
        LogCoverDrawCandidate("glDrawArrays", mode, count);
        MaybeAutoSwapAfterDraw("glDrawArrays");
    }
}
void PS4_SYSV_ABI GlDrawElements(GLenum mode, GLsizei count, GLenum type, const void* indices) {
    if (EnsureContext()) {
        const std::uint64_t sequence = ++g_draw_call_sequence;
        const GLsizei vertex_count = MaxVertexCountFromElements(count, type, indices);
        PrepareClientAttribsForDraw(vertex_count);
        PrepareSyntheticTextureAttribsForDraw(mode, vertex_count);
        ForceWindowDrawState();
        ProbeVboDrawElements(count, type, indices);
        LogFontAtlasDrawState("glDrawElements", mode, count);
        DrainGlErrors("glDrawElements");
        TraceDrawEvent("glDrawElements", "GL_CALL_BEGIN", sequence, mode, 0, count, type,
                       indices);
        glDrawElements(mode, count, type, indices);
        const GLenum error = glGetError();
        TraceDrawEvent("glDrawElements", "GL_CALL_RETURN", sequence, mode, 0, count, type,
                       indices, error);
        if (error != GL_NO_ERROR) {
            LogDrawErrorState("glDrawElements", mode, count, error);
        } else if (g_draw_elements_success_log_budget-- > 0) {
            NoteWindowDrawBounds();
            Log("[EXECUTOR_PIGLET] glDrawElements success mode=0x%x count=%d type=0x%x "
                "vertexCount=%d indices=%p",
                mode, count, type, vertex_count, indices);
            LogDrawProbe("glDrawElements success state", mode, 0, count);
            ProbeReadbackAfterDraw("glDrawElements success");
        }
        LogCoverDrawCandidate("glDrawElements", mode, count);
        MaybeAutoSwapAfterDraw("glDrawElements");
    }
}
GLuint PS4_SYSV_ABI GlCreateShader(GLenum type) {
    if (!EnsureContext()) {
        return 0;
    }
    const GLuint shader = glCreateShader(type);
    if (shader != 0) {
        g_shader_types[shader] = type;
    }
    return shader;
}
void PS4_SYSV_ABI GlShaderSource(GLuint shader, GLsizei count, const GLchar* const* string,
                                 const GLint* length) {
    if (!EnsureContext()) {
        return;
    }
    std::string combined;
    for (GLsizei i = 0; i < count && string != nullptr; ++i) {
        if (string[i] == nullptr) {
            continue;
        }
        const GLint part_length = length != nullptr ? length[i] : -1;
        if (part_length >= 0) {
            combined.append(string[i], string[i] + part_length);
        } else {
            combined.append(string[i]);
        }
        if (combined.size() > 8192) {
            combined.resize(8192);
            break;
        }
    }
    if (!combined.empty()) {
        g_shader_sources[shader] = combined;
    }
    const GLenum type = g_shader_types.count(shader) ? g_shader_types[shader] : 0;
    if (g_shader_source_log_budget-- > 0) {
        std::string preview = combined.substr(0, std::min<std::size_t>(combined.size(), 240));
        for (char& ch : preview) {
            if (ch == '\n' || ch == '\r' || ch == '\t') {
                ch = ' ';
            }
        }
        Log("[EXECUTOR_PIGLET_SHADER] glShaderSource shader=%u type=0x%x bytes=%zu "
            "sample=%d readA=%d readR=%d colorRef=%d preview='%s'",
            shader, type, combined.size(),
            (SourceContains(combined, "texture2D") || SourceContains(combined, "texture(")) ? 1
                                                                                            : 0,
            (SourceContains(combined, ".a") || SourceContains(combined, "alpha")) ? 1 : 0,
            SourceContains(combined, ".r") ? 1 : 0,
            (SourceContains(combined, "color") || SourceContains(combined, "Color") ||
             SourceContains(combined, "v_color"))
                ? 1
                : 0,
            preview.c_str());
    }
    glShaderSource(shader, count, string, length);
}
void PS4_SYSV_ABI GlShaderBinary(GLsizei count, const GLuint* shaders, GLenum binaryformat,
                                 const void* binary, GLsizei length) {
    if (!EnsureContext()) {
        return;
    }
    GLint source_length = 0;
    const char* embedded_source = FindEmbeddedGlslSource(binary, length, &source_length);
    if (embedded_source != nullptr) {
        bool all_ok = true;
        for (GLsizei i = 0; i < count; ++i) {
            bool ok = CompileShaderSource(shaders[i], embedded_source, source_length,
                                          "embedded GLSL");
            if (!ok) {
                const auto type_it = g_shader_types.find(shaders[i]);
                const GLenum type =
                    type_it != g_shader_types.end() ? type_it->second : GL_FRAGMENT_SHADER;
                const char* fallback = FallbackShaderFor(type);
                const GLint fallback_length = static_cast<GLint>(std::strlen(fallback));
                ok = CompileShaderSource(shaders[i], fallback, fallback_length,
                                         "fallback after embedded GLSL");
            }
            all_ok &= ok;
        }
        Log("[EXECUTOR_PIGLET] glShaderBinary compiled embedded/fallback GLSL size=%d ok=%d",
            source_length, all_ok ? 1 : 0);
        return;
    }

    for (GLsizei i = 0; i < count; ++i) {
        const auto type_it = g_shader_types.find(shaders[i]);
        const GLenum type = type_it != g_shader_types.end() ? type_it->second : GL_FRAGMENT_SHADER;
        const char* fallback = FallbackShaderFor(type);
        const GLint fallback_length = static_cast<GLint>(std::strlen(fallback));
        CompileShaderSource(shaders[i], fallback, fallback_length, "fallback GLSL");
    }
    Log("[EXECUTOR_PIGLET] glShaderBinary used fallback GLSL count=%d format=0x%x length=%d",
        count, binaryformat, length);
}
void PS4_SYSV_ABI GlCompileShader(GLuint shader) { if (EnsureContext()) glCompileShader(shader); }
void PS4_SYSV_ABI GlGetShaderiv(GLuint shader, GLenum pname, GLint* params) {
    if (EnsureContext()) glGetShaderiv(shader, pname, params);
}
void PS4_SYSV_ABI GlGetShaderInfoLog(GLuint shader, GLsizei bufSize, GLsizei* length,
                                     GLchar* infoLog) {
    if (EnsureContext()) glGetShaderInfoLog(shader, bufSize, length, infoLog);
}
void PS4_SYSV_ABI GlDeleteShader(GLuint shader) {
    g_shader_types.erase(shader);
    g_shader_sources.erase(shader);
    if (EnsureContext()) glDeleteShader(shader);
}
GLuint PS4_SYSV_ABI GlCreateProgram() {
    return EnsureContext() ? glCreateProgram() : 0;
}
void PS4_SYSV_ABI GlAttachShader(GLuint program, GLuint shader) {
    if (EnsureContext()) {
        g_program_attached_shaders[program].push_back(shader);
        glAttachShader(program, shader);
    }
}
void PS4_SYSV_ABI GlLinkProgram(GLuint program) {
    if (!EnsureContext()) {
        return;
    }
    glBindAttribLocation(program, 0, "a_Position");
    glBindAttribLocation(program, 1, "vertex");
    glBindAttribLocation(program, 2, "a_TextureCoordinates");
    glBindAttribLocation(program, 3, "tex_coord");
    glBindAttribLocation(program, 4, "color");
    const ProgramShaderInfo shader_info = AnalyzeProgramShaders(program);
    g_program_shader_info[program] = shader_info;
    glLinkProgram(program);
    GLint link_status = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &link_status);
    if (g_program_link_log_budget-- > 0 || link_status != GL_TRUE) {
        char info_log[1024] = {};
        glGetProgramInfoLog(program, sizeof(info_log), nullptr, info_log);
        Log("[EXECUTOR_PIGLET] glLinkProgram program=%u status=%d info='%s' "
            "fragSource=%d sample=%d readA=%d readR=%d colorRef=%d",
            program, link_status == GL_TRUE ? 1 : 0, info_log,
            shader_info.has_fragment_source ? 1 : 0, shader_info.samples_texture ? 1 : 0,
            shader_info.reads_alpha ? 1 : 0, shader_info.reads_red ? 1 : 0,
            shader_info.mentions_color ? 1 : 0);
    }
}
void PS4_SYSV_ABI GlGetProgramiv(GLuint program, GLenum pname, GLint* params) {
    if (EnsureContext()) glGetProgramiv(program, pname, params);
}
void PS4_SYSV_ABI GlDeleteProgram(GLuint program) {
    g_program_attached_shaders.erase(program);
    g_program_shader_info.erase(program);
    if (EnsureContext()) glDeleteProgram(program);
}
GLint PS4_SYSV_ABI GlGetUniformLocation(GLuint program, const GLchar* name) {
    const GLint actual_location = EnsureContext() ? glGetUniformLocation(program, name) : -1;
    const GLint synthetic_location = SyntheticUniformLocation(name);
    GLint returned_location = actual_location;
    if (actual_location >= 0 && synthetic_location >= 0) {
        g_uniform_location_map[UniformMapKey(program, synthetic_location)] = actual_location;
        returned_location = synthetic_location;
    }
    if (g_location_log_budget-- > 0) {
        Log("[EXECUTOR_PIGLET] glGetUniformLocation program=%u name=%s => guest:%d actual:%d",
            program, name ? name : "<null>", returned_location, actual_location);
    }
    return returned_location;
}
GLint PS4_SYSV_ABI GlGetAttribLocation(GLuint program, const GLchar* name) {
    const GLint location = EnsureContext() ? glGetAttribLocation(program, name) : -1;
    if (g_location_log_budget-- > 0) {
        Log("[EXECUTOR_PIGLET] glGetAttribLocation program=%u name=%s => %d", program,
            name ? name : "<null>", location);
    }
    return location;
}
void PS4_SYSV_ABI GlUniform1f(GLint location, GLfloat v0) {
    if (EnsureContext()) {
        const GLint actual_location = ResolveUniformLocationForCurrentProgram(location);
        if (actual_location >= 0) {
            glUniform1f(actual_location, v0);
            LogGlErrorAfter("glUniform1f");
        }
    }
}
void PS4_SYSV_ABI GlUniform2f(GLint location, GLfloat v0, GLfloat v1) {
    if (EnsureContext()) {
        const GLint actual_location = ResolveUniformLocationForCurrentProgram(location);
        if (actual_location >= 0) {
            glUniform2f(actual_location, v0, v1);
            LogGlErrorAfter("glUniform2f");
        }
    }
}
void PS4_SYSV_ABI GlUniform4f(GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3) {
    if (EnsureContext()) {
        const GLint actual_location = ResolveUniformLocationForCurrentProgram(location);
        RememberUniform4f(actual_location, v0, v1, v2, v3);
        if (g_uniform4f_log_budget-- > 0) {
            GLint program = 0;
            glGetIntegerv(GL_CURRENT_PROGRAM, &program);
            Log("[EXECUTOR_PIGLET] glUniform4f program=%d guestLocation=%d actualLocation=%d "
                "value=%.3f %.3f %.3f %.3f",
                program, location, actual_location, v0, v1, v2, v3);
        }
        if (actual_location >= 0) {
            glUniform4f(actual_location, v0, v1, v2, v3);
            LogGlErrorAfter("glUniform4f");
        }
    }
}
void PS4_SYSV_ABI GlUniform1i(GLint location, GLint v0) {
    if (EnsureContext()) {
        const GLint actual_location = ResolveUniformLocationForCurrentProgram(location);
        GLint program = 0;
        glGetIntegerv(GL_CURRENT_PROGRAM, &program);
        bool rebound_texture = false;
        GLuint rebound_from = 0;
        const char* rebound_source = "none";
        if (v0 >= 0 && v0 < static_cast<GLint>(g_bound_texture_2d.size()) &&
            g_bound_texture_2d[static_cast<std::size_t>(v0)] == 0 &&
            g_last_nonzero_texture_2d[static_cast<std::size_t>(v0)] != 0) {
            const auto unit = static_cast<std::size_t>(v0);
            const bool prefer_alpha_atlas =
                program != 0 && glGetAttribLocation(static_cast<GLuint>(program), "tex_coord") >= 0 &&
                glGetAttribLocation(static_cast<GLuint>(program), "color") >= 0 &&
                glGetUniformLocation(static_cast<GLuint>(program), "projection") >= 0;
            if (prefer_alpha_atlas && g_last_alpha_texture_2d[unit] != 0) {
                rebound_from = g_last_alpha_texture_2d[unit];
                rebound_source = "alpha_atlas";
            } else {
                rebound_from = g_last_nonzero_texture_2d[unit];
                rebound_source = "last_nonzero";
            }
            glActiveTexture(GL_TEXTURE0 + static_cast<GLenum>(unit));
            glBindTexture(GL_TEXTURE_2D, rebound_from);
            g_bound_texture_2d[unit] = rebound_from;
            if (unit != g_active_texture_unit) {
                glActiveTexture(GL_TEXTURE0 + g_active_texture_unit);
            }
            rebound_texture = true;
        }
        if (g_uniform_sampler_log_budget-- > 0) {
            GLuint bound = 0;
            if (v0 >= 0 && v0 < static_cast<GLint>(g_bound_texture_2d.size())) {
                bound = g_bound_texture_2d[static_cast<std::size_t>(v0)];
            }
            Log("[EXECUTOR_PIGLET] glUniform1i program=%d guestLocation=%d actualLocation=%d "
                "value=%d boundTexture2D=%u rebound=%d reboundTexture=%u reboundSource=%s",
                program, location, actual_location, v0, bound, rebound_texture ? 1 : 0,
                rebound_from, rebound_source);
        }
        if (rebound_texture && g_texture_rebind_log_budget-- > 0) {
            Log("[EXECUTOR_PIGLET] sampler rebound unit=%d texture=%u source=%s before glUniform1i",
                v0, rebound_from, rebound_source);
        }
        if (actual_location >= 0) {
            glUniform1i(actual_location, v0);
            LogGlErrorAfter("glUniform1i");
        }
    }
}
void PS4_SYSV_ABI GlUniformMatrix4fv(GLint location, GLsizei count, GLboolean transpose,
                                     const GLfloat* value) {
    if (EnsureContext()) {
        const GLint actual_location = ResolveUniformLocationForCurrentProgram(location);
        static int matrix_log_budget = 32;
        if (matrix_log_budget-- > 0) {
            GLint program = 0;
            glGetIntegerv(GL_CURRENT_PROGRAM, &program);
            const GLfloat v0 = value != nullptr ? value[0] : 0.0f;
            const GLfloat v5 = value != nullptr ? value[5] : 0.0f;
            const GLfloat v10 = value != nullptr ? value[10] : 0.0f;
            const GLfloat v12 = value != nullptr ? value[12] : 0.0f;
            const GLfloat v13 = value != nullptr ? value[13] : 0.0f;
            Log("[EXECUTOR_PIGLET] glUniformMatrix4fv program=%d guestLocation=%d "
                "actualLocation=%d count=%d transpose=%d diag=%.4f %.4f %.4f trans=%.4f %.4f",
                program, location, actual_location, count, transpose ? 1 : 0, v0, v5, v10,
                v12, v13);
        }
        if (actual_location >= 0) {
            glUniformMatrix4fv(actual_location, count, transpose, value);
            LogGlErrorAfter("glUniformMatrix4fv");
        }
    }
}
void PS4_SYSV_ABI GlGenVertexArrays(GLsizei n, GLuint* arrays) {
    if (EnsureContext()) glGenVertexArrays(n, arrays);
}
void PS4_SYSV_ABI GlBindVertexArray(GLuint array) {
    if (EnsureContext()) glBindVertexArray(array);
}
void PS4_SYSV_ABI GlDeleteVertexArrays(GLsizei n, const GLuint* arrays) {
    if (EnsureContext()) glDeleteVertexArrays(n, arrays);
}
void PS4_SYSV_ABI GlGenFramebuffers(GLsizei n, GLuint* framebuffers) {
    if (EnsureContext()) glGenFramebuffers(n, framebuffers);
}
void PS4_SYSV_ABI GlDeleteFramebuffers(GLsizei n, const GLuint* framebuffers) {
    if (EnsureContext()) glDeleteFramebuffers(n, framebuffers);
}
void PS4_SYSV_ABI GlBindFramebuffer(GLenum target, GLuint framebuffer) {
    if (EnsureContext()) {
        glBindFramebuffer(target, framebuffer);
        if (g_fbo_log_budget-- > 0) {
            GLint status = 0;
            if (target == GL_FRAMEBUFFER || target == GL_DRAW_FRAMEBUFFER) {
                status = static_cast<GLint>(glCheckFramebufferStatus(GL_FRAMEBUFFER));
            }
            Log("[EXECUTOR_PIGLET] glBindFramebuffer target=0x%x framebuffer=%u status=0x%x",
                target, framebuffer, status);
        }
    }
}
void PS4_SYSV_ABI GlGenRenderbuffers(GLsizei n, GLuint* renderbuffers) {
    if (EnsureContext()) glGenRenderbuffers(n, renderbuffers);
}
void PS4_SYSV_ABI GlBindRenderbuffer(GLenum target, GLuint renderbuffer) {
    if (EnsureContext()) glBindRenderbuffer(target, renderbuffer);
}
void PS4_SYSV_ABI GlRenderbufferStorage(GLenum target, GLenum internalformat, GLsizei width,
                                        GLsizei height) {
    if (EnsureContext()) glRenderbufferStorage(target, internalformat, width, height);
}
void PS4_SYSV_ABI GlFramebufferTexture2D(GLenum target, GLenum attachment, GLenum textarget,
                                         GLuint texture, GLint level) {
    if (EnsureContext()) {
        glFramebufferTexture2D(target, attachment, textarget, texture, level);
        if (g_fbo_log_budget-- > 0) {
            Log("[EXECUTOR_PIGLET] glFramebufferTexture2D target=0x%x attachment=0x%x "
                "textarget=0x%x texture=%u level=%d status=0x%x",
                target, attachment, textarget, texture, level, glCheckFramebufferStatus(target));
        }
    }
}
void PS4_SYSV_ABI GlFramebufferRenderbuffer(GLenum target, GLenum attachment,
                                            GLenum renderbuffertarget, GLuint renderbuffer) {
    if (EnsureContext()) {
        glFramebufferRenderbuffer(target, attachment, renderbuffertarget, renderbuffer);
        if (g_fbo_log_budget-- > 0) {
            Log("[EXECUTOR_PIGLET] glFramebufferRenderbuffer target=0x%x attachment=0x%x "
                "renderbuffertarget=0x%x renderbuffer=%u status=0x%x",
                target, attachment, renderbuffertarget, renderbuffer,
                glCheckFramebufferStatus(target));
        }
    }
}
GLenum PS4_SYSV_ABI GlCheckFramebufferStatus(GLenum target) {
    return EnsureContext() ? glCheckFramebufferStatus(target) : GL_FRAMEBUFFER_COMPLETE;
}
void PS4_SYSV_ABI GlGenerateMipmap(GLenum target) {
    if (EnsureContext()) {
        PigletLastGlTrace("[EXECUTOR_PIGLET_LAST_GL] mipmap api=glGenerateMipmap "
                          "stage=GL_CALL_BEGIN target=0x%x active=%u bound2d=%u",
                          target, g_active_texture_unit, CurrentBoundTexture2D());
        glGenerateMipmap(target);
        const GLenum error = glGetError();
        PigletLastGlTrace("[EXECUTOR_PIGLET_LAST_GL] mipmap api=glGenerateMipmap "
                          "stage=GL_CALL_RETURN target=0x%x active=%u bound2d=%u error=0x%x",
                          target, g_active_texture_unit, CurrentBoundTexture2D(), error);
    }
}
void PS4_SYSV_ABI GlGetIntegerv(GLenum pname, GLint* data) {
    if (EnsureContext()) glGetIntegerv(pname, data);
}
void PS4_SYSV_ABI GlReadPixels(GLint x, GLint y, GLsizei width, GLsizei height, GLenum format,
                               GLenum type, void* pixels) {
    if (EnsureContext()) glReadPixels(x, y, width, height, format, type, pixels);
}

}

void SetGuestSwapExpected(const bool enabled, const char* reason) {
    g_guest_egl_controls_swap = enabled;
    g_itemzflow_overlay_enabled =
        enabled && reason != nullptr &&
        (std::strstr(reason, "itemzflow") != nullptr ||
         std::strstr(reason, "app0_probe") != nullptr);
    if (g_guest_egl_swap_control_log_budget-- > 0) {
        Log("[EXECUTOR_PIGLET] guest swap expectation enabled=%d reason=%s",
            enabled ? 1 : 0, reason ? reason : "unspecified");
    }
}

void* ResolveSymbolByName(const char* name) {
    if (name == nullptr || name[0] == '\0') {
        return nullptr;
    }

#define PIGLET_NAME(symbol, function)                                                             \
    if (std::strcmp(name, symbol) == 0) {                                                          \
        return reinterpret_cast<void*>(Core::AeroLib::GetAndroidX64HleStubForNative(               \
            symbol, reinterpret_cast<u64>(&function)));                                            \
    }

    PIGLET_NAME("eglGetDisplay", EglGetDisplay);
    PIGLET_NAME("eglInitialize", EglInitialize);
    PIGLET_NAME("eglChooseConfig", EglChooseConfig);
    PIGLET_NAME("eglCreateWindowSurface", EglCreateWindowSurface);
    PIGLET_NAME("eglCreatePbufferSurface", EglCreatePbufferSurface);
    PIGLET_NAME("eglCreatePixmapSurface", EglCreatePixmapSurface);
    PIGLET_NAME("eglCreatePbufferFromClientBuffer", EglCreatePbufferFromClientBuffer);
    PIGLET_NAME("eglCreateContext", EglCreateContext);
    PIGLET_NAME("eglMakeCurrent", EglMakeCurrent);
    PIGLET_NAME("eglSwapBuffers", EglSwapBuffers);
    PIGLET_NAME("orbisGlSwapBuffers", OrbisGlSwapBuffers);
    PIGLET_NAME("eglDestroySurface", EglDestroySurface);
    PIGLET_NAME("eglDestroyContext", EglDestroyContext);
    PIGLET_NAME("eglBindAPI", EglBindAPI);
    PIGLET_NAME("eglBindTexImage", EglBindTexImage);
    PIGLET_NAME("eglSwapInterval", EglSwapInterval);
    PIGLET_NAME("eglGetConfigAttrib", EglGetConfigAttrib);
    PIGLET_NAME("eglQuerySurface", EglQuerySurface);
    PIGLET_NAME("eglQueryString", EglQueryString);
    PIGLET_NAME("eglGetCurrentDisplay", EglGetCurrentDisplay);
    PIGLET_NAME("eglGetCurrentContext", EglGetCurrentContext);
    PIGLET_NAME("eglGetCurrentSurface", EglGetCurrentSurface);
    PIGLET_NAME("eglTerminate", EglTerminate);
    PIGLET_NAME("eglGetError", EglGetError);
    PIGLET_NAME("eglGetProcAddress", EglGetProcAddress);

    PIGLET_NAME("glClear", GlClear);
    PIGLET_NAME("glClearColor", GlClearColor);
    PIGLET_NAME("glGetError", GlGetError);
    PIGLET_NAME("glGetString", GlGetString);
    PIGLET_NAME("glGetStringi", GlGetStringi);
    PIGLET_NAME("glViewport", GlViewport);
    PIGLET_NAME("glUseProgram", GlUseProgram);
    PIGLET_NAME("glEnable", GlEnable);
    PIGLET_NAME("glDisable", GlDisable);
    PIGLET_NAME("glBlendEquation", GlBlendEquation);
    PIGLET_NAME("glBlendFunc", GlBlendFunc);
    PIGLET_NAME("glActiveTexture", GlActiveTexture);
    PIGLET_NAME("glBindTexture", GlBindTexture);
    PIGLET_NAME("glTexParameteri", GlTexParameteri);
    PIGLET_NAME("glTexParameterf", GlTexParameterf);
    PIGLET_NAME("glTexImage2D", GlTexImage2D);
    PIGLET_NAME("glTexSubImage2D", GlTexSubImage2D);
    PIGLET_NAME("glGenTextures", GlGenTextures);
    PIGLET_NAME("glDeleteTextures", GlDeleteTextures);
    PIGLET_NAME("glGenBuffers", GlGenBuffers);
    PIGLET_NAME("glDeleteBuffers", GlDeleteBuffers);
    PIGLET_NAME("glBindBuffer", GlBindBuffer);
    PIGLET_NAME("glBufferData", GlBufferData);
    PIGLET_NAME("glBufferSubData", GlBufferSubData);
    PIGLET_NAME("glVertexAttribPointer", GlVertexAttribPointer);
    PIGLET_NAME("glEnableVertexAttribArray", GlEnableVertexAttribArray);
    PIGLET_NAME("glDisableVertexAttribArray", GlDisableVertexAttribArray);
    PIGLET_NAME("glDrawArrays", GlDrawArrays);
    PIGLET_NAME("glDrawElements", GlDrawElements);
    PIGLET_NAME("glCreateShader", GlCreateShader);
    PIGLET_NAME("glShaderSource", GlShaderSource);
    PIGLET_NAME("glShaderBinary", GlShaderBinary);
    PIGLET_NAME("glCompileShader", GlCompileShader);
    PIGLET_NAME("glGetShaderiv", GlGetShaderiv);
    PIGLET_NAME("glGetShaderInfoLog", GlGetShaderInfoLog);
    PIGLET_NAME("glDeleteShader", GlDeleteShader);
    PIGLET_NAME("glCreateProgram", GlCreateProgram);
    PIGLET_NAME("glAttachShader", GlAttachShader);
    PIGLET_NAME("glLinkProgram", GlLinkProgram);
    PIGLET_NAME("glGetProgramiv", GlGetProgramiv);
    PIGLET_NAME("glDeleteProgram", GlDeleteProgram);
    PIGLET_NAME("glGetUniformLocation", GlGetUniformLocation);
    PIGLET_NAME("glGetAttribLocation", GlGetAttribLocation);
    PIGLET_NAME("glUniform1f", GlUniform1f);
    PIGLET_NAME("glUniform2f", GlUniform2f);
    PIGLET_NAME("glUniform4f", GlUniform4f);
    PIGLET_NAME("glUniform1i", GlUniform1i);
    PIGLET_NAME("glUniformMatrix4fv", GlUniformMatrix4fv);
    PIGLET_NAME("glGenVertexArrays", GlGenVertexArrays);
    PIGLET_NAME("glBindVertexArray", GlBindVertexArray);
    PIGLET_NAME("glDeleteVertexArrays", GlDeleteVertexArrays);
    PIGLET_NAME("glGenFramebuffers", GlGenFramebuffers);
    PIGLET_NAME("glDeleteFramebuffers", GlDeleteFramebuffers);
    PIGLET_NAME("glBindFramebuffer", GlBindFramebuffer);
    PIGLET_NAME("glGenRenderbuffers", GlGenRenderbuffers);
    PIGLET_NAME("glBindRenderbuffer", GlBindRenderbuffer);
    PIGLET_NAME("glRenderbufferStorage", GlRenderbufferStorage);
    PIGLET_NAME("glFramebufferTexture2D", GlFramebufferTexture2D);
    PIGLET_NAME("glFramebufferRenderbuffer", GlFramebufferRenderbuffer);
    PIGLET_NAME("glCheckFramebufferStatus", GlCheckFramebufferStatus);
    PIGLET_NAME("glGenerateMipmap", GlGenerateMipmap);
    PIGLET_NAME("glGetIntegerv", GlGetIntegerv);
    PIGLET_NAME("glReadPixels", GlReadPixels);

#undef PIGLET_NAME

    Log("[EXECUTOR_PIGLET] unresolved dynamic symbol %s", name);
    return nullptr;
}

#endif

#ifndef __ANDROID__
void* ResolveSymbolByName(const char*) {
    return nullptr;
}
#endif

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
#ifdef __ANDROID__
#define PIGLET(nid, function)                                                                      \
    LIB_FUNCTION(nid, "libScePigletv2VSH", 1, "libScePigletv2VSH", function)

    PIGLET("0oZKK6QmW-k", EglCreateContext);
    PIGLET("1Ml5nNkeImU", EglGetProcAddress);
    PIGLET("51YpII41z34", EglSwapBuffers);
    PIGLET("9TB8RLYXg+g", EglDestroySurface);
    PIGLET("D947DWfj9tY", EglMakeCurrent);
    PIGLET("EdefUaKYEPE", EglBindTexImage);
    PIGLET("1h8m1KuIR9o", EglSwapInterval);
    PIGLET("5O92iRi6Hk4", EglGetConfigAttrib);
    PIGLET("SR9U4-+eibc", EglQuerySurface);
    PIGLET("mwSvDt0dHF8", EglQueryString);
    PIGLET("13Y3oJ7jRHQ", EglGetCurrentDisplay);
    PIGLET("7Nm1o42CbJ8", EglGetCurrentContext);
    PIGLET("LP5RvgCb3AA", EglGetCurrentSurface);
    PIGLET("ConU-nXswn8", EglTerminate);
    PIGLET("GGIG9nj2zXk", EglCreatePbufferSurface);
    PIGLET("KJmeNzmA6UM", EglInitialize);
    PIGLET("PSqt1KLK7o0", EglCreateWindowSurface);
    PIGLET("Pu5Yd9+FY9Y", EglGetError);
    PIGLET("RqcWVDCXZcY", EglCreatePbufferFromClientBuffer);
    PIGLET("Z7pNCsK2dAs", EglGetDisplay);
    PIGLET("fesoNJCZpSA", EglChooseConfig);
    PIGLET("tcVQcBMJato", EglCreatePixmapSurface);
    PIGLET("vrBLBaiUr5g", EglDestroyContext);
    PIGLET("yCo3HzKQSts", EglBindAPI);

    PIGLET("3gtNzvkq-XY", ScePigletZero);
    PIGLET("6u98-pOEZ7A", ScePigletZero);
    PIGLET("M9RtXpjSYtE", ScePigletTrue);
    PIGLET("SL6AQAnM5WU", ScePigletTrue);
    PIGLET("oWhbxCJiBMM", ScePigletZero);

    PIGLET("KOJ4+zzhpAg", GlClear);
    PIGLET("Pkhe5Qcq++0", GlClearColor);
    PIGLET("uyjjXGqy-Ek", GlGetError);
    PIGLET("xqbiS7PMrSE", GlGetString);
    PIGLET("OOI5nqLihBw", GlGetStringi);
    PIGLET("idWUOMHlXf0", GlViewport);
    PIGLET("ADsMyXKjHVo", GlUseProgram);
    PIGLET("TIdn+yGbJlo", GlEnable);
    PIGLET("pwl29nuNnqI", GlDisable);
    PIGLET("HFIXBmlQmXI", GlBlendEquation);
    PIGLET("39BK09X7zLY", GlBlendFunc);
    PIGLET("y20mWUHGzj8", GlActiveTexture);
    PIGLET("I9IXZ61WkD0", GlBindTexture);
    PIGLET("Kzbvd9G7ieE", GlTexParameteri);
    PIGLET("1dtq0xFWSNU", GlTexParameterf);
    PIGLET("52+a3D15aw4", GlTexImage2D);
    PIGLET("3DK2dxDedi0", GlTexSubImage2D);
    PIGLET("OR6-CZL+oOw", GlGenTextures);
    PIGLET("tZ1lckB5o2A", GlDeleteTextures);
    PIGLET("4p8IUn5CM0I", GlGenBuffers);
    PIGLET("mPRFA5CE48g", GlDeleteBuffers);
    PIGLET("nl8lQxwAceM", GlBindBuffer);
    PIGLET("hKN5OJIT5ko", GlBufferData);
    PIGLET("9rH9n8530IU", GlBufferSubData);
    PIGLET("Yk0ofesYP1U", GlVertexAttribPointer);
    PIGLET("sA3IshgN8u8", GlEnableVertexAttribArray);
    PIGLET("BpP68iXiK5I", GlDisableVertexAttribArray);
    PIGLET("19N0joauSsE", GlDrawArrays);
    PIGLET("Xe-vzCJeV3Y", GlDrawElements);
    PIGLET("0TyJ6AxRs5w", GlCreateShader);
    PIGLET("2PuCkkEw9eE", GlShaderSource);
    PIGLET("yFJY7Te6ZRQ", GlShaderBinary);
    PIGLET("zyebgmQUnWY", GlCompileShader);
    PIGLET("sfRsq9Yq2Y8", GlGetShaderiv);
    PIGLET("Yo4iw4JYCTI", GlGetShaderInfoLog);
    PIGLET("r5IsX1dPwUk", GlDeleteShader);
    PIGLET("q0iQ6QttHjQ", GlCreateProgram);
    PIGLET("uVdONJ4yNrc", GlAttachShader);
    PIGLET("ON26vhraMMU", GlLinkProgram);
    PIGLET("m-7cQfah6KY", GlGetProgramiv);
    PIGLET("oOigDJBsN2c", GlDeleteProgram);
    PIGLET("UMeDmpCpF40", GlGetUniformLocation);
    PIGLET("beY9ZJv5-dU", GlGetAttribLocation);
    PIGLET("POq8+5-inHY", GlUniform1f);
    PIGLET("e5HOpV5feq8", GlUniform2f);
    PIGLET("H8Dp8lwh8QQ", GlUniform4f);
    PIGLET("hR59aDWzHPw", GlUniform1i);
    PIGLET("v8-KyLvjWKw", GlUniformMatrix4fv);
    PIGLET("Ai5yrljUSjI", GlGenVertexArrays);
    PIGLET("f8f0+hYyOb0", GlBindVertexArray);
    PIGLET("En+dYUM8u7U", GlDeleteVertexArrays);
    PIGLET("6OfeCf-u4bg", GlGenFramebuffers);
    PIGLET("3xsE3RDNnQw", GlDeleteFramebuffers);
    PIGLET("vWkaB9NZpGA", GlBindFramebuffer);
    PIGLET("ZggLwqNsGIk", GlGenRenderbuffers);
    PIGLET("lGJTnZqchvA", GlBindRenderbuffer);
    PIGLET("2jKVW2uBSw8", GlRenderbufferStorage);
    PIGLET("326-26W1Ht4", GlFramebufferTexture2D);
    PIGLET("YjT8GrH5bsM", GlFramebufferRenderbuffer);
    PIGLET("-HAzeV7vL-Q", GlCheckFramebufferStatus);
    PIGLET("nmUsbuL3Rc0", GlGenerateMipmap);
    PIGLET("-CyF1XyqmaQ", GlGetIntegerv);
    PIGLET("tPHxFS+55pY", GlReadPixels);

#undef PIGLET
#endif
}

}
