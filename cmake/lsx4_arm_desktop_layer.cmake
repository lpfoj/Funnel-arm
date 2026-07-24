# SPDX-FileCopyrightText: Copyright 2024-2026 LSX4 Project
# SPDX-License-Identifier: GPL-2.0-or-later

set(LSX4_ARM_DESKTOP_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../src"
    CACHE INTERNAL "Funnel ARM-adapted desktop source root")

# Funnel is Android/ARM-only. Every C++ implementation under the active source
# roots is part of the profile, except generated font translation units linked
# once through ImGui_Resources.
function(lsx4_define_android_desktop_profile)
    file(GLOB_RECURSE common_sources CONFIGURE_DEPENDS
        "${LSX4_ARM_DESKTOP_SOURCE_DIR}/common/*.cpp")
    file(GLOB_RECURSE core_sources CONFIGURE_DEPENDS
        "${LSX4_ARM_DESKTOP_SOURCE_DIR}/core/*.cpp")
    file(GLOB_RECURSE imgui_sources CONFIGURE_DEPENDS
        "${LSX4_ARM_DESKTOP_SOURCE_DIR}/imgui/*.cpp")
    file(GLOB_RECURSE input_sources CONFIGURE_DEPENDS
        "${LSX4_ARM_DESKTOP_SOURCE_DIR}/input/*.cpp")
    file(GLOB_RECURSE shader_sources CONFIGURE_DEPENDS
        "${LSX4_ARM_DESKTOP_SOURCE_DIR}/shader_recompiler/*.cpp")
    file(GLOB_RECURSE video_sources CONFIGURE_DEPENDS
        "${LSX4_ARM_DESKTOP_SOURCE_DIR}/video_core/*.cpp")

    list(FILTER imgui_sources EXCLUDE REGEX "/renderer/generated_fonts/")

    set(profile_sources
        ${common_sources}
        ${core_sources}
        ${imgui_sources}
        ${input_sources}
        ${shader_sources}
        ${video_sources}
        "${LSX4_ARM_DESKTOP_SOURCE_DIR}/emulator.cpp"
        "${LSX4_ARM_DESKTOP_SOURCE_DIR}/sdl_window.cpp")

    list(SORT profile_sources)
    list(LENGTH profile_sources profile_source_count)
    message(STATUS
        "Funnel owns the LSX4 Android desktop profile: ${profile_source_count} C++ sources")

    set(LSX4_ARM_DESKTOP_SOURCES "${profile_sources}" PARENT_SCOPE)
    set(LSX4_ARM_SHADER_SOURCES "${shader_sources}" PARENT_SCOPE)
    set(LSX4_ARM_VIDEO_SOURCES "${video_sources}" PARENT_SCOPE)
endfunction()

function(lsx4_assert_arm_desktop_ownership)
    foreach(source_path IN LISTS ARGN)
        if(source_path MATCHES "/(common|core|video_core|shader_recompiler|input|imgui)/" OR
           source_path MATCHES "/(emulator|sdl_window)\\.(cpp|h)$")
            cmake_path(ABSOLUTE_PATH source_path NORMALIZE OUTPUT_VARIABLE absolute_source)
            cmake_path(IS_PREFIX CMAKE_BINARY_DIR "${absolute_source}" NORMALIZE is_generated_source)
            if(is_generated_source)
                continue()
            endif()
            cmake_path(IS_PREFIX LSX4_ARM_DESKTOP_SOURCE_DIR "${absolute_source}" NORMALIZE is_layer_source)
            if(NOT is_layer_source)
                message(FATAL_ERROR
                    "Desktop/emulation source escaped Funnel ownership: ${source_path}")
            endif()
        endif()
    endforeach()
endfunction()
