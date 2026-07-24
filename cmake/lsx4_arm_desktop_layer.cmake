# SPDX-FileCopyrightText: Copyright 2024-2026 LSX4 Project
# SPDX-License-Identifier: GPL-2.0-or-later

set(LSX4_ARM_DESKTOP_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../src"
    CACHE INTERNAL "Funnel ARM-adapted desktop source root")

function(lsx4_use_arm_desktop_sources variable_name)
    set(resolved_sources)
    foreach(source_path IN LISTS ${variable_name})
        if(source_path MATCHES "^src/")
            string(REGEX REPLACE "^src/" "" relative_path "${source_path}")
            set(layer_path "${LSX4_ARM_DESKTOP_SOURCE_DIR}/${relative_path}")
            if(NOT EXISTS "${layer_path}")
                message(FATAL_ERROR
                    "Funnel ARM desktop source is missing: ${relative_path}")
            endif()
            list(APPEND resolved_sources "${layer_path}")
        else()
            list(APPEND resolved_sources "${source_path}")
        endif()
    endforeach()
    set(${variable_name} "${resolved_sources}" PARENT_SCOPE)
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
