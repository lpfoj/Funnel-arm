# SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
# SPDX-License-Identifier: GPL-2.0-or-later

set(LSX4_PS5_DESKTOP_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../src/ps5_desktop"
    CACHE INTERNAL "Funnel isolated PS5 desktop source root")

function(lsx4_define_ps5_desktop_profile)
    if(CMAKE_SCRIPT_MODE_FILE)
        file(GLOB_RECURSE ps5_desktop_sources
            "${LSX4_PS5_DESKTOP_SOURCE_DIR}/*.cpp")
    else()
        file(GLOB_RECURSE ps5_desktop_sources CONFIGURE_DEPENDS
            "${LSX4_PS5_DESKTOP_SOURCE_DIR}/*.cpp")
    endif()

    if(NOT ps5_desktop_sources)
        message(FATAL_ERROR
            "The Funnel PS5 desktop profile contains no C++ implementation")
    endif()

    list(SORT ps5_desktop_sources)
    list(LENGTH ps5_desktop_sources source_count)
    message(STATUS
        "Funnel owns the isolated PS5 desktop profile: ${source_count} C++ sources")

    set(LSX4_PS5_DESKTOP_SOURCES "${ps5_desktop_sources}" PARENT_SCOPE)
endfunction()

function(lsx4_assert_ps5_desktop_ownership)
    foreach(source_path IN LISTS ARGN)
        cmake_path(ABSOLUTE_PATH source_path NORMALIZE
                   OUTPUT_VARIABLE absolute_source)
        cmake_path(IS_PREFIX LSX4_PS5_DESKTOP_SOURCE_DIR
                   "${absolute_source}" NORMALIZE is_ps5_layer_source)
        if(NOT is_ps5_layer_source)
            message(FATAL_ERROR
                "PS5 desktop source escaped its isolated Funnel directory: ${source_path}")
        endif()
    endforeach()
endfunction()
