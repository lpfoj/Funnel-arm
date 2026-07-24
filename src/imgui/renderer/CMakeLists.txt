# SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

project(ImGui_Resources)

set(IMGUI_GENERATED_FONTS
    generated_fonts/imgui_fonts/notosansjp_regular.ttf.g.cpp
    generated_fonts/imgui_fonts/proggyvector_regular.ttf.g.cpp
)

add_library(ImGui_Resources STATIC ${IMGUI_GENERATED_FONTS})
set(IMGUI_RESOURCES_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/generated_fonts PARENT_SCOPE)
