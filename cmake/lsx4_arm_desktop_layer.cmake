# SPDX-FileCopyrightText: Copyright 2024-2026 LSX4 Project
# SPDX-License-Identifier: GPL-2.0-or-later

set(LSX4_ARM_DESKTOP_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../src"
    CACHE INTERNAL "Funnel ARM-adapted desktop source root")

# Funnel owns the Android composition of the adapted desktop/emulation layer.
# The client repository consumes this profile and must not maintain a second
# source inventory. Files excluded here remain Funnel-owned, but represent
# desktop frontends, unsupported services, or mutually exclusive host backends.
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

    set(profile_sources
        ${common_sources}
        ${core_sources}
        ${imgui_sources}
        ${input_sources}
        ${shader_sources}
        ${video_sources}
        "${LSX4_ARM_DESKTOP_SOURCE_DIR}/emulator.cpp"
        "${LSX4_ARM_DESKTOP_SOURCE_DIR}/sdl_window.cpp")

    set(android_excluded_sources
        # Desktop presence and full-screen UI.
        common/discord_rpc_handler.cpp
        common/logging/log.cpp
        imgui/big_picture/big_picture.cpp
        imgui/big_picture/imgui_impl_sdl3_big_picture.cpp
        imgui/big_picture/imgui_impl_sdlrenderer3.cpp
        imgui/big_picture/settings_dialog_imgui.cpp
        imgui/big_picture/settings_dialog_layer.cpp
        imgui/friends_layer.cpp
        imgui/imgui_translations.cpp
        imgui/invitation_prompt_layer.cpp
        imgui/notifications_layer.cpp
        imgui/renderer/font_data.cpp
        imgui/renderer/font_stack.cpp
        imgui/renderer/generated_fonts/imgui_fonts/notosansjp_regular.ttf.g.cpp
        imgui/renderer/generated_fonts/imgui_fonts/proggyvector_regular.ttf.g.cpp
        imgui/shadnet_notifications_layer.cpp

        # Host alternatives superseded by Android/ARM implementations.
        core/cpu_patches.cpp
        core/libraries/audio/openal_audio_out.cpp
        core/libraries/audio3d/audio3d_openal.cpp
        core/libraries/fiber/fiber_context.cpp
        core/libraries/kernel/threads/stack_asm.cpp
        core/libraries/mouse/sdl_mouse.cpp

        # Desktop configuration, content management, and unsupported services.
        core/emulator_settings.cpp
        core/file_sys/file.cpp
        core/user_manager.cpp
        core/user_settings.cpp
        core/libraries/content_export/content_export.cpp
        core/libraries/invitation_dialog/invitation_dialog.cpp
        core/libraries/invitation_dialog/invitation_dialog_ui.cpp
        core/libraries/kernel/coredump/coredump.cpp
        core/libraries/network/net_obj.cpp
        core/libraries/network/net_upnp.cpp
        core/libraries/usbd/emulated/dimensions.cpp
        core/libraries/usbd/emulated/infinity.cpp
        core/libraries/usbd/emulated/skylander.cpp
        core/libraries/usbd/usbd.cpp
        core/libraries/video_recording/video_recording.cpp

        # Alternative codec paths not selected by the current Android profile.
        core/libraries/ajm/ajm.cpp
        core/libraries/ajm/ajm_aac.cpp
        core/libraries/ajm/ajm_at9.cpp
        core/libraries/ajm/ajm_batch.cpp
        core/libraries/ajm/ajm_context.cpp
        core/libraries/ajm/ajm_instance.cpp
        core/libraries/ajm/ajm_instance_statistics.cpp
        core/libraries/ajm/ajm_mp3.cpp
        core/libraries/videodec/videodec.cpp
        core/libraries/videodec/videodec2.cpp
        core/libraries/videodec/videodec_impl.cpp
        core/libraries/videodec/videodec2_impl.cpp
        core/libraries/font/font_internal.cpp
        core/libraries/font/fontft_internal.cpp
        core/libraries/ime/ime_kb_layout.cpp
        core/libraries/ime/ime_ui_shared.cpp

        # Online NP/ShadNet profile. It remains Funnel-owned and is disabled as
        # one coherent feature until protobuf and Android account UI are enabled.
        core/libraries/np/np_commerce/np_commerce.cpp
        core/libraries/np/np_commerce/np_commerce_ui.cpp
        core/libraries/np/np_handler.cpp
        core/libraries/np/np_matching2/np_matching2.cpp
        core/libraries/np/np_matching2/np_matching2_internal.cpp
        core/libraries/np/np_matching2/np_matching2_mm.cpp
        core/libraries/np/np_matching2/np_matching2_signaling.cpp
        core/libraries/np/np_profile_dialog/np_profile_dialog.cpp
        core/libraries/np/np_profile_dialog/np_profile_dialog_ui.cpp
        core/libraries/np/np_score/np_score.cpp
        core/libraries/np/np_signaling/np_signaling.cpp
        core/libraries/np/np_signaling/np_signaling_helpers.cpp
        core/libraries/np/np_signaling/np_signaling_state.cpp
        core/libraries/np/np_signaling/np_signaling_stubs.cpp
        core/libraries/np/np_web_api/np_web_api.cpp
        core/libraries/np/np_web_api/np_web_api_internal.cpp
        core/libraries/np/np_web_api2/np_web_api2.cpp
        core/libraries/np/np_web_api2/np_web_api2_context.cpp
        core/libraries/np/np_web_api2/np_web_api2_internal.cpp
        core/libraries/np/np_web_api2/np_web_api2_push_event.cpp)

    foreach(relative_source IN LISTS android_excluded_sources)
        set(excluded_source "${LSX4_ARM_DESKTOP_SOURCE_DIR}/${relative_source}")
        if(NOT EXISTS "${excluded_source}")
            message(FATAL_ERROR
                "Funnel Android profile exclusion is stale: ${relative_source}")
        endif()
        list(REMOVE_ITEM profile_sources "${excluded_source}")
    endforeach()

    list(SORT profile_sources)
    list(LENGTH profile_sources profile_source_count)
    message(STATUS
        "Funnel owns the LSX4 Android desktop profile: ${profile_source_count} C++ sources")

    set(LSX4_ARM_DESKTOP_SOURCES "${profile_sources}" PARENT_SCOPE)
    set(LSX4_ARM_SHADER_SOURCES "${shader_sources}" PARENT_SCOPE)
    set(LSX4_ARM_VIDEO_SOURCES "${video_sources}" PARENT_SCOPE)
    set(LSX4_ARM_ANDROID_EXCLUDED_SOURCES "${android_excluded_sources}" PARENT_SCOPE)
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
