//  SPDX-FileCopyrightText: 2024 Dear ImGui Club Contributors
//  SPDX-License-Identifier: MIT

// Licensed under The MIT License (MIT)

#pragma once

#include <stdint.h>
#include <stdio.h>

#if defined(_MSC_VER) || defined(_UCRT)
#define _PRISizeT "I"
#define ImSnprintf _snprintf
#else
#define _PRISizeT "z"
#define ImSnprintf snprintf
#endif

#if defined(_MSC_VER) || defined(_UCRT)
#pragma warning(push)
#pragma warning(                                                                                   \
    disable : 4996)
#endif

struct MemoryEditor {
    enum DataFormat {
        DataFormat_Bin = 0,
        DataFormat_Dec = 1,
        DataFormat_Hex = 2,
        DataFormat_COUNT
    };

    bool Open;
    bool ReadOnly;
    int Cols;
    bool OptShowOptions;
    bool OptShowDataPreview;
    bool OptShowHexII;
    bool OptShowAscii;
    bool OptGreyOutZeroes;
    bool OptUpperCaseHex;
    int OptMidColsCount;
    int OptAddrDigitsCount;
    float OptFooterExtraHeight;
    ImU32 HighlightColor;

    ImU8 (*ReadFn)(const ImU8* mem, size_t off, void* user_data);
    void (*WriteFn)(ImU8* mem, size_t off, ImU8 d, void* user_data);
    bool (*HighlightFn)(const ImU8* mem, size_t off, void* user_data);
    ImU32 (*BgColorFn)(const ImU8* mem, size_t off, void* user_data);
    void* UserData;

    bool MouseHovered;
    size_t MouseHoveredAddr;
    bool ContentsWidthChanged;
    size_t DataPreviewAddr;
    size_t DataEditingAddr;
    bool DataEditingTakeFocus;
    char DataInputBuf[32];
    char AddrInputBuf[32];
    size_t GotoAddr;
    size_t HighlightMin, HighlightMax;
    int PreviewEndianness;
    ImGuiDataType PreviewDataType;

    MemoryEditor() {
        Open = true;
        ReadOnly = false;
        Cols = 16;
        OptShowOptions = true;
        OptShowDataPreview = false;
        OptShowHexII = false;
        OptShowAscii = true;
        OptGreyOutZeroes = true;
        OptUpperCaseHex = true;
        OptMidColsCount = 8;
        OptAddrDigitsCount = 0;
        OptFooterExtraHeight = 0.0f;
        HighlightColor = IM_COL32(255, 255, 255, 50);
        ReadFn = nullptr;
        WriteFn = nullptr;
        HighlightFn = nullptr;
        BgColorFn = nullptr;
        UserData = nullptr;

        ContentsWidthChanged = false;
        DataPreviewAddr = DataEditingAddr = (size_t)-1;
        DataEditingTakeFocus = false;
        memset(DataInputBuf, 0, sizeof(DataInputBuf));
        memset(AddrInputBuf, 0, sizeof(AddrInputBuf));
        GotoAddr = (size_t)-1;
        MouseHovered = false;
        MouseHoveredAddr = 0;
        HighlightMin = HighlightMax = (size_t)-1;
        PreviewEndianness = 0;
        PreviewDataType = ImGuiDataType_S32;
    }

    void GotoAddrAndHighlight(size_t addr_min, size_t addr_max) {
        GotoAddr = addr_min;
        HighlightMin = addr_min;
        HighlightMax = addr_max;
    }

    struct Sizes {
        int AddrDigitsCount;
        float LineHeight;
        float GlyphWidth;
        float HexCellWidth;
        float SpacingBetweenMidCols;
        float PosHexStart;
        float PosHexEnd;
        float PosAsciiStart;
        float PosAsciiEnd;
        float WindowWidth;

        Sizes() {
            memset(this, 0, sizeof(*this));
        }
    };

    void CalcSizes(Sizes& s, size_t mem_size, size_t base_display_addr) {
        ImGuiStyle& style = ImGui::GetStyle();
        s.AddrDigitsCount = OptAddrDigitsCount;
        if (s.AddrDigitsCount == 0)
            for (size_t n = base_display_addr + mem_size - 1; n > 0; n >>= 4)
                s.AddrDigitsCount++;
        s.LineHeight = ImGui::GetTextLineHeight();
        s.GlyphWidth = ImGui::CalcTextSize("F").x + 1;
        s.HexCellWidth = (float)(int)(s.GlyphWidth * 2.5f);
        s.SpacingBetweenMidCols = (float)(int)(s.HexCellWidth * 0.25f);
        s.PosHexStart = (s.AddrDigitsCount + 2) * s.GlyphWidth;
        s.PosHexEnd = s.PosHexStart + (s.HexCellWidth * Cols);
        s.PosAsciiStart = s.PosAsciiEnd = s.PosHexEnd;
        if (OptShowAscii) {
            s.PosAsciiStart = s.PosHexEnd + s.GlyphWidth * 1;
            if (OptMidColsCount > 0)
                s.PosAsciiStart += (float)((Cols + OptMidColsCount - 1) / OptMidColsCount) *
                                   s.SpacingBetweenMidCols;
            s.PosAsciiEnd = s.PosAsciiStart + Cols * s.GlyphWidth;
        }
        s.WindowWidth =
            s.PosAsciiEnd + style.ScrollbarSize + style.WindowPadding.x * 2 + s.GlyphWidth;
    }

    void DrawWindow(const char* title, void* mem_data, size_t mem_size,
                    size_t base_display_addr = 0x0000) {
        Sizes s;
        CalcSizes(s, mem_size, base_display_addr);
        ImGui::SetNextWindowSize(ImVec2(s.WindowWidth, s.WindowWidth * 0.60f),
                                 ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSizeConstraints(ImVec2(0.0f, 0.0f), ImVec2(s.WindowWidth, FLT_MAX));

        Open = true;
        if (ImGui::Begin(title, &Open, ImGuiWindowFlags_NoScrollbar)) {
            DrawContents(mem_data, mem_size, base_display_addr);
            if (ContentsWidthChanged) {
                CalcSizes(s, mem_size, base_display_addr);
                ImGui::SetWindowSize(ImVec2(s.WindowWidth, ImGui::GetWindowSize().y));
            }
        }
        ImGui::End();
    }

    void DrawContents(void* mem_data_void, size_t mem_size, size_t base_display_addr = 0x0000) {
        if (Cols < 1)
            Cols = 1;

        ImU8* mem_data = (ImU8*)mem_data_void;
        Sizes s;
        CalcSizes(s, mem_size, base_display_addr);
        ImGuiStyle& style = ImGui::GetStyle();

        const ImVec2 contents_pos_start = ImGui::GetCursorScreenPos();

        const float height_separator = style.ItemSpacing.y;
        float footer_height = OptFooterExtraHeight;
        if (OptShowOptions)
            footer_height += height_separator + ImGui::GetFrameHeightWithSpacing() * 1;
        if (OptShowDataPreview)
            footer_height += height_separator + ImGui::GetFrameHeightWithSpacing() * 1 +
                             ImGui::GetTextLineHeightWithSpacing() * 3;
        ImGui::BeginChild("##scrolling", ImVec2(-FLT_MIN, -footer_height), ImGuiChildFlags_None,
                          ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoNav);
        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));

        const int line_total_count = (int)((mem_size + Cols - 1) / Cols);
        ImGuiListClipper clipper;
        clipper.Begin(line_total_count, s.LineHeight);

        bool data_next = false;

        if (DataEditingAddr >= mem_size)
            DataEditingAddr = (size_t)-1;
        if (DataPreviewAddr >= mem_size)
            DataPreviewAddr = (size_t)-1;

        size_t preview_data_type_size = OptShowDataPreview ? DataTypeGetSize(PreviewDataType) : 0;

        size_t data_editing_addr_next = (size_t)-1;
        if (DataEditingAddr != (size_t)-1) {
            if (ImGui::IsKeyPressed(ImGuiKey_UpArrow) &&
                (ptrdiff_t)DataEditingAddr >= (ptrdiff_t)Cols) {
                data_editing_addr_next = DataEditingAddr - Cols;
            } else if (ImGui::IsKeyPressed(ImGuiKey_DownArrow) &&
                       (ptrdiff_t)DataEditingAddr < (ptrdiff_t)mem_size - Cols) {
                data_editing_addr_next = DataEditingAddr + Cols;
            } else if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow) &&
                       (ptrdiff_t)DataEditingAddr > (ptrdiff_t)0) {
                data_editing_addr_next = DataEditingAddr - 1;
            } else if (ImGui::IsKeyPressed(ImGuiKey_RightArrow) &&
                       (ptrdiff_t)DataEditingAddr < (ptrdiff_t)mem_size - 1) {
                data_editing_addr_next = DataEditingAddr + 1;
            }
        }

        ImVec2 window_pos = ImGui::GetWindowPos();
        if (OptShowAscii)
            draw_list->AddLine(
                ImVec2(window_pos.x + s.PosAsciiStart - s.GlyphWidth, window_pos.y),
                ImVec2(window_pos.x + s.PosAsciiStart - s.GlyphWidth, window_pos.y + 9999),
                ImGui::GetColorU32(ImGuiCol_Border));

        const ImU32 color_text = ImGui::GetColorU32(ImGuiCol_Text);
        const ImU32 color_disabled =
            OptGreyOutZeroes ? ImGui::GetColorU32(ImGuiCol_TextDisabled) : color_text;

        const char* format_address =
            OptUpperCaseHex ? "%0*" _PRISizeT "X: " : "%0*" _PRISizeT "x: ";
        const char* format_data = OptUpperCaseHex ? "%0*" _PRISizeT "X" : "%0*" _PRISizeT "x";
        const char* format_byte = OptUpperCaseHex ? "%02X" : "%02x";
        const char* format_byte_space = OptUpperCaseHex ? "%02X " : "%02x ";

        MouseHovered = false;
        MouseHoveredAddr = 0;

        while (clipper.Step())
            for (int line_i = clipper.DisplayStart; line_i < clipper.DisplayEnd; line_i++)
            {
                size_t addr = (size_t)line_i * Cols;
                ImGui::Text(format_address, s.AddrDigitsCount, base_display_addr + addr);

                for (int n = 0; n < Cols && addr < mem_size; n++, addr++) {
                    float byte_pos_x = s.PosHexStart + s.HexCellWidth * n;
                    if (OptMidColsCount > 0)
                        byte_pos_x += (float)(n / OptMidColsCount) * s.SpacingBetweenMidCols;
                    ImGui::SameLine(byte_pos_x);

                    const bool is_highlight_from_user_range =
                        (addr >= HighlightMin && addr < HighlightMax);
                    const bool is_highlight_from_user_func =
                        (HighlightFn && HighlightFn(mem_data, addr, UserData));
                    const bool is_highlight_from_preview =
                        (addr >= DataPreviewAddr &&
                         addr < DataPreviewAddr + preview_data_type_size);

                    ImU32 bg_color = 0;
                    bool is_next_byte_highlighted = false;
                    if (is_highlight_from_user_range || is_highlight_from_user_func ||
                        is_highlight_from_preview) {
                        is_next_byte_highlighted =
                            (addr + 1 < mem_size) &&
                            ((HighlightMax != (size_t)-1 && addr + 1 < HighlightMax) ||
                             (HighlightFn && HighlightFn(mem_data, addr + 1, UserData)) ||
                             (addr + 1 < DataPreviewAddr + preview_data_type_size));
                        bg_color = HighlightColor;
                    } else if (BgColorFn != nullptr) {
                        is_next_byte_highlighted =
                            (addr + 1 < mem_size) &&
                            ((BgColorFn(mem_data, addr + 1, UserData) & IM_COL32_A_MASK) != 0);
                        bg_color = BgColorFn(mem_data, addr, UserData);
                    }
                    if (bg_color != 0) {
                        float bg_width = s.GlyphWidth * 2;
                        if (is_next_byte_highlighted || (n + 1 == Cols)) {
                            bg_width = s.HexCellWidth;
                            if (OptMidColsCount > 0 && n > 0 && (n + 1) < Cols &&
                                ((n + 1) % OptMidColsCount) == 0)
                                bg_width += s.SpacingBetweenMidCols;
                        }
                        ImVec2 pos = ImGui::GetCursorScreenPos();
                        draw_list->AddRectFilled(
                            pos, ImVec2(pos.x + bg_width, pos.y + s.LineHeight), bg_color);
                    }

                    if (DataEditingAddr == addr) {
                        bool data_write = false;
                        ImGui::PushID((void*)addr);
                        if (DataEditingTakeFocus) {
                            ImGui::SetKeyboardFocusHere(0);
                            ImSnprintf(AddrInputBuf, 32, format_data, s.AddrDigitsCount,
                                       base_display_addr + addr);
                            ImSnprintf(DataInputBuf, 32, format_byte,
                                       ReadFn ? ReadFn(mem_data, addr, UserData) : mem_data[addr]);
                        }
                        struct InputTextUserData {
                            static int Callback(ImGuiInputTextCallbackData* data) {
                                InputTextUserData* user_data = (InputTextUserData*)data->UserData;
                                if (!data->HasSelection())
                                    user_data->CursorPos = data->CursorPos;
#if IMGUI_VERSION_NUM < 19102
                                if (data->Flags & ImGuiInputTextFlags_ReadOnly)
                                    return 0;
#endif
                                if (data->SelectionStart == 0 &&
                                    data->SelectionEnd == data->BufTextLen) {
                                    data->DeleteChars(0, data->BufTextLen);
                                    data->InsertChars(0, user_data->CurrentBufOverwrite);
                                    data->SelectionStart = 0;
                                    data->SelectionEnd = 2;
                                    data->CursorPos = 0;
                                }
                                return 0;
                            }
                            char CurrentBufOverwrite[3];
                            int CursorPos;
                        };
                        InputTextUserData input_text_user_data;
                        input_text_user_data.CursorPos = -1;
                        ImSnprintf(input_text_user_data.CurrentBufOverwrite, 3, format_byte,
                                   ReadFn ? ReadFn(mem_data, addr, UserData) : mem_data[addr]);
                        ImGuiInputTextFlags flags = ImGuiInputTextFlags_CharsHexadecimal |
                                                    ImGuiInputTextFlags_EnterReturnsTrue |
                                                    ImGuiInputTextFlags_AutoSelectAll |
                                                    ImGuiInputTextFlags_NoHorizontalScroll |
                                                    ImGuiInputTextFlags_CallbackAlways;
                        if (ReadOnly)
                            flags |= ImGuiInputTextFlags_ReadOnly;
                        flags |=
                        ImGuiInputTextFlags_AlwaysOverwrite;
                        ImGui::SetNextItemWidth(s.GlyphWidth * 2);
                        if (ImGui::InputText("##data", DataInputBuf, IM_ARRAYSIZE(DataInputBuf),
                                             flags, InputTextUserData::Callback,
                                             &input_text_user_data))
                            data_write = data_next = true;
                        else if (!DataEditingTakeFocus && !ImGui::IsItemActive())
                            DataEditingAddr = data_editing_addr_next = (size_t)-1;
                        DataEditingTakeFocus = false;
                        if (input_text_user_data.CursorPos >= 2)
                            data_write = data_next = true;
                        if (data_editing_addr_next != (size_t)-1)
                            data_write = data_next = false;
                        u32 data_input_value = 0;
                        if (!ReadOnly && data_write &&
                            sscanf(DataInputBuf, "%X", &data_input_value) == 1) {
                            if (WriteFn)
                                WriteFn(mem_data, addr, (ImU8)data_input_value, UserData);
                            else
                                mem_data[addr] = (ImU8)data_input_value;
                        }
                        ImGui::PopID();
                    } else {
                        ImU8 b = ReadFn ? ReadFn(mem_data, addr, UserData) : mem_data[addr];

                        if (OptShowHexII) {
                            if ((b >= 32 && b < 128))
                                ImGui::Text(".%c ", b);
                            else if (b == 0xFF && OptGreyOutZeroes)
                                ImGui::TextDisabled("## ");
                            else if (b == 0x00)
                                ImGui::Text("   ");
                            else
                                ImGui::Text(format_byte_space, b);
                        } else {
                            if (b == 0 && OptGreyOutZeroes)
                                ImGui::TextDisabled("00 ");
                            else
                                ImGui::Text(format_byte_space, b);
                        }
                        if (ImGui::IsItemHovered()) {
                            MouseHovered = true;
                            MouseHoveredAddr = addr;
                            if (ImGui::IsMouseClicked(0)) {
                                DataEditingTakeFocus = true;
                                data_editing_addr_next = addr;
                            }
                        }
                    }
                }

                if (OptShowAscii) {
                    ImGui::SameLine(s.PosAsciiStart);
                    ImVec2 pos = ImGui::GetCursorScreenPos();
                    addr = (size_t)line_i * Cols;

                    const float mouse_off_x = ImGui::GetIO().MousePos.x - pos.x;
                    const size_t mouse_addr =
                        (mouse_off_x >= 0.0f && mouse_off_x < s.PosAsciiEnd - s.PosAsciiStart)
                            ? addr + (size_t)(mouse_off_x / s.GlyphWidth)
                            : (size_t)-1;

                    ImGui::PushID(line_i);
                    if (ImGui::InvisibleButton(
                            "ascii", ImVec2(s.PosAsciiEnd - s.PosAsciiStart, s.LineHeight))) {
                        DataEditingAddr = DataPreviewAddr = mouse_addr;
                        DataEditingTakeFocus = true;
                    }
                    if (ImGui::IsItemHovered()) {
                        MouseHovered = true;
                        MouseHoveredAddr = mouse_addr;
                    }
                    ImGui::PopID();
                    for (int n = 0; n < Cols && addr < mem_size; n++, addr++) {
                        if (addr == DataEditingAddr) {
                            draw_list->AddRectFilled(
                                pos, ImVec2(pos.x + s.GlyphWidth, pos.y + s.LineHeight),
                                ImGui::GetColorU32(ImGuiCol_FrameBg));
                            draw_list->AddRectFilled(
                                pos, ImVec2(pos.x + s.GlyphWidth, pos.y + s.LineHeight),
                                ImGui::GetColorU32(ImGuiCol_TextSelectedBg));
                        } else if (BgColorFn) {
                            draw_list->AddRectFilled(
                                pos, ImVec2(pos.x + s.GlyphWidth, pos.y + s.LineHeight),
                                BgColorFn(mem_data, addr, UserData));
                        }
                        unsigned char c =
                            ReadFn ? ReadFn(mem_data, addr, UserData) : mem_data[addr];
                        char display_c = (c < 32 || c >= 128) ? '.' : c;
                        draw_list->AddText(pos, (display_c == c) ? color_text : color_disabled,
                                           &display_c, &display_c + 1);
                        pos.x += s.GlyphWidth;
                    }
                }
            }
        ImGui::PopStyleVar(2);
        const float child_width = ImGui::GetWindowSize().x;
        ImGui::EndChild();

        ImGui::SetCursorPosX(s.WindowWidth);
        ImGui::Dummy(ImVec2(0.0f, 0.0f));

        if (data_next && DataEditingAddr + 1 < mem_size) {
            DataEditingAddr = DataPreviewAddr = DataEditingAddr + 1;
            DataEditingTakeFocus = true;
        } else if (data_editing_addr_next != (size_t)-1) {
            DataEditingAddr = DataPreviewAddr = data_editing_addr_next;
            DataEditingTakeFocus = true;
        }

        const bool lock_show_data_preview = OptShowDataPreview;
        if (OptShowOptions) {
            ImGui::Separator();
            DrawOptionsLine(s, mem_data, mem_size, base_display_addr);
        }

        if (lock_show_data_preview) {
            ImGui::Separator();
            DrawPreviewLine(s, mem_data, mem_size, base_display_addr);
        }

        const ImVec2 contents_pos_end(contents_pos_start.x + child_width,
                                      ImGui::GetCursorScreenPos().y);
        if (OptShowOptions)
            if (ImGui::IsMouseHoveringRect(contents_pos_start, contents_pos_end))
                if (ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) &&
                    ImGui::IsMouseReleased(ImGuiMouseButton_Right))
                    ImGui::OpenPopup("OptionsPopup");

        if (ImGui::BeginPopup("OptionsPopup")) {
            ImGui::SetNextItemWidth(s.GlyphWidth * 7 + style.FramePadding.x * 2.0f);
            if (ImGui::DragInt("##cols", &Cols, 0.2f, 4, 32, "%d cols")) {
                ContentsWidthChanged = true;
                if (Cols < 1)
                    Cols = 1;
            }
            ImGui::Checkbox("Show Data Preview", &OptShowDataPreview);
            ImGui::Checkbox("Show HexII", &OptShowHexII);
            if (ImGui::Checkbox("Show Ascii", &OptShowAscii)) {
                ContentsWidthChanged = true;
            }
            ImGui::Checkbox("Grey out zeroes", &OptGreyOutZeroes);
            ImGui::Checkbox("Uppercase Hex", &OptUpperCaseHex);

            ImGui::EndPopup();
        }
    }

    void DrawOptionsLine(const Sizes& s, void* mem_data, size_t mem_size,
                         size_t base_display_addr) {
        IM_UNUSED(mem_data);
        ImGuiStyle& style = ImGui::GetStyle();
        const char* format_range = OptUpperCaseHex ? "Range %0*" _PRISizeT "X..%0*" _PRISizeT "X"
                                                   : "Range %0*" _PRISizeT "x..%0*" _PRISizeT "x";

        if (ImGui::Button("Options"))
            ImGui::OpenPopup("OptionsPopup");

        ImGui::SameLine();
        ImGui::Text(format_range, s.AddrDigitsCount, base_display_addr, s.AddrDigitsCount,
                    base_display_addr + mem_size - 1);
        ImGui::SameLine();
        ImGui::SetNextItemWidth((s.AddrDigitsCount + 1) * s.GlyphWidth +
                                style.FramePadding.x * 2.0f);
        if (ImGui::InputText("##addr", AddrInputBuf, IM_ARRAYSIZE(AddrInputBuf),
                             ImGuiInputTextFlags_CharsHexadecimal |
                                 ImGuiInputTextFlags_EnterReturnsTrue)) {
            size_t goto_addr;
            if (sscanf(AddrInputBuf, "%" _PRISizeT "X", &goto_addr) == 1) {
                GotoAddr = goto_addr - base_display_addr;
                HighlightMin = HighlightMax = (size_t)-1;
            }
        }

        if (GotoAddr != (size_t)-1) {
            if (GotoAddr < mem_size) {
                ImGui::BeginChild("##scrolling");
                ImGui::SetScrollFromPosY(ImGui::GetCursorStartPos().y +
                                         (GotoAddr / Cols) * ImGui::GetTextLineHeight());
                ImGui::EndChild();
                DataEditingAddr = DataPreviewAddr = GotoAddr;
                DataEditingTakeFocus = true;
            }
            GotoAddr = (size_t)-1;
        }

    }

    void DrawPreviewLine(const Sizes& s, void* mem_data_void, size_t mem_size,
                         size_t base_display_addr) {
        IM_UNUSED(base_display_addr);
        ImU8* mem_data = (ImU8*)mem_data_void;
        ImGuiStyle& style = ImGui::GetStyle();
        ImGui::AlignTextToFramePadding();
        ImGui::Text("Preview as:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth((s.GlyphWidth * 10.0f) + style.FramePadding.x * 2.0f +
                                style.ItemInnerSpacing.x);

        static const ImGuiDataType supported_data_types[] = {
            ImGuiDataType_S8,    ImGuiDataType_U8,    ImGuiDataType_S16, ImGuiDataType_U16,
            ImGuiDataType_S32,   ImGuiDataType_U32,   ImGuiDataType_S64, ImGuiDataType_U64,
            ImGuiDataType_Float, ImGuiDataType_Double};
        if (ImGui::BeginCombo("##combo_type", DataTypeGetDesc(PreviewDataType),
                              ImGuiComboFlags_HeightLargest)) {
            for (int n = 0; n < IM_ARRAYSIZE(supported_data_types); n++) {
                ImGuiDataType data_type = supported_data_types[n];
                if (ImGui::Selectable(DataTypeGetDesc(data_type), PreviewDataType == data_type))
                    PreviewDataType = data_type;
            }
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth((s.GlyphWidth * 6.0f) + style.FramePadding.x * 2.0f +
                                style.ItemInnerSpacing.x);
        ImGui::Combo("##combo_endianness", &PreviewEndianness, "LE\0BE\0\0");

        char buf[128] = "";
        float x = s.GlyphWidth * 6.0f;
        bool has_value = DataPreviewAddr != (size_t)-1;
        if (has_value)
            DrawPreviewData(DataPreviewAddr, mem_data, mem_size, PreviewDataType, DataFormat_Dec,
                            buf, (size_t)IM_ARRAYSIZE(buf));
        ImGui::Text("Dec");
        ImGui::SameLine(x);
        ImGui::TextUnformatted(has_value ? buf : "N/A");
        if (has_value)
            DrawPreviewData(DataPreviewAddr, mem_data, mem_size, PreviewDataType, DataFormat_Hex,
                            buf, (size_t)IM_ARRAYSIZE(buf));
        ImGui::Text("Hex");
        ImGui::SameLine(x);
        ImGui::TextUnformatted(has_value ? buf : "N/A");
        if (has_value)
            DrawPreviewData(DataPreviewAddr, mem_data, mem_size, PreviewDataType, DataFormat_Bin,
                            buf, (size_t)IM_ARRAYSIZE(buf));
        buf[IM_ARRAYSIZE(buf) - 1] = 0;
        ImGui::Text("Bin");
        ImGui::SameLine(x);
        ImGui::TextUnformatted(has_value ? buf : "N/A");
    }

    const char* DataTypeGetDesc(ImGuiDataType data_type) const {
        const char* descs[] = {"Int8",   "Uint8", "Int16",  "Uint16", "Int32",
                               "Uint32", "Int64", "Uint64", "Float",  "Double"};
        IM_ASSERT(data_type >= 0 && data_type < IM_ARRAYSIZE(descs));
        return descs[data_type];
    }

    size_t DataTypeGetSize(ImGuiDataType data_type) const {
        const size_t sizes[] = {1, 1, 2, 2, 4, 4, 8, 8, sizeof(float), sizeof(double)};
        IM_ASSERT(data_type >= 0 && data_type < IM_ARRAYSIZE(sizes));
        return sizes[data_type];
    }

    const char* DataFormatGetDesc(DataFormat data_format) const {
        const char* descs[] = {"Bin", "Dec", "Hex"};
        IM_ASSERT(data_format >= 0 && data_format < DataFormat_COUNT);
        return descs[data_format];
    }

    bool IsBigEndian() const {
        uint16_t x = 1;
        char c[2];
        memcpy(c, &x, 2);
        return c[0] != 0;
    }

    static void* EndiannessCopyBigEndian(void* _dst, void* _src, size_t s, int is_little_endian) {
        if (is_little_endian) {
            uint8_t* dst = (uint8_t*)_dst;
            uint8_t* src = (uint8_t*)_src + s - 1;
            for (int i = 0, n = (int)s; i < n; ++i)
                memcpy(dst++, src--, 1);
            return _dst;
        } else {
            return memcpy(_dst, _src, s);
        }
    }

    static void* EndiannessCopyLittleEndian(void* _dst, void* _src, size_t s,
                                            int is_little_endian) {
        if (is_little_endian) {
            return memcpy(_dst, _src, s);
        } else {
            uint8_t* dst = (uint8_t*)_dst;
            uint8_t* src = (uint8_t*)_src + s - 1;
            for (int i = 0, n = (int)s; i < n; ++i)
                memcpy(dst++, src--, 1);
            return _dst;
        }
    }

    void* EndiannessCopy(void* dst, void* src, size_t size) const {
        static void* (*fp)(void*, void*, size_t, int) = nullptr;
        if (fp == nullptr)
            fp = IsBigEndian() ? EndiannessCopyBigEndian : EndiannessCopyLittleEndian;
        return fp(dst, src, size, PreviewEndianness);
    }

    const char* FormatBinary(const uint8_t* buf, int width) const {
        IM_ASSERT(width <= 64);
        size_t out_n = 0;
        static char out_buf[64 + 8 + 1];
        int n = width / 8;
        for (int j = n - 1; j >= 0; --j) {
            for (int i = 0; i < 8; ++i)
                out_buf[out_n++] = (buf[j] & (1 << (7 - i))) ? '1' : '0';
            out_buf[out_n++] = ' ';
        }
        IM_ASSERT(out_n < IM_ARRAYSIZE(out_buf));
        out_buf[out_n] = 0;
        return out_buf;
    }

    void DrawPreviewData(size_t addr, const ImU8* mem_data, size_t mem_size,
                         ImGuiDataType data_type, DataFormat data_format, char* out_buf,
                         size_t out_buf_size) const {
        uint8_t buf[8];
        size_t elem_size = DataTypeGetSize(data_type);
        size_t size = addr + elem_size > mem_size ? mem_size - addr : elem_size;
        if (ReadFn)
            for (int i = 0, n = (int)size; i < n; ++i)
                buf[i] = ReadFn(mem_data, addr + i, UserData);
        else
            memcpy(buf, mem_data + addr, size);

        if (data_format == DataFormat_Bin) {
            uint8_t binbuf[8];
            EndiannessCopy(binbuf, buf, size);
            ImSnprintf(out_buf, out_buf_size, "%s", FormatBinary(binbuf, (int)size * 8));
            return;
        }

        out_buf[0] = 0;
        switch (data_type) {
        case ImGuiDataType_S8: {
            int8_t data = 0;
            EndiannessCopy(&data, buf, size);
            if (data_format == DataFormat_Dec) {
                ImSnprintf(out_buf, out_buf_size, "%hhd", data);
                return;
            }
            if (data_format == DataFormat_Hex) {
                ImSnprintf(out_buf, out_buf_size, "0x%02x", data & 0xFF);
                return;
            }
            break;
        }
        case ImGuiDataType_U8: {
            uint8_t data = 0;
            EndiannessCopy(&data, buf, size);
            if (data_format == DataFormat_Dec) {
                ImSnprintf(out_buf, out_buf_size, "%hhu", data);
                return;
            }
            if (data_format == DataFormat_Hex) {
                ImSnprintf(out_buf, out_buf_size, "0x%02x", data & 0XFF);
                return;
            }
            break;
        }
        case ImGuiDataType_S16: {
            int16_t data = 0;
            EndiannessCopy(&data, buf, size);
            if (data_format == DataFormat_Dec) {
                ImSnprintf(out_buf, out_buf_size, "%hd", data);
                return;
            }
            if (data_format == DataFormat_Hex) {
                ImSnprintf(out_buf, out_buf_size, "0x%04x", data & 0xFFFF);
                return;
            }
            break;
        }
        case ImGuiDataType_U16: {
            uint16_t data = 0;
            EndiannessCopy(&data, buf, size);
            if (data_format == DataFormat_Dec) {
                ImSnprintf(out_buf, out_buf_size, "%hu", data);
                return;
            }
            if (data_format == DataFormat_Hex) {
                ImSnprintf(out_buf, out_buf_size, "0x%04x", data & 0xFFFF);
                return;
            }
            break;
        }
        case ImGuiDataType_S32: {
            int32_t data = 0;
            EndiannessCopy(&data, buf, size);
            if (data_format == DataFormat_Dec) {
                ImSnprintf(out_buf, out_buf_size, "%d", data);
                return;
            }
            if (data_format == DataFormat_Hex) {
                ImSnprintf(out_buf, out_buf_size, "0x%08x", data);
                return;
            }
            break;
        }
        case ImGuiDataType_U32: {
            uint32_t data = 0;
            EndiannessCopy(&data, buf, size);
            if (data_format == DataFormat_Dec) {
                ImSnprintf(out_buf, out_buf_size, "%u", data);
                return;
            }
            if (data_format == DataFormat_Hex) {
                ImSnprintf(out_buf, out_buf_size, "0x%08x", data);
                return;
            }
            break;
        }
        case ImGuiDataType_S64: {
            int64_t data = 0;
            EndiannessCopy(&data, buf, size);
            if (data_format == DataFormat_Dec) {
                ImSnprintf(out_buf, out_buf_size, "%lld", (long long)data);
                return;
            }
            if (data_format == DataFormat_Hex) {
                ImSnprintf(out_buf, out_buf_size, "0x%016llx", (long long)data);
                return;
            }
            break;
        }
        case ImGuiDataType_U64: {
            uint64_t data = 0;
            EndiannessCopy(&data, buf, size);
            if (data_format == DataFormat_Dec) {
                ImSnprintf(out_buf, out_buf_size, "%llu", (long long)data);
                return;
            }
            if (data_format == DataFormat_Hex) {
                ImSnprintf(out_buf, out_buf_size, "0x%016llx", (long long)data);
                return;
            }
            break;
        }
        case ImGuiDataType_Float: {
            float data = 0.0f;
            EndiannessCopy(&data, buf, size);
            if (data_format == DataFormat_Dec) {
                ImSnprintf(out_buf, out_buf_size, "%f", data);
                return;
            }
            if (data_format == DataFormat_Hex) {
                ImSnprintf(out_buf, out_buf_size, "%a", data);
                return;
            }
            break;
        }
        case ImGuiDataType_Double: {
            double data = 0.0;
            EndiannessCopy(&data, buf, size);
            if (data_format == DataFormat_Dec) {
                ImSnprintf(out_buf, out_buf_size, "%f", data);
                return;
            }
            if (data_format == DataFormat_Hex) {
                ImSnprintf(out_buf, out_buf_size, "%a", data);
                return;
            }
            break;
        }
        default:
        case ImGuiDataType_COUNT:
            break;
        }
        IM_ASSERT(0);
    }
};

#undef _PRISizeT
#undef ImSnprintf

#ifdef _MSC_VER
#pragma warning(pop)
#endif
