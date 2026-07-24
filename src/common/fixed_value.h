// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

template <typename T, T Value>
class FixedValue {
    T m_value{Value};

public:
    constexpr FixedValue() = default;

    constexpr explicit(false) operator T() const {
        return m_value;
    }

    FixedValue& operator=(const T&) {
        m_value = Value;
        return *this;
    }
    FixedValue& operator=(T&&) noexcept {
        m_value = {Value};
        return *this;
    }
};
