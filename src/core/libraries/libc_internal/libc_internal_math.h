// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Core::Loader {
class SymbolsResolver;
}

namespace Libraries::LibcInternal {
double PS4_SYSV_ABI internal_sin(double x);
float PS4_SYSV_ABI internal_sinf(float x);
double PS4_SYSV_ABI internal_cos(double x);
float PS4_SYSV_ABI internal_cosf(float x);
void PS4_SYSV_ABI internal_sincos(double x, double* sinp, double* cosp);
void PS4_SYSV_ABI internal_sincosf(float x, float* sinp, float* cosp);
double PS4_SYSV_ABI internal_tan(double x);
float PS4_SYSV_ABI internal_tanf(float x);
double PS4_SYSV_ABI internal_asin(double x);
float PS4_SYSV_ABI internal_asinf(float x);
double PS4_SYSV_ABI internal_acos(double x);
float PS4_SYSV_ABI internal_acosf(float x);
double PS4_SYSV_ABI internal_atan(double x);
float PS4_SYSV_ABI internal_atanf(float x);
double PS4_SYSV_ABI internal_atan2(double y, double x);
float PS4_SYSV_ABI internal_atan2f(float y, float x);
double PS4_SYSV_ABI internal_exp(double x);
float PS4_SYSV_ABI internal_expf(float x);
double PS4_SYSV_ABI internal_exp2(double x);
float PS4_SYSV_ABI internal_exp2f(float x);
double PS4_SYSV_ABI internal_pow(double x, double y);
float PS4_SYSV_ABI internal_powf(float x, float y);
double PS4_SYSV_ABI internal_log(double x);
float PS4_SYSV_ABI internal_logf(float x);
double PS4_SYSV_ABI internal_log10(double x);
float PS4_SYSV_ABI internal_log10f(float x);
void RegisterlibSceLibcInternalMath(Core::Loader::SymbolsResolver* sym);
}
