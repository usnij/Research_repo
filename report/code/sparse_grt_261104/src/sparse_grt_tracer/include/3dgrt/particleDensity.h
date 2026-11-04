// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
struct ParticleDensity
{
    float3 position;
    float density;
    float4 quaternion;
    float3 scale;
    float padding;
};

// Gradient storage is separate from the forward model representation.
#ifndef GRT_GRAD_ACCUM_DOUBLE
#define GRT_GRAD_ACCUM_DOUBLE 0
#endif
#if GRT_GRAD_ACCUM_DOUBLE
using GradientScalar = double;
#else
using GradientScalar = float;
#endif
struct GradientVec3 { GradientScalar x, y, z; };
struct GradientVec4 { GradientScalar x, y, z, w; };
struct ParticleDensityGradient {
    GradientVec3 position;
    GradientScalar density;
    GradientVec4 quaternion;
    GradientVec3 scale;
    GradientScalar padding;
};
static_assert(sizeof(GradientVec3) == 3 * sizeof(GradientScalar), "SH gradient stride");
static_assert(sizeof(ParticleDensityGradient) == 12 * sizeof(GradientScalar), "packed gradient stride");
static_assert(sizeof(ParticleDensity) == 12 * sizeof(float), "forward packed layout unchanged");

#if !defined(__CUDACC_RTC__)
#include <stddef.h>
static_assert(offsetof(ParticleDensityGradient, density) == 3 * sizeof(GradientScalar), "density offset");
static_assert(offsetof(ParticleDensityGradient, quaternion) == 4 * sizeof(GradientScalar), "rotation offset");
static_assert(offsetof(ParticleDensityGradient, scale) == 8 * sizeof(GradientScalar), "scale offset");
static_assert(offsetof(ParticleDensityGradient, padding) == 11 * sizeof(GradientScalar), "padding offset");

#endif // host offset checks; NVRTC retains size assertions
