/////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Tencent is pleased to support the open source community by making libpag available.
//
//  Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
//  except in compliance with the License. You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  unless required by applicable law or agreed to in writing, software distributed under the
//  license is distributed on an "as is" basis, without warranties or conditions of any kind,
//  either express or implied. see the license for the specific language governing permissions
//  and limitations under the license.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#include "tgfx/gpu/opengl/GLYUVTexture.h"
#include "GLUtil.h"
#include "core/utils/UniqueID.h"
#include "gpu/Gpu.h"

namespace tgfx {
#define I420_PLANE_COUNT 3
#define NV12_PLANE_COUNT 2

struct YUVConfig {
  YUVConfig(YUVColorSpace colorSpace, YUVColorRange colorRange, int width, int height,
            int planeCount)
      : colorSpace(colorSpace),
        colorRange(colorRange),
        width(width),
        height(height),
        planeCount(planeCount) {
  }
  YUVColorSpace colorSpace;
  YUVColorRange colorRange;
  int width = 0;
  int height = 0;
  uint8_t* pixelsPlane[3]{};
  int rowBytes[3]{};
  PixelFormat formats[3]{};
  int planeCount = 0;
};

class GLI420Texture : public GLYUVTexture {
 public:
  static void ComputeRecycleKey(BytesKey* recycleKey, int width, int height) {
    static const uint32_t I420Type = UniqueID::Next();
    recycleKey->write(I420Type);
    recycleKey->write(static_cast<uint32_t>(width));
    recycleKey->write(static_cast<uint32_t>(height));
  }

  GLI420Texture(YUVColorSpace colorSpace, YUVColorRange colorRange, int width, int height)
      : GLYUVTexture(colorSpace, colorRange, width, height) {
  }

  YUVPixelFormat pixelFormat() const override {
    return YUVPixelFormat::I420;
  }

 protected:
  void computeRecycleKey(BytesKey* recycleKey) const override {
    ComputeRecycleKey(recycleKey, width(), height());
  }
};

class GLNV12Texture : public GLYUVTexture {
 public:
  static void ComputeRecycleKey(BytesKey* recycleKey, int width, int height) {
    static const uint32_t NV12Type = UniqueID::Next();
    recycleKey->write(NV12Type);
    recycleKey->write(static_cast<uint32_t>(width));
    recycleKey->write(static_cast<uint32_t>(height));
  }
  GLNV12Texture(YUVColorSpace colorSpace, YUVColorRange colorRange, int width, int height)
      : GLYUVTexture(colorSpace, colorRange, width, height) {
  }

  YUVPixelFormat pixelFormat() const override {
    return YUVPixelFormat::NV12;
  }

 protected:
  void computeRecycleKey(BytesKey* recycleKey) const override {
    ComputeRecycleKey(recycleKey, width(), height());
  }
};

static std::vector<GLSampler> MakeTexturePlanes(Context* context, const YUVConfig& yuvConfig) {
  std::vector<GLSampler> texturePlanes{};
  static constexpr int factor[] = {0, 1, 1};
  for (int index = 0; index < yuvConfig.planeCount; index++) {
    auto w = yuvConfig.width >> factor[index];
    auto h = yuvConfig.height >> factor[index];
    auto sampler = context->gpu()->createTexture(w, h, yuvConfig.formats[index]);
    if (sampler == nullptr) {
      for (auto& glSampler : texturePlanes) {
        context->gpu()->deleteTexture(&glSampler);
      }
      return {};
    }
    texturePlanes.emplace_back(*static_cast<GLSampler*>(sampler.get()));
  }
  return texturePlanes;
}

static void SubmitYUVTexture(Context* context, const YUVConfig& yuvConfig,
                             const GLSampler yuvTextures[]) {
  static constexpr int factor[] = {0, 1, 1};
  for (int index = 0; index < yuvConfig.planeCount; index++) {
    const auto& sampler = yuvTextures[index];
    auto w = yuvConfig.width >> factor[index];
    auto h = yuvConfig.height >> factor[index];
    auto rowBytes = yuvConfig.rowBytes[index];
    auto pixels = yuvConfig.pixelsPlane[index];
    context->gpu()->writePixels(
        &sampler, Rect::MakeWH(static_cast<float>(w), static_cast<float>(h)), pixels, rowBytes);
  }
}

std::shared_ptr<YUVTexture> YUVTexture::MakeI420(Context* context, YUVColorSpace colorSpace,
                                                 YUVColorRange colorRange, int width, int height,
                                                 uint8_t* pixelsPlane[3], const int lineSize[3]) {
  YUVConfig yuvConfig = YUVConfig(colorSpace, colorRange, width, height, I420_PLANE_COUNT);
  for (int i = 0; i < 3; i++) {
    yuvConfig.pixelsPlane[i] = pixelsPlane[i];
    yuvConfig.rowBytes[i] = lineSize[i];
    yuvConfig.formats[i] = PixelFormat::GRAY_8;
  }

  BytesKey recycleKey = {};
  GLI420Texture::ComputeRecycleKey(&recycleKey, width, height);
  auto texture =
      std::static_pointer_cast<GLYUVTexture>(context->resourceCache()->getRecycled(recycleKey));
  if (texture == nullptr) {
    auto texturePlanes = MakeTexturePlanes(context, yuvConfig);
    if (texturePlanes.empty()) {
      return nullptr;
    }
    texture = std::static_pointer_cast<GLYUVTexture>(
        Resource::Wrap(context, new GLI420Texture(yuvConfig.colorSpace, yuvConfig.colorRange,
                                                  yuvConfig.width, yuvConfig.height)));
    texture->samplers = texturePlanes;
  }
  SubmitYUVTexture(context, yuvConfig, &texture->samplers[0]);
  return texture;
}

std::shared_ptr<YUVTexture> YUVTexture::MakeNV12(Context* context, YUVColorSpace colorSpace,
                                                 YUVColorRange colorRange, int width, int height,
                                                 uint8_t* pixelsPlane[2], const int lineSize[2]) {
  YUVConfig yuvConfig = YUVConfig(colorSpace, colorRange, width, height, NV12_PLANE_COUNT);
  for (int i = 0; i < 2; i++) {
    yuvConfig.pixelsPlane[i] = pixelsPlane[i];
    yuvConfig.rowBytes[i] = lineSize[i];
  }
  yuvConfig.formats[0] = PixelFormat::GRAY_8;
  yuvConfig.formats[1] = PixelFormat::RG_88;

  BytesKey recycleKey = {};
  GLNV12Texture::ComputeRecycleKey(&recycleKey, width, height);
  auto texture =
      std::static_pointer_cast<GLYUVTexture>(context->resourceCache()->getRecycled(recycleKey));
  if (texture == nullptr) {
    auto texturePlanes = MakeTexturePlanes(context, yuvConfig);
    if (texturePlanes.empty()) {
      return nullptr;
    }
    texture = std::static_pointer_cast<GLYUVTexture>(
        Resource::Wrap(context, new GLNV12Texture(yuvConfig.colorSpace, yuvConfig.colorRange,
                                                  yuvConfig.width, yuvConfig.height)));
    texture->samplers = texturePlanes;
  }
  SubmitYUVTexture(context, yuvConfig, &texture->samplers[0]);
  return texture;
}

GLYUVTexture::GLYUVTexture(YUVColorSpace colorSpace, YUVColorRange colorRange, int width,
                           int height)
    : YUVTexture(colorSpace, colorRange, width, height) {
}

Point GLYUVTexture::getTextureCoord(float x, float y) const {
  return {x / static_cast<float>(width()), y / static_cast<float>(height())};
}

const TextureSampler* GLYUVTexture::getSamplerAt(size_t index) const {
  if (index >= samplers.size()) {
    return nullptr;
  }
  return &samplers[index];
}

void GLYUVTexture::onReleaseGPU() {
  for (auto& sampler : samplers) {
    context->gpu()->deleteTexture(&sampler);
  }
}
}  // namespace tgfx
