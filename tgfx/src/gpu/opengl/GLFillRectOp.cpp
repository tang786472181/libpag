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

#include "GLFillRectOp.h"

#include "gpu/QuadPerEdgeAAGeometryProcessor.h"

namespace tgfx {
std::unique_ptr<GeometryProcessor> GLFillRectOp::getGeometryProcessor(const DrawArgs& args) {
  return QuadPerEdgeAAGeometryProcessor::Make(
      args.renderTarget->width(), args.renderTarget->height(), args.viewMatrix, args.aa);
}

std::vector<float> GLFillRectOp::vertices(const DrawArgs& args) {
  auto bounds = rects[0];
  auto normalBounds = Rect::MakeLTRB(0, 0, 1, 1);
  // Vertex coordinates are arranged in a 2D pixel coordinate system, and textures are arranged
  // according to a texture coordinate system (0 - 1).
  if (args.aa != AAType::Coverage) {
    std::vector<float> vertexes;
    for (size_t i = 0; i < rects.size(); ++i) {
      auto quad = Quad::MakeFromRect(rects[i], matrices[i]);
      auto localQuad = Quad::MakeFromRect(normalBounds, localMatrices[i]);
      std::vector<float> vert = {
          quad.point(3).x, quad.point(3).y, localQuad.point(3).x, localQuad.point(3).y,
          quad.point(2).x, quad.point(2).y, localQuad.point(2).x, localQuad.point(2).y,
          quad.point(1).x, quad.point(1).y, localQuad.point(1).x, localQuad.point(1).y,
          quad.point(2).x, quad.point(2).y, localQuad.point(2).x, localQuad.point(2).y,
          quad.point(1).x, quad.point(1).y, localQuad.point(1).x, localQuad.point(1).y,
          quad.point(0).x, quad.point(0).y, localQuad.point(0).x, localQuad.point(0).y,
      };
      vertexes.insert(vertexes.end(), vert.begin(), vert.end());
    }
    return vertexes;
  }
  auto scale = sqrtf(args.viewMatrix.getScaleX() * args.viewMatrix.getScaleX() +
                     args.viewMatrix.getSkewY() * args.viewMatrix.getSkewY());
  // we want the new edge to be .5px away from the old line.
  auto padding = 0.5f / scale;
  auto insetBounds = bounds.makeInset(padding, padding);
  auto insetQuad = Quad::MakeFromRect(insetBounds, matrices[0]);
  auto outsetBounds = bounds.makeOutset(padding, padding);
  auto outsetQuad = Quad::MakeFromRect(outsetBounds, matrices[0]);

  auto normalPadding = Point::Make(padding / bounds.width(), padding / bounds.height());
  auto normalInset = normalBounds.makeInset(normalPadding.x, normalPadding.y);
  auto normalOutset = normalBounds.makeOutset(normalPadding.x, normalPadding.y);
  return {
      insetQuad.point(0).x,  insetQuad.point(0).y,  1.0f, normalInset.left,   normalInset.top,
      insetQuad.point(1).x,  insetQuad.point(1).y,  1.0f, normalInset.left,   normalInset.bottom,
      insetQuad.point(2).x,  insetQuad.point(2).y,  1.0f, normalInset.right,  normalInset.top,
      insetQuad.point(3).x,  insetQuad.point(3).y,  1.0f, normalInset.right,  normalInset.bottom,
      outsetQuad.point(0).x, outsetQuad.point(0).y, 0.0f, normalOutset.left,  normalOutset.top,
      outsetQuad.point(1).x, outsetQuad.point(1).y, 0.0f, normalOutset.left,  normalOutset.bottom,
      outsetQuad.point(2).x, outsetQuad.point(2).y, 0.0f, normalOutset.right, normalOutset.top,
      outsetQuad.point(3).x, outsetQuad.point(3).y, 0.0f, normalOutset.right, normalOutset.bottom,
  };
}

std::unique_ptr<GLFillRectOp> GLFillRectOp::Make(const Rect& rect, const Matrix& matrix) {
  return std::unique_ptr<GLFillRectOp>(new GLFillRectOp({rect}, {matrix}, {Matrix::I()}));
}

std::unique_ptr<GLFillRectOp> GLFillRectOp::Make(const std::vector<Rect>& rects,
                                                 const std::vector<Matrix>& matrices,
                                                 const std::vector<Matrix>& localMatrices) {
  return std::unique_ptr<GLFillRectOp>(new GLFillRectOp(rects, matrices, localMatrices));
}

static constexpr size_t kIndicesPerAAFillRect = 30;

// clang-format off
static constexpr uint16_t gFillAARectIdx[] = {
  0, 1, 2, 1, 3, 2,
  0, 4, 1, 4, 5, 1,
  0, 6, 4, 0, 2, 6,
  2, 3, 6, 3, 7, 6,
  1, 5, 3, 3, 5, 7,
};
// clang-format on

std::shared_ptr<GLBuffer> GLFillRectOp::getIndexBuffer(const DrawArgs& args) {
  if (args.aa == AAType::Coverage) {
    return GLBuffer::Make(args.context, gFillAARectIdx, kIndicesPerAAFillRect);
  }
  return nullptr;
}
}  // namespace tgfx
