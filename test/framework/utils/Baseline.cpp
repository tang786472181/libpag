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

#include "Baseline.h"
#ifdef USE_SSE2
#include <x86intrin.h>
#endif
#include <chrono>
#include <fstream>
#include <unordered_set>
#include "LzmaUtil.h"
#include "core/Data.h"
#include "core/Image.h"
#include <sys/time.h>
#include <iostream>

namespace pag {
#define BASELINE_ROOT "../test/baseline/"
#define OUT_BASELINE_ROOT "../test/out/baseline/"
#define OUT_COMPARE_ROOT "../test/out/compare/"
#define COMPRESS_FILE_EXT ".lzma2"
#define MAX_DIFF_COUNT 10
#define MAX_DIFF_VALUE 5

static ImageInfo MakeInfo(int with, int height) {
  return ImageInfo::Make(with, height, ColorType::RGBA_8888, AlphaType::Premultiplied);
}

static std::shared_ptr<Data> LoadImageData(const std::string& key) {
  auto data = Data::MakeFromFile(BASELINE_ROOT + key + COMPRESS_FILE_EXT);
  if (data == nullptr) {
    return nullptr;
  }
  return LzmaUtil::Decompress(data);
}

static void SaveData(const std::shared_ptr<Data>& data, const std::string& path) {
  std::filesystem::path filePath = path;
  std::filesystem::create_directories(filePath.parent_path());
  std::ofstream out(path);
  out.write(reinterpret_cast<const char*>(data->data()),
            static_cast<std::streamsize>(data->size()));
  out.close();
}

static void SaveImage(const ImageInfo& info, const std::shared_ptr<Data>& imageData,
                      const std::string& key) {
  auto data = LzmaUtil::Compress(imageData);
  if (data == nullptr) {
    return;
  }
  auto path = OUT_BASELINE_ROOT + key + COMPRESS_FILE_EXT;
  SaveData(data, path);
  auto baselineData = LoadImageData(key);
  if (baselineData == nullptr) {
    return;
  }
  auto baselineImage = Bitmap(info, baselineData->data()).encode(EncodedFormat::WEBP, 100);
  SaveData(baselineImage, OUT_COMPARE_ROOT + key + "_baseline.webp");
  auto compareImage = Bitmap(info, imageData->data()).encode(EncodedFormat::WEBP, 100);
  SaveData(compareImage, OUT_COMPARE_ROOT + key + "_new.webp");
}

static void ClearPreviousOutput(const std::string& key) {
  std::filesystem::remove(OUT_BASELINE_ROOT + key + COMPRESS_FILE_EXT);
  std::filesystem::remove(OUT_COMPARE_ROOT + key + "_baseline.webp");
  std::filesystem::remove(OUT_COMPARE_ROOT + key + "_new.webp");
}

class TimeMonitor {
 public:
  static TimeMonitor* GetInstance() {
    static auto* timeMonitor = new TimeMonitor();
    return timeMonitor;
  }

  void begin() {
    gettimeofday(&before, nullptr);
  }

  void end() {
    gettimeofday(&after, nullptr);
    total += currentTime();
  }

  uint64_t totalTimeUs() const {
    return total;
  }

 private:
  struct timeval before {
  }, after{};
  uint64_t total = 0;

  uint64_t currentTime() const {
    timeval result{};
    timeval_subtract(&result, &before, &after);
    return static_cast<uint64_t>(result.tv_sec) * 1000000 + result.tv_usec;
  }

  /**
   * 计算两个时间的间隔，得到时间差
   * @param struct timeval* resule 返回计算出来的时间
   * @param struct timeval* x 需要计算的前一个时间
   * @param struct timeval* y 需要计算的后一个时间
   * return -1 failure ,0 success
   **/
  int timeval_subtract(struct timeval *result, const struct timeval *x, const struct timeval *y) const {
    if (x->tv_sec > y->tv_sec)
      return -1;

    if ((x->tv_sec == y->tv_sec) && (x->tv_usec > y->tv_usec))
      return -1;

    result->tv_sec = (y->tv_sec - x->tv_sec);
    result->tv_usec = (y->tv_usec - x->tv_usec);

    if (result->tv_usec < 0) {
      result->tv_sec--;
      result->tv_usec += 1000000;
    }

    return 0;
  }
};
#ifdef USE_SSE2
#define _mm_cmpge_epu8(a, b) _mm_cmpeq_epi8(_mm_max_epu8(a, b), a)

size_t FastComparePixelData(const uint8_t* baseLine, const uint8_t* pixels, size_t byteSize) {
  auto maxDiffValue = _mm_set1_epi8(MAX_DIFF_VALUE + 1);
  size_t i;
  auto baselineBytes = baseLine;
  auto pixelBytes = pixels;
  auto mask = _mm_set1_epi8(1);
  size_t diffCount = 0;

  for (i = 0; i < (byteSize & ((~(unsigned)0xF))); i += 16) {
    auto pixelA = _mm_loadu_si128((__m128i*)baselineBytes);
    auto pixelB = _mm_loadu_si128((__m128i*)pixelBytes);
    auto maxPixel = _mm_max_epu8(pixelA, pixelB);        // maxPixel = MAX(pixelA, pixelB)
    auto minPixel = _mm_min_epu8(pixelA, pixelB);        // minPixel = MIN(pixelA, pixelB)
    auto diffPixel = _mm_subs_epu8(maxPixel, minPixel);  // diffPixel = maxPixel - minPixel

    auto result = _mm_and_si128(_mm_cmpge_epu8(diffPixel, maxDiffValue), mask);
    // result = maxDiffValue >= maxDiffValue + 1 ? 1 : 0

    diffCount +=
        _mm_extract_epi8(result, 0) + _mm_extract_epi8(result, 1) + _mm_extract_epi8(result, 2) +
        _mm_extract_epi8(result, 3) + _mm_extract_epi8(result, 4) + _mm_extract_epi8(result, 5) +
        _mm_extract_epi8(result, 6) + _mm_extract_epi8(result, 7) + _mm_extract_epi8(result, 8) +
        _mm_extract_epi8(result, 9) + _mm_extract_epi8(result, 10) + _mm_extract_epi8(result, 11) +
        _mm_extract_epi8(result, 12) + _mm_extract_epi8(result, 13) + _mm_extract_epi8(result, 14) +
        _mm_extract_epi8(result, 15);

    baselineBytes += 16;
    pixelBytes += 16;
  }

  for (; i < byteSize; i++) {
    auto pixelA = baseLine[i];
    auto pixelB = pixels[i];
    if (abs(pixelA - pixelB) > MAX_DIFF_VALUE) {
      diffCount++;
    }
  }
  return 0;
}
#endif

size_t NormalPixelCompare(const uint8_t* baseline, const uint8_t* pixels, size_t byteSize) {
  size_t diffCount = 0;
  for (size_t index = 0; index < byteSize; index++) {
    auto pixelA = pixels[index];
    auto pixelB = baseline[index];
    if (abs(pixelA - pixelB) > MAX_DIFF_VALUE) {
      diffCount++;
    }
  }
  return diffCount;
}

static bool ComparePixelData(const std::shared_ptr<Data>& pixelData, const std::string& key,
                             const ImageInfo& info) {
  if (pixelData == nullptr) {
    return false;
  }
  auto baselineData = LoadImageData(key);
  if (baselineData == nullptr || pixelData->size() != baselineData->size()) {
    return false;
  }
  size_t diffCount = 0;
  auto baseline = baselineData->bytes();
  auto pixels = pixelData->bytes();
  auto byteSize = pixelData->size();
  TimeMonitor::GetInstance()->begin();
  #ifdef USE_SSE2
    diffCount = FastComparePixelData(baseline, pixels, byteSize);
  #else
    diffCount = NormalPixelCompare(baseline, pixels, byteSize);
  #endif
  TimeMonitor::GetInstance()->end();
  std::cout<< "Execution time:" << TimeMonitor::GetInstance()->totalTimeUs() << "us" << std::endl;

  // We assume that the two images are the same if the number of different pixels is less than 10.
  if (diffCount > MAX_DIFF_COUNT) {
    SaveImage(info, pixelData, key);
    return false;
  }
  ClearPreviousOutput(key);
  return true;
}

bool Baseline::Compare(const std::shared_ptr<PixelBuffer>& pixelBuffer, const std::string& key) {
  if (pixelBuffer == nullptr) {
    return false;
  }
  Bitmap bitmap(pixelBuffer);
  auto info = MakeInfo(bitmap.width(), bitmap.height());
  auto pixels = new uint8_t[info.byteSize()];
  auto data = Data::MakeAdopted(pixels, info.byteSize(), Data::DeleteProc);
  auto result = bitmap.readPixels(info, pixels);
  if (!result) {
    return false;
  }
  return ComparePixelData(data, key, info);
}

bool Baseline::Compare(const Bitmap& bitmap, const std::string& key) {
  if (bitmap.isEmpty()) {
    return false;
  }
  auto info = MakeInfo(bitmap.width(), bitmap.height());
  auto pixels = new uint8_t[info.byteSize()];
  auto data = Data::MakeAdopted(pixels, info.byteSize(), Data::DeleteProc);
  auto result = bitmap.readPixels(info, pixels);
  if (!result) {
    return false;
  }
  return ComparePixelData(data, key, info);
}

bool Baseline::Compare(const std::shared_ptr<PAGSurface>& surface, const std::string& key) {
  if (surface == nullptr) {
    return false;
  }
  auto info = MakeInfo(surface->width(), surface->height());
  auto pixels = new uint8_t[info.byteSize()];
  auto data = Data::MakeAdopted(pixels, info.byteSize(), Data::DeleteProc);
  auto result =
      surface->readPixels(info.colorType(), AlphaType::Premultiplied, pixels, info.rowBytes());
  if (!result) {
    return false;
  }
  return ComparePixelData(data, key, info);
}
}  // namespace pag
