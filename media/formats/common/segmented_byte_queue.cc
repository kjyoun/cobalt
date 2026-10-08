// Copyright 2026 The Cobalt Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "media/formats/common/segmented_byte_queue.h"

#include <algorithm>
#include <utility>

#include "base/check_op.h"
#include "base/memory/raw_span.h"
#include "base/notreached.h"
#include "base/numerics/safe_conversions.h"
#include "media/base/byte_queue.h"

namespace media {

namespace {

// A SegmentedByteQueue that borrows appended byte ranges instead of copying
// them, preserving the boundary of each append.
//
// ByteQueue copies every appended buffer into a single growable block so that
// parsers always see contiguous memory. That costs a full copy of every byte
// of media data, and the block only ever grows. BorrowedSegmentedByteQueue
// never copies on append: each append becomes its own segment pointing
// directly at the caller's memory, kept alive by its `release` closure until
// the segment has been entirely popped, or until Reset() or destruction.
// Popping releases whole segments, so the underlying buffers are handed back
// as soon as they have been fully consumed.
//
// The trade-off is that reads are only contiguous within a segment.
// PeekLinearizedData() gathers a range straddling a boundary into scratch
// storage, which is reused across calls: it grows to fit the largest range
// gathered so far, and is only freed by Reset().
//
// Push() always succeeds, and does not invalidate spans returned earlier.
class BorrowedSegmentedByteQueue final : public SegmentedByteQueue {
 public:
  BorrowedSegmentedByteQueue() = default;

  BorrowedSegmentedByteQueue(const BorrowedSegmentedByteQueue&) = delete;
  BorrowedSegmentedByteQueue& operator=(const BorrowedSegmentedByteQueue&) =
      delete;

  ~BorrowedSegmentedByteQueue() override = default;

  // SegmentedByteQueue implementation.
  void Reset() override;
  [[nodiscard]] bool Push(base::span<const uint8_t> data,
                          base::ScopedClosureRunner release) override;
  void Pop(size_t count) override;
  size_t size() const override;
  [[nodiscard]] base::span<const uint8_t> PeekContiguousData(
      size_t offset) const override;
  [[nodiscard]] std::optional<Segments> PeekSegmentedData(
      size_t offset,
      size_t size) const override;
  [[nodiscard]] std::optional<base::span<const uint8_t>> PeekLinearizedData(
      size_t offset,
      size_t size) override;

 private:
  struct BorrowedSegment {
    // Declaration order is load-bearing: members are destroyed in reverse
    // order, so `data` must come last to be destroyed before the `release`
    // that frees the memory it points at. Otherwise it briefly references
    // released memory, which BackupRefPtr can flag.
    base::ScopedClosureRunner release;
    base::raw_span<const uint8_t> data;
  };

  // Maps `offset`, relative to the front of the queue, to the index of the
  // segment holding it and the offset within that segment. `offset` must be
  // less than `total_bytes_`.
  std::pair<size_t, size_t> Locate(size_t offset) const;

  std::vector<BorrowedSegment> segments_;

  // Total number of unread bytes across all segments.
  size_t total_bytes_ = 0u;

  // Scratch storage that PeekLinearizedData() gathers ranges straddling a
  // segment boundary into. Created on first use. ByteQueue::Reset() keeps its
  // storage, so Reset() destroys the ByteQueue to free it.
  std::optional<ByteQueue> scratch_;
};

// A SegmentedByteQueue that copies every append into a single ByteQueue. The
// whole queue is one segment, so PeekContiguousData() returns everything from
// `offset` to the end, PeekSegmentedData() returns at most one run, and
// PeekLinearizedData() never copies.
//
// Push() copies `data` and destroys `release` before returning. It may move
// the queued bytes, so it invalidates every span returned earlier.
class OwnedSegmentedByteQueue final : public SegmentedByteQueue {
 public:
  OwnedSegmentedByteQueue() = default;

  OwnedSegmentedByteQueue(const OwnedSegmentedByteQueue&) = delete;
  OwnedSegmentedByteQueue& operator=(const OwnedSegmentedByteQueue&) = delete;

  ~OwnedSegmentedByteQueue() override = default;

  // SegmentedByteQueue implementation.
  void Reset() override;
  [[nodiscard]] bool Push(base::span<const uint8_t> data,
                          base::ScopedClosureRunner release) override;
  void Pop(size_t count) override;
  size_t size() const override;
  [[nodiscard]] base::span<const uint8_t> PeekContiguousData(
      size_t offset) const override;
  [[nodiscard]] std::optional<Segments> PeekSegmentedData(
      size_t offset,
      size_t size) const override;
  [[nodiscard]] std::optional<base::span<const uint8_t>> PeekLinearizedData(
      size_t offset,
      size_t size) override;

 private:
  ByteQueue queue_;
};

void BorrowedSegmentedByteQueue::Reset() {
  segments_.clear();
  total_bytes_ = 0u;
  scratch_.reset();
}

bool BorrowedSegmentedByteQueue::Push(base::span<const uint8_t> data,
                                      base::ScopedClosureRunner release) {
  if (data.empty()) {
    return true;
  }

  segments_.push_back(BorrowedSegment{std::move(release), data});
  total_bytes_ += data.size();
  return true;
}

void BorrowedSegmentedByteQueue::Pop(size_t count) {
  CHECK_LE(count, total_bytes_);
  total_bytes_ -= count;

  size_t fully_consumed = 0u;
  while (count > 0u) {
    DCHECK_LT(fully_consumed, segments_.size());
    BorrowedSegment& front = segments_[fully_consumed];
    if (count < front.data.size()) {
      front.data = front.data.subspan(count);
      break;
    }
    count -= front.data.size();

    // Clear the span here rather than leaving it to ~BorrowedSegment: the
    // erase() below move-assigns the survivors down over these slots, and
    // member-wise move-assignment runs in declaration order, so `release` would
    // free the memory while `data` still pointed at it.
    front.data = {};
    ++fully_consumed;
  }

  segments_.erase(segments_.begin(), segments_.begin() + fully_consumed);
}

size_t BorrowedSegmentedByteQueue::size() const {
  return total_bytes_;
}

base::span<const uint8_t> BorrowedSegmentedByteQueue::PeekContiguousData(
    size_t offset) const {
  if (offset >= total_bytes_) {
    return {};
  }

  auto [index, offset_in_segment] = Locate(offset);
  return segments_[index].data.subspan(offset_in_segment);
}

std::optional<SegmentedByteQueue::Segments>
BorrowedSegmentedByteQueue::PeekSegmentedData(size_t offset,
                                              size_t size) const {
  if (offset > total_bytes_ || total_bytes_ - offset < size) {
    return std::nullopt;
  }

  Segments segments;
  if (size == 0u) {
    return segments;
  }

  auto [index, offset_in_segment] = Locate(offset);
  size_t remaining = size;
  while (remaining > 0u) {
    const base::span<const uint8_t> segment_data = segments_[index].data;
    const size_t taken =
        std::min(remaining, segment_data.size() - offset_in_segment);
    segments.push_back(segment_data.subspan(offset_in_segment, taken));
    remaining -= taken;

    offset_in_segment = 0u;
    ++index;
  }
  return segments;
}

std::optional<base::span<const uint8_t>>
BorrowedSegmentedByteQueue::PeekLinearizedData(size_t offset, size_t size) {
  if (offset > total_bytes_ || total_bytes_ - offset < size) {
    return std::nullopt;
  }
  if (size == 0u) {
    // Spelled out, since `return {};` would be std::nullopt.
    return base::span<const uint8_t>();
  }

  auto [index, offset_in_segment] = Locate(offset);
  base::span<const uint8_t> source =
      segments_[index].data.subspan(offset_in_segment);
  if (source.size() >= size) {
    // The range lies within one segment, so read it in place.
    return source.first(size);
  }

  // Gather the range: the tail of the segment holding `offset`, then as much of
  // each following segment as is needed. There are enough bytes buffered, as
  // checked above, so this never runs out of segments. ByteQueue::Reset() keeps
  // the storage grown by earlier calls, so it is reused here.
  if (!scratch_) {
    scratch_.emplace();
  }
  scratch_->Reset();
  size_t remaining = size;
  while (true) {
    const size_t taken = std::min(remaining, source.size());
    if (!scratch_->Push(source.first(taken))) {
      return std::nullopt;
    }
    remaining -= taken;
    if (remaining == 0u) {
      break;
    }
    source = segments_[++index].data;
  }
  return scratch_->Data();
}

std::pair<size_t, size_t> BorrowedSegmentedByteQueue::Locate(
    size_t offset) const {
  DCHECK_LT(offset, total_bytes_);

  for (size_t index = 0u; index < segments_.size(); ++index) {
    const size_t segment_size = segments_[index].data.size();
    if (offset < segment_size) {
      return {index, offset};
    }
    offset -= segment_size;
  }

  // Locate() is a private function and the caller ensures `offset` is less
  // than `total_bytes_`, so the offset must have been found above.
  NOTREACHED();
}

void OwnedSegmentedByteQueue::Reset() {
  queue_.Reset();
}

bool OwnedSegmentedByteQueue::Push(base::span<const uint8_t> data,
                                   base::ScopedClosureRunner release) {
  // Nothing refers to `data` once this returns, so `release` is simply left to
  // be destroyed on return.
  if (data.empty()) {
    // ByteQueue::Push() does not accept empty data.
    return true;
  }
  return queue_.Push(data);
}

void OwnedSegmentedByteQueue::Pop(size_t count) {
  CHECK_LE(count, size());
  queue_.Pop(base::checked_cast<int>(count));
}

size_t OwnedSegmentedByteQueue::size() const {
  return queue_.Data().size();
}

base::span<const uint8_t> OwnedSegmentedByteQueue::PeekContiguousData(
    size_t offset) const {
  const base::span<const uint8_t> data = queue_.Data();
  if (offset >= data.size()) {
    return {};
  }
  return data.subspan(offset);
}

std::optional<SegmentedByteQueue::Segments>
OwnedSegmentedByteQueue::PeekSegmentedData(size_t offset, size_t size) const {
  const base::span<const uint8_t> data = queue_.Data();
  if (offset > data.size() || data.size() - offset < size) {
    return std::nullopt;
  }

  Segments segments;
  if (size > 0u) {
    segments.push_back(data.subspan(offset, size));
  }
  return segments;
}

std::optional<base::span<const uint8_t>>
OwnedSegmentedByteQueue::PeekLinearizedData(size_t offset, size_t size) {
  const base::span<const uint8_t> data = queue_.Data();
  if (offset > data.size() || data.size() - offset < size) {
    return std::nullopt;
  }
  return data.subspan(offset, size);
}

}  // namespace

// static
std::unique_ptr<SegmentedByteQueue> SegmentedByteQueue::Create(
    bool borrow_mode) {
  if (borrow_mode) {
    return std::make_unique<BorrowedSegmentedByteQueue>();
  }
  return std::make_unique<OwnedSegmentedByteQueue>();
}

SegmentedByteQueue::~SegmentedByteQueue() = default;

}  // namespace media
