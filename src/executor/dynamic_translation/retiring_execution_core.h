// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// Public contract of the LSX translation engine.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace Executor::Jit {

enum class LsxGpr : std::uint8_t {
    Rax,
    Rcx,
    Rdx,
    Rbx,
    Rsp,
    Rbp,
    Rsi,
    Rdi,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
};

// The state bank follows the architectural register encoding directly.  Keeping one canonical
// index avoids a private storage ABI and lets decoders, diagnostics and generated code exchange
// register ordinals without a translation table.
constexpr std::size_t GprSlot(const LsxGpr reg) noexcept {
    return static_cast<std::size_t>(reg);
}

static_assert(GprSlot(LsxGpr::Rax) == 0);
static_assert(GprSlot(LsxGpr::Rsp) == 4);
static_assert(GprSlot(LsxGpr::R15) == 15);

enum class AluOp : std::uint8_t {
    Add,
    Subtract,
    Compare,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    BitwiseTest,
};

enum class ShiftOp : std::uint8_t {
    LogicalLeft,
    LogicalRight,
    ArithmeticRight,
};

enum class IncDecOp : std::uint8_t {
    Decrement,
    Increment,
};

enum class BitTestOp : std::uint8_t {
    Inspect,
    Raise,
    Clear,
    Toggle,
};

enum class ScalarArithOp : std::uint8_t {
    Sum,
    Difference,
    Product,
    Quotient,
    Minimum,
    Maximum,
    AlternatingDifferenceSum,
};

inline constexpr std::size_t kLsxInstructionByteLimit = 15;
inline constexpr std::size_t kLsxDecodedOperandLimit = 10;
inline constexpr std::size_t kLsxVisibleOperandLimit = 5;

using LsxInstructionAttributes = std::uint64_t;
using LsxMnemonicCode = std::uint32_t;

struct alignas(32) LsxMachineImage {
    // Shared execution image for translated code, fallback semantics, signal recovery and HLE.
    // Keep this layout byte-exact: generated AArch64 and the signal dispatcher address fields by
    // fixed offset while guest handlers observe the same state object.
    // Slots use the x86 encoding order declared by LsxGpr: RAX,RCX,RDX,RBX,RSP,RBP,RSI,RDI,
    // R8,R9,R10,R11,R12,R13,R14,R15.
    std::array<std::uint64_t, 16> gpr{};
    std::uint64_t rip_or_exit = 0;
    // Generated code derives these displacements from the type itself. Keeping segment bases
    // before the architectural flag word gives this engine its own control-state ABI without
    // growing the image or moving the x87/SIMD tail.
    std::uint64_t gs_base = 0;
    std::uint64_t fs_base = 0;
    std::uint64_t rflags = 0;
    std::uint16_t x87_control_word = 0x037f;
    std::uint8_t x87_top = 0;
    std::uint8_t x87_tag_word = 0;
    std::uint32_t mxcsr = 0x1f80;
    // One non-architectural finite-window phase shares the ABI padding. Its expiration schedules
    // dispatcher admission and executable-byte validation together, avoiding two independent
    // hot-path loads while preserving every architectural x87/SIMD offset below.
    std::uint32_t native_edge_phase = 0;
    std::uint16_t x87_status_word = 0;
    std::uint16_t x87_last_opcode = 0;
    // Physical x87 registers: architectural binary80 in bytes 0..9, zero padding in 10..15.
    std::array<std::array<std::uint8_t, 0x10>, 8> x87_values{};
    std::uint64_t x87_instruction_pointer = 0;
    std::uint64_t x87_data_pointer = 0;
    std::array<std::array<std::uint8_t, 0x20>, 16> ymm{};
    // Non-architectural address of this host thread's g_current_fault_ir TLS slot. Specialized
    // helper-backed paths retain it for compatibility; generic native memory operations publish
    // through native_fault_ir without touching dynamic TLS.
    std::uint64_t fault_ir_slot = 0;
    // Generic native blocks publish the current decoded instruction here immediately before a
    // faultable host access. Keeping the pointer in the execution image lets the signal bridge
    // derive guest RIP from the IR and removes both the TLS store and the separate rip_or_exit
    // store from the hot memory path. Specialized/semantic paths may continue using fault_ir_slot.
    std::uint64_t native_fault_ir = 0;
};

static_assert(std::is_standard_layout_v<LsxMachineImage>);
static_assert(alignof(LsxMachineImage) == 0x20);
static_assert(offsetof(LsxMachineImage, gpr) == 0x00);
static_assert(offsetof(LsxMachineImage, rip_or_exit) == 0x80);
static_assert(offsetof(LsxMachineImage, gs_base) == 0x88);
static_assert(offsetof(LsxMachineImage, fs_base) == 0x90);
static_assert(offsetof(LsxMachineImage, rflags) == 0x98);
static_assert(offsetof(LsxMachineImage, x87_control_word) == 0xa0);
static_assert(offsetof(LsxMachineImage, x87_top) == 0xa2);
static_assert(offsetof(LsxMachineImage, x87_tag_word) == 0xa3);
static_assert(offsetof(LsxMachineImage, mxcsr) == 0xa4);
static_assert(offsetof(LsxMachineImage, native_edge_phase) == 0xa8);
static_assert(offsetof(LsxMachineImage, x87_status_word) == 0xac);
static_assert(offsetof(LsxMachineImage, x87_last_opcode) == 0xae);
static_assert(offsetof(LsxMachineImage, x87_values) == 0xb0);
static_assert(offsetof(LsxMachineImage, x87_instruction_pointer) == 0x130);
static_assert(offsetof(LsxMachineImage, x87_data_pointer) == 0x138);
static_assert(offsetof(LsxMachineImage, ymm) == 0x140);
static_assert(offsetof(LsxMachineImage, fault_ir_slot) == 0x340);
static_assert(offsetof(LsxMachineImage, native_fault_ir) == 0x348);
static_assert(sizeof(LsxMachineImage) == 0x360);

struct LsxEntryPacket {
    using Argument = std::uint64_t;
    using Count = std::uint32_t;
    using Switch = std::uint8_t;

    std::array<Argument, 8> args{};
    Count arg_count{};
    Switch stack_args_enabled{};
    Switch allow_guest_return_sentinel{1};
    std::array<std::uint8_t, 2> reserved_46{};
    Argument segment_base{};
    Argument stack_base{};
    Argument stack_size{};
};

static_assert(std::is_standard_layout_v<LsxEntryPacket> &&
              offsetof(LsxEntryPacket, args) == 0x00 &&
              offsetof(LsxEntryPacket, arg_count) == 0x40 &&
              offsetof(LsxEntryPacket, stack_args_enabled) == 0x44 &&
              offsetof(LsxEntryPacket, segment_base) == 0x48 &&
              offsetof(LsxEntryPacket, stack_base) == 0x50 &&
              offsetof(LsxEntryPacket, stack_size) == 0x58 &&
              sizeof(LsxEntryPacket) == 0x60);

using Arm64BlockEntry = std::uint64_t (*)(LsxMachineImage*);

struct TranslationRecord {
    Arm64BlockEntry entry{};
    std::uint32_t guest_size{};
    std::uint32_t reserved_0c{};
    std::uint64_t guest_hash{};
    std::uint32_t flags{};
    std::uint32_t reserved_1c{};
};

static_assert(std::is_standard_layout_v<TranslationRecord> &&
              offsetof(TranslationRecord, entry) == 0x00 &&
              offsetof(TranslationRecord, guest_size) == 0x08 &&
              offsetof(TranslationRecord, guest_hash) == 0x10 &&
              offsetof(TranslationRecord, flags) == 0x18 &&
              sizeof(TranslationRecord) == 0x20);

enum class LsxOpClass : std::uint8_t {
    Unsupported,
    Branch,
    Return,
    Call,
    Basic,
    Conditional,
    Memory,
    Simd,
    System,
};

// Runtime execution retains only the scalar decode summary consumed by address calculation,
// vector semantics, diagnostics and branch resolution. The complete vendor decode object remains
// private to decode_bridge.cpp instead of being retained by every published operation.
struct DecodeSummary {
    LsxInstructionAttributes attributes = 0;
    std::uint32_t mnemonic = 0;
    std::uint8_t length = 0;
    std::uint8_t encoding = 0;
    std::uint8_t address_width = 0;
    std::uint8_t operand_width = 0;
    std::uint8_t operand_count = 0;
    std::uint8_t operand_count_visible = 0;
};

static_assert(std::is_standard_layout_v<DecodeSummary>);
static_assert(sizeof(DecodeSummary) == 24);

using LsxRegisterCode = std::uint32_t;

struct LsxRegisterOperand {
    LsxRegisterCode value;
};

struct LsxMemoryOperand {
    std::uint32_t type;
    LsxRegisterCode segment;
    LsxRegisterCode base;
    LsxRegisterCode index;
    std::uint8_t scale;
    struct {
        std::int64_t value;
        std::uint8_t offset;
        std::uint8_t size;
    } disp;
};

struct LsxPointerOperand {
    std::uint16_t segment;
    std::uint32_t offset;
};

struct LsxImmediateOperand {
    std::uint8_t is_signed;
    std::uint8_t is_relative;
    union {
        std::uint64_t u;
        std::int64_t s;
    } value;
    std::uint8_t offset;
    std::uint8_t size;
};

// Stable JIT operand vocabulary. Decoder-specific layouts are translated once by
// decode_bridge.cpp; published regions and persistent-cache code retain only these fixed-width
// architectural values.
struct LsxOperandRecord {
    std::uint8_t id;
    std::uint8_t visibility;
    std::uint8_t actions;
    std::uint8_t encoding;
    std::uint16_t size;
    std::uint16_t element_type;
    std::uint16_t element_size;
    std::uint16_t element_count;
    std::uint8_t attributes;
    std::uint8_t type;
    union {
        LsxRegisterOperand reg;
        LsxMemoryOperand mem;
        LsxPointerOperand ptr;
        LsxImmediateOperand imm;
    };
};

static_assert(std::is_standard_layout_v<LsxOperandRecord>);
static_assert(std::is_trivially_copyable_v<LsxOperandRecord>);

struct LsxDecodedOp {
    std::uint64_t guest_rip = 0;
    std::uint8_t length = 0;
    std::uint16_t attributes = 0;
    std::uint32_t mnemonic = 0;
    std::array<std::uint8_t, kLsxInstructionByteLimit> bytes{};
    DecodeSummary decoded{};
    // The retained IR stores only operand_count_visible entries (see DecodeGuestBasicRegion). The
    // architectural maximum for the visible subset is five; hidden decoder operands never enter
    // the published region.
    std::array<LsxOperandRecord, kLsxVisibleOperandLimit> operands{};
    std::uint8_t operand_count = 0;
    LsxOpClass category = LsxOpClass::Unsupported;
    // Decode-time dispatch classification. The decoded IR is immutable after publication, so
    // repeating the large SIMD/scalar-float predicate trees for every execution is pure overhead.
    bool can_execute_vector = false;
    bool can_execute_scalar_float = false;
    bool can_execute_x87 = false;
    bool eligible_for_initial_native = false;
    bool terminates_block = false;
};

static_assert(std::is_standard_layout_v<LsxDecodedOp>);

enum class RegionDecodeStatus : std::uint8_t {
    Ready,
    Rejected,
};

enum class RegionExecutionTier : std::uint8_t {
    Interpreter,
    InitialNative,
};

struct LsxDecodedRegion {
    std::uint64_t start_rip = 0;
    std::uint64_t end_rip = 0;
    std::vector<LsxDecodedOp> instructions{};
    RegionDecodeStatus decode_status = RegionDecodeStatus::Ready;
    RegionExecutionTier execution_tier = RegionExecutionTier::Interpreter;
    std::string diagnostic{};

    [[nodiscard]] bool HasDecodeFailure() const noexcept {
        return decode_status == RegionDecodeStatus::Rejected;
    }
    [[nodiscard]] bool SupportsInitialNativeTier() const noexcept {
        return execution_tier == RegionExecutionTier::InitialNative;
    }
    void RejectDecode() noexcept { decode_status = RegionDecodeStatus::Rejected; }
    void SelectInitialNativeTier(const bool enabled) noexcept {
        execution_tier = enabled ? RegionExecutionTier::InitialNative
                                 : RegionExecutionTier::Interpreter;
    }
};

struct LsxStackReservoir {
    using BytePointer = std::uint8_t*;
    BytePointer allocation{};
    BytePointer low{};
    BytePointer top{};
    BytePointer current{};
};

static_assert(std::is_standard_layout_v<LsxStackReservoir> &&
              offsetof(LsxStackReservoir, allocation) == 0x00 &&
              offsetof(LsxStackReservoir, low) == 0x08 &&
              offsetof(LsxStackReservoir, top) == 0x10 &&
              offsetof(LsxStackReservoir, current) == 0x18 &&
              sizeof(LsxStackReservoir) == 0x20);

struct LsxStackLease {
    std::uint64_t stack_low{};
    std::uint64_t current_sp{};
    std::uint64_t previous_current_sp{};
};

static_assert(std::is_standard_layout_v<LsxStackLease> &&
              offsetof(LsxStackLease, stack_low) == 0x00 &&
              offsetof(LsxStackLease, current_sp) == 0x08 &&
              offsetof(LsxStackLease, previous_current_sp) == 0x10 &&
              sizeof(LsxStackLease) == 0x18);

struct LsxEngineReport {
    bool compiled = true;
    bool executable = false;
    std::uint32_t machine_contract_functions = 243;
    std::uint32_t translation_manifest_functions = 148;
    std::uint32_t implemented_core_helpers = 31;
    std::uint32_t compiler_functions = 8;
    std::uint32_t decode_functions = 3;
    std::uint32_t operand_functions = 17;
    std::uint32_t memory_functions = 6;
    std::uint32_t flags_functions = 6;
    std::uint32_t stack_lease_functions = 4;
    std::uint32_t signal_functions = 6;
    std::uint32_t tls_symbols = 16;
    const char* reason = "lsx_translation_contract_incomplete";
};

std::uint64_t DispatchGuestOnArm64(std::uint64_t guest_rip, const LsxEntryPacket& ctx);

LsxEngineReport QueryTranslationEngine();

// Cheap, lock-free counters for the Android on-screen runtime HUD.  Keep this separate from
// DescribeTranslationEngineJson(): the latter sorts the hot-helper table and builds a large diagnostic JSON,
// while the HUD samples these relaxed atomics only once per second.
struct RuntimeStatsSnapshot {
    std::uint64_t decoded_blocks{};
    std::uint64_t native_blocks{};
    std::uint64_t helper_blocks{};
    std::uint64_t unsupported_blocks{};
    std::uint64_t native_cache_hits{};
    std::uint64_t persistent_ir_enabled{};
    std::uint64_t persistent_ir_hits{};
    std::uint64_t persistent_ir_misses{};
    std::uint64_t persistent_ir_records_loaded{};
    std::uint64_t persistent_ir_records_written{};
    std::uint64_t persistent_native_hits{};
    std::uint64_t persistent_native_segments_restored{};
    std::uint64_t persistent_native_records_captured{};
    std::uint64_t persistent_native_records_written{};
    std::uint64_t persistent_native_restore_fallbacks{};
    std::uint64_t persistent_native_capture_rejected{};
};

[[nodiscard]] RuntimeStatsSnapshot SnapshotTranslationTelemetry() noexcept;
// Configures the per-title, cross-process decoded/native-block cache. The executable fingerprint
// must be content-derived (never a host pointer); every hit is still verified against the live
// guest bytes before relocated native code is allowed to publish.
void ConfigureTranslationStore(const std::string& root_dir, const std::string& title_id,
                                   std::uint64_t executable_fingerprint, bool enabled);
std::string DescribeTranslationEngineJson();
std::string ExerciseTranslationEngineJson();
// Minimal metadata-only test seam for the isolated primary/fallback compiler
// differential. It never executes the returned entry.
extern "C" bool ExecutorProbeTranslationMetadata(
    std::uint64_t guest_rip, TranslationRecord* out) noexcept;
// Diagnostic-only proof seam: resolves a captured host PC to the exact emitted
// code allocation that owns it.  It does not expose or execute guest state.
extern "C" bool ExecutorProbeNativeOwnership(
    std::uint64_t host_pc, std::uint64_t* range_base,
    std::uint64_t* range_size) noexcept;

std::uint64_t ReadGuestGpr64(const LsxMachineImage& state, LsxGpr reg);
void WriteGuestGpr64(LsxMachineImage& state, LsxGpr reg, std::uint64_t value);
std::uint32_t ReadGuestGpr32(const LsxMachineImage& state, LsxGpr reg);
void WriteGuestGpr32(LsxMachineImage& state, LsxGpr reg, std::uint32_t value);
std::uint16_t ReadGuestGpr16(const LsxMachineImage& state, LsxGpr reg);
void WriteGuestGpr16(LsxMachineImage& state, LsxGpr reg, std::uint16_t value);
std::uint8_t ReadGuestGpr8(const LsxMachineImage& state, LsxGpr reg, bool high8 = false);
void WriteGuestGpr8(LsxMachineImage& state, LsxGpr reg, std::uint8_t value, bool high8 = false);
std::uint64_t TruncateToOperandWidth(std::uint64_t value, std::uint32_t size_bits);
std::uint64_t ExtendSignedOperand(std::uint64_t value, std::uint32_t size_bits);
// Byte-exact, domain-separated fingerprint contract for live-code validation. Cached native blocks
// are reusable only
// while this fingerprint and the decoded guest span match.
void RecordIntegerAdditionFlags(LsxMachineImage& state, std::uint64_t lhs, std::uint64_t rhs, std::uint64_t result,
                    std::uint32_t size_bits);
void RecordIntegerSubtractionFlags(LsxMachineImage& state, std::uint64_t lhs, std::uint64_t rhs, std::uint64_t result,
                    std::uint32_t size_bits);
void RecordLogicalResultFlags(LsxMachineImage& state, std::uint64_t result, std::uint32_t size_bits);
void RecordCarryingAdditionFlags(LsxMachineImage& state, std::uint64_t lhs, std::uint64_t rhs, bool carry,
                    std::uint64_t result, std::uint32_t size_bits);
void RecordBorrowingSubtractionFlags(LsxMachineImage& state, std::uint64_t lhs, std::uint64_t rhs, bool borrow,
                    std::uint64_t result, std::uint32_t size_bits);
std::uint64_t LoadGuestScalar(std::uint64_t guest_va, std::uint32_t size_bytes);
void CopyBytesFromGuest(std::uint64_t guest_va, void* dst, std::uint32_t size_bytes);
void StoreGuestScalar(std::uint64_t guest_va, std::uint64_t value, std::uint32_t size_bytes);
void CopyBytesToGuest(std::uint64_t guest_va, const void* src, std::uint32_t size_bytes);
void ExchangeGuestMemoryAtomically(LsxMachineImage& state, void* target, std::size_t state_offset,
                       std::uint32_t size_bits);
LsxStackReservoir& LocalGuestStackReservoir();
LsxStackLease BorrowGuestStackWindow(LsxMachineImage* state);
void ReturnGuestStackWindow(const LsxStackLease& frame);
LsxMachineImage* FindActiveMachineImage();
LsxMachineImage* ExchangeDiagnosticMachineImage(LsxMachineImage* state);
// Returns the architectural state at the continuation after the currently executing HLE call.
// Bridge markers temporarily repurpose guest registers for their sideband protocol, so code that
// suspends execution from inside an HLE must capture this normalized image instead of the raw
// marker image returned by FindActiveMachineImage().
bool SnapshotGuestStateAfterBridge(LsxMachineImage& destination);
// HLE implementations that perform an architectural context switch can replace the live machine
// image and mark it authoritative. The bridge then skips its ordinary function-return epilogue,
// which would otherwise overwrite the resumed RIP/RSP and callee-saved registers.
bool AnnounceGuestStateReplacement();
bool TakeGuestStateReplacementNotice();
LsxStackLease InitializeGuestEntryFrame(LsxMachineImage& state, std::uint64_t guest_rip,
                                          const LsxEntryPacket& ctx);
bool SignalGuestRunCompletion(std::uint64_t result);
bool GuestRunCompletionPending();
std::uint64_t GuestRunCompletionValue();
void ResetGuestRunCompletion();
LsxDecodedRegion DecodeGuestBasicRegion(std::uint64_t guest_rip, std::uint32_t max_instructions);
bool RegionAcceptsNativeTier(const LsxDecodedRegion& block);

} // namespace Executor::Jit
