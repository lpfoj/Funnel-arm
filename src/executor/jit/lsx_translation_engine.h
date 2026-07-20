// SPDX-FileCopyrightText: Copyright 2026 Executor Project
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
    Rax = 0,
    Rcx = 1,
    Rdx = 2,
    Rbx = 3,
    Rsp = 4,
    Rbp = 5,
    Rsi = 6,
    Rdi = 7,
    R8 = 8,
    R9 = 9,
    R10 = 10,
    R11 = 11,
    R12 = 12,
    R13 = 13,
    R14 = 14,
    R15 = 15,
};

enum class AluOp : std::uint8_t {
    Add = 0,
    Sub = 1,
    Cmp = 2,
    And = 3,
    Or = 4,
    Xor = 5,
    Test = 6,
};

enum class ShiftOp : std::uint8_t {
    Shl = 0,
    Shr = 1,
    Sar = 2,
};

enum class IncDecOp : std::uint8_t {
    Dec = 0,
    Inc = 1,
};

enum class BitTestOp : std::uint8_t {
    Test = 0,
    Set = 1,
    Reset = 2,
    Complement = 3,
};

enum class VectorLogicOp : std::uint8_t {
    Xor = 0,
    And = 1,
    Or = 2,
    AndNot = 3,
};

enum class ScalarArithOp : std::uint8_t {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Min = 4,
    Max = 5,
    AddSub = 6,
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
    // Physical order is RAX,RBX,RCX,RDX,RSI,RDI,RBP,RSP,R8,R9,R10,R11,R12,R13,R14,R15.
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
    // The first dword is a non-architectural, per-execution direct-chain budget.  The
    // remaining word-sized fields are architectural x87 state that used to be discarded by
    // FNSTENV/FXSAVE.  Keeping them in the ABI padding preserves every proven state offset.
    std::uint32_t direct_chain_budget = 0;
    std::uint16_t x87_status_word = 0;
    std::uint16_t x87_last_opcode = 0;
    // Physical x87 registers: architectural binary80 in bytes 0..9, zero padding in 10..15.
    std::array<std::array<std::uint8_t, 0x10>, 8> x87_values{};
    std::uint64_t x87_instruction_pointer = 0;
    std::uint64_t x87_data_pointer = 0;
    std::array<std::array<std::uint8_t, 0x20>, 16> ymm{};
    // Non-architectural address of this host thread's g_current_fault_ir TLS slot. Keeping the
    // resolved address in per-execution state lets every directly chained memory block load it
    // with one LDR instead of crossing into C++ merely to resolve TLS at block entry.
    std::uint64_t fault_ir_slot = 0;
};

// AAPCS64 returns this two-qword aggregate in x0/x1.  The leaf-HLE fast bridge uses the second
// word as a fail-closed predicate so generated code can take the original guest CALL/PLT path if
// a GOT slot is ever rebound to a non-leaf target.
struct JitLeafHleCallResult {
    std::uint64_t result = 0;
    std::uint64_t executed = 0;
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
static_assert(offsetof(LsxMachineImage, direct_chain_budget) == 0xa8);
static_assert(offsetof(LsxMachineImage, x87_status_word) == 0xac);
static_assert(offsetof(LsxMachineImage, x87_last_opcode) == 0xae);
static_assert(offsetof(LsxMachineImage, x87_values) == 0xb0);
static_assert(offsetof(LsxMachineImage, x87_instruction_pointer) == 0x130);
static_assert(offsetof(LsxMachineImage, x87_data_pointer) == 0x138);
static_assert(offsetof(LsxMachineImage, ymm) == 0x140);
static_assert(offsetof(LsxMachineImage, fault_ir_slot) == 0x340);
static_assert(sizeof(LsxMachineImage) == 0x360);

struct LsxEntryPacket {
    std::array<std::uint64_t, 8> args{};
    std::uint32_t arg_count = 0;
    std::uint8_t stack_args_enabled = 0;
    std::uint8_t allow_guest_return_sentinel = 1;
    std::array<std::uint8_t, 2> reserved_46{};
    std::uint64_t segment_base = 0;
    std::uint64_t stack_base = 0;
    std::uint64_t stack_size = 0;
};

static_assert(std::is_standard_layout_v<LsxEntryPacket>);
static_assert(offsetof(LsxEntryPacket, args) == 0x00);
static_assert(offsetof(LsxEntryPacket, arg_count) == 0x40);
static_assert(offsetof(LsxEntryPacket, stack_args_enabled) == 0x44);
static_assert(offsetof(LsxEntryPacket, segment_base) == 0x48);
static_assert(offsetof(LsxEntryPacket, stack_base) == 0x50);
static_assert(offsetof(LsxEntryPacket, stack_size) == 0x58);
static_assert(sizeof(LsxEntryPacket) == 0x60);

using Arm64BlockEntry = std::uint64_t (*)(LsxMachineImage*);

struct TranslationRecord {
    Arm64BlockEntry entry = nullptr;
    std::uint32_t guest_size = 0;
    std::uint32_t reserved_0c = 0;
    std::uint64_t guest_hash = 0;
    std::uint32_t flags = 0;
    std::uint32_t reserved_1c = 0;
};

static_assert(std::is_standard_layout_v<TranslationRecord>);
static_assert(offsetof(TranslationRecord, entry) == 0x00);
static_assert(offsetof(TranslationRecord, guest_size) == 0x08);
static_assert(offsetof(TranslationRecord, guest_hash) == 0x10);
static_assert(offsetof(TranslationRecord, flags) == 0x18);
static_assert(sizeof(TranslationRecord) == 0x20);

enum class LsxOpClass : std::uint8_t {
    Unsupported = 0,
    Branch = 1,
    Return = 2,
    Call = 3,
    Basic = 4,
    Conditional = 5,
    Memory = 6,
    Simd = 7,
    System = 8,
};

// Runtime execution retains only the scalar decode summary consumed by address calculation,
// vector semantics, diagnostics and branch resolution. The complete vendor decode object remains
// private to machine_code_lens.cpp instead of being retained by every published operation.
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
// machine_code_lens.cpp; published regions and persistent-cache code retain only these fixed-width
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
    // The retained IR stores only operand_count_visible entries (see BuildLsxCodeRegion). The
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
    bool can_jit_initial = false;
    bool terminates_block = false;
};

static_assert(std::is_standard_layout_v<LsxDecodedOp>);

template <std::uint32_t TrueCode>
struct LsxEncodedLatch {
    std::uint32_t code = 0;

    constexpr LsxEncodedLatch() = default;
    constexpr LsxEncodedLatch(const bool value) : code(value ? TrueCode : 0) {}

    constexpr LsxEncodedLatch& operator=(const bool value) noexcept {
        code = value ? TrueCode : 0;
        return *this;
    }

    [[nodiscard]] constexpr operator bool() const noexcept {
        return code == TrueCode;
    }
};

static_assert(std::is_standard_layout_v<LsxEncodedLatch<1>>);
static_assert(std::is_trivially_copyable_v<LsxEncodedLatch<1>>);
static_assert(sizeof(LsxEncodedLatch<1>) == sizeof(std::uint32_t));

struct LsxDecodedRegion {
    std::uint64_t start_rip = 0;
    std::uint64_t end_rip = 0;
    std::vector<LsxDecodedOp> instructions{};
    // Full-width, fail-closed latches avoid a byte-header ABI while retaining field-style access
    // for decode, cache serialization and diagnostic code.
    LsxEncodedLatch<0xc17a4e31u> can_jit_initial{};
    LsxEncodedLatch<0xd3c0de65u> decode_failed{};
    std::string diagnostic{};
};

struct LsxStackReservoir {
    std::uint8_t* allocation = nullptr;
    std::uint8_t* low = nullptr;
    std::uint8_t* top = nullptr;
    std::uint8_t* current = nullptr;
};

static_assert(std::is_standard_layout_v<LsxStackReservoir>);
static_assert(offsetof(LsxStackReservoir, allocation) == 0x00);
static_assert(offsetof(LsxStackReservoir, low) == 0x08);
static_assert(offsetof(LsxStackReservoir, top) == 0x10);
static_assert(offsetof(LsxStackReservoir, current) == 0x18);
static_assert(sizeof(LsxStackReservoir) == 0x20);

struct LsxStackLease {
    std::uint64_t stack_low = 0;
    std::uint64_t current_sp = 0;
    std::uint64_t previous_current_sp = 0;
};

static_assert(std::is_standard_layout_v<LsxStackLease>);
static_assert(offsetof(LsxStackLease, stack_low) == 0x00);
static_assert(offsetof(LsxStackLease, current_sp) == 0x08);
static_assert(offsetof(LsxStackLease, previous_current_sp) == 0x10);
static_assert(sizeof(LsxStackLease) == 0x18);

struct LsxEngineReport {
    bool compiled = true;
    bool executable = false;
    std::uint32_t machine_contract_functions = 243;
    std::uint32_t translation_manifest_functions = 148;
    std::uint32_t instruction_handlers = 114;
    std::uint32_t implemented_handlers = 114;
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

std::uint64_t RunArm64Translation(std::uint64_t guest_rip, const LsxEntryPacket& ctx);

LsxEngineReport InspectLsxEngine();

// Cheap, lock-free counters for the Android on-screen runtime HUD.  Keep this separate from
// LsxEngineStatusJson(): the latter sorts the hot-helper table and builds a large diagnostic JSON,
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

[[nodiscard]] RuntimeStatsSnapshot GetRuntimeStatsSnapshot() noexcept;
// Configures the per-title, cross-process decoded/native-block cache. The executable fingerprint
// must be content-derived (never a host pointer); every hit is still verified against the live
// guest bytes before relocated native code is allowed to publish.
void ConfigurePersistentBlockCache(const std::string& root_dir, const std::string& title_id,
                                   std::uint64_t executable_fingerprint, bool enabled);
std::string LsxEngineStatusJson();
std::string LsxEngineSelfTestJson();
// Minimal metadata-only test seam for the isolated primary/fallback compiler
// differential. It never executes the returned entry.
extern "C" bool LsxProbeResolveTranslation(
    std::uint64_t guest_rip, TranslationRecord* out) noexcept;
// Diagnostic-only proof seam: resolves a captured host PC to the exact emitted
// code allocation that owns it.  It does not expose or execute guest state.
extern "C" bool LsxProbeLocateNativeRange(
    std::uint64_t host_pc, std::uint64_t* range_base,
    std::uint64_t* range_size) noexcept;

std::uint64_t GetGpr64(const LsxMachineImage& state, LsxGpr reg);
void SetGpr64(LsxMachineImage& state, LsxGpr reg, std::uint64_t value);
std::uint32_t GetGpr32(const LsxMachineImage& state, LsxGpr reg);
void SetGpr32(LsxMachineImage& state, LsxGpr reg, std::uint32_t value);
std::uint16_t GetGpr16(const LsxMachineImage& state, LsxGpr reg);
void SetGpr16(LsxMachineImage& state, LsxGpr reg, std::uint16_t value);
std::uint8_t GetGpr8(const LsxMachineImage& state, LsxGpr reg, bool high8 = false);
void SetGpr8(LsxMachineImage& state, LsxGpr reg, std::uint8_t value, bool high8 = false);
std::uint64_t MaskToSize(std::uint64_t value, std::uint32_t size_bits);
std::uint64_t SignExtend(std::uint64_t value, std::uint32_t size_bits);
// Byte-exact, domain-separated fingerprint contract for live-code validation. Cached native blocks
// are reusable only
// while this fingerprint and the decoded guest span match.
std::uint64_t FingerprintCodeSpan(std::uint64_t guest_va, std::uint32_t size_bytes);
void CommitAddFlags(LsxMachineImage& state, std::uint64_t lhs, std::uint64_t rhs, std::uint64_t result,
                    std::uint32_t size_bits);
void CommitSubtractFlags(LsxMachineImage& state, std::uint64_t lhs, std::uint64_t rhs, std::uint64_t result,
                    std::uint32_t size_bits);
void CommitLogicFlags(LsxMachineImage& state, std::uint64_t result, std::uint32_t size_bits);
void CommitCarryAddFlags(LsxMachineImage& state, std::uint64_t lhs, std::uint64_t rhs, bool carry,
                    std::uint64_t result, std::uint32_t size_bits);
void CommitBorrowSubtractFlags(LsxMachineImage& state, std::uint64_t lhs, std::uint64_t rhs, bool borrow,
                    std::uint64_t result, std::uint32_t size_bits);
std::uint64_t ReadMemory(std::uint64_t guest_va, std::uint32_t size_bytes);
void ReadMemoryBytes(std::uint64_t guest_va, void* dst, std::uint32_t size_bytes);
void WriteMemory(std::uint64_t guest_va, std::uint64_t value, std::uint32_t size_bytes);
void WriteMemoryBytes(std::uint64_t guest_va, const void* src, std::uint32_t size_bytes);
void ExecuteJitXchgMem(LsxMachineImage& state, void* target, std::size_t state_offset,
                       std::uint32_t size_bits);
LsxStackReservoir& AccessLsxStackReservoir();
LsxStackLease AcquireLsxStackLease(LsxMachineImage* state);
void ReleaseLsxStackLease(const LsxStackLease& frame);
LsxMachineImage* CurrentMachineImage();
LsxMachineImage* SwapDiagnosticMachineImage(LsxMachineImage* state);
// HLE implementations that perform an architectural context switch can replace the live machine
// image and mark it authoritative. The bridge then skips its ordinary function-return epilogue,
// which would otherwise overwrite the resumed RIP/RSP and callee-saved registers.
bool MarkCurrentGuestStateOverride();
bool ConsumeCurrentGuestStateOverride();
LsxStackLease PrepareLsxEntry(LsxMachineImage& state, std::uint64_t guest_rip,
                                          const LsxEntryPacket& ctx);
bool RequestCurrentGuestExit(std::uint64_t result);
bool IsGuestExitRequested();
std::uint64_t GetGuestExitResult();
void ClearGuestExitRequest();
LsxDecodedRegion BuildLsxCodeRegion(std::uint64_t guest_rip, std::uint32_t max_instructions);
bool QualifiesForInitialLowering(const LsxDecodedRegion& block);

// Scalar precision-conversion contract: x0=dst xmm, x1=old dst xmm,
// x2=src scalar, x3=direction.
void Alu(LsxMachineImage& state, std::size_t state_offset, std::uint64_t operand, AluOp op,
         std::uint32_t size_bits, std::uint32_t shift_bits);
std::uint64_t AluMem(LsxMachineImage& state, std::uint64_t mem_value, std::uint64_t operand,
                     AluOp op, std::uint32_t size_bits);
std::uint64_t AluMemLocked(LsxMachineImage& state, void* target, std::uint64_t operand,
                           AluOp op, std::uint32_t size_bits);
void MovbeLoad(LsxMachineImage& state, const void* src, std::size_t state_offset, std::uint32_t size_bits);
void MovbeStore(const LsxMachineImage& state, void* dst, std::size_t state_offset, std::uint32_t size_bits);
void MovN(void* dst, const void* src, std::size_t size);
bool TestCondition(const LsxMachineImage& state, LsxMnemonicCode mnemonic);
std::uint64_t EvalCond(const LsxMachineImage& state, LsxMnemonicCode mnemonic,
                       std::uint64_t true_value, std::uint64_t false_value);
void Setcc(LsxMachineImage& state, std::size_t state_offset, LsxMnemonicCode mnemonic);
void SetccMem(const LsxMachineImage& state, void* dst, LsxMnemonicCode mnemonic);
void Cmov(LsxMachineImage& state, std::size_t state_offset, std::uint64_t value,
          LsxMnemonicCode mnemonic, std::uint32_t size_bits);
void Shift(LsxMachineImage& state, std::size_t state_offset, std::uint64_t count,
           ShiftOp op, std::uint32_t size_bits);
void ShiftMem(LsxMachineImage& state, void* target, std::uint64_t count, ShiftOp op,
              std::uint32_t size_bits);
void Shld(LsxMachineImage& state, std::size_t state_offset, std::uint64_t source,
          std::uint64_t count, std::uint32_t size_bits);
void ShldMem(LsxMachineImage& state, void* target, std::uint64_t source,
             std::uint64_t count, std::uint32_t size_bits);
void Rotate(LsxMachineImage& state, std::size_t state_offset, std::uint64_t count,
            bool rotate_right, std::uint32_t size_bits);
void RotateMem(LsxMachineImage& state, void* target, std::uint64_t count,
               bool rotate_right, std::uint32_t size_bits);
void IncDec(LsxMachineImage& state, std::size_t state_offset, IncDecOp op, std::uint32_t size_bits);
void IncDecMem(LsxMachineImage& state, void* target, IncDecOp op, std::uint32_t size_bits);
void IncDecMemLocked(LsxMachineImage& state, void* target, IncDecOp op, std::uint32_t size_bits);
void Neg(LsxMachineImage& state, std::size_t state_offset, std::uint32_t size_bits);
void NegMem(LsxMachineImage& state, void* target, std::uint32_t size_bits);
void NegMemLocked(LsxMachineImage& state, void* target, std::uint32_t size_bits);
void MulOne(LsxMachineImage& state, std::uint64_t operand, std::uint32_t size_bits, bool is_signed);
void DivOne(LsxMachineImage& state, std::uint64_t divisor, std::uint32_t size_bits, bool is_signed);
void AdcSbb(LsxMachineImage& state, std::size_t state_offset, std::uint64_t operand,
            bool subtract, std::uint32_t size_bits);
void AdcSbbMem(LsxMachineImage& state, void* target, std::uint64_t operand, bool subtract,
               std::uint32_t size_bits);
void AdcSbbMemLocked(LsxMachineImage& state, void* target, std::uint64_t operand, bool subtract,
                     std::uint32_t size_bits);
void Andn(LsxMachineImage& state, std::size_t state_offset, std::uint64_t lhs_not,
          std::uint64_t rhs, std::uint32_t size_bits);
void Blsi(LsxMachineImage& state, std::size_t state_offset, std::uint64_t value,
          std::uint32_t size_bits);
void SetDirectionFlag(LsxMachineImage& state, bool set);
void Movs(LsxMachineImage& state, std::uint64_t element_size, bool repeat);
void Stos(LsxMachineImage& state, std::uint64_t element_size, bool repeat);
void Cmps(LsxMachineImage& state, LsxMnemonicCode mnemonic, bool repeat_equal,
          bool repeat_not_equal);
void Cmpxchg(LsxMachineImage& state, void* target, bool target_is_register_lane, std::uint32_t target_shift,
             std::size_t src_state_offset, std::uint32_t src_shift, std::uint32_t size_bits);
void CmpxchgLocked(LsxMachineImage& state, void* target, bool target_is_register_lane,
                   std::uint32_t target_shift, std::size_t src_state_offset,
                   std::uint32_t src_shift, std::uint32_t size_bits);
void Bextr(LsxMachineImage& state, std::size_t state_offset, std::uint64_t value,
           std::uint32_t control, std::uint32_t size_bits);
void Blsr(LsxMachineImage& state, std::size_t state_offset, std::uint64_t value,
          std::uint32_t size_bits);
void BtReg(LsxMachineImage& state, std::size_t state_offset, std::uint64_t bit_index,
           std::uint32_t size_bits);
void BitTest(LsxMachineImage& state, std::uintptr_t state_offset_or_mem_ptr, bool target_is_memory,
             std::uint64_t bit_index, bool unsigned_memory_index, std::uint32_t size_bits,
             BitTestOp op);
void BitTestLocked(LsxMachineImage& state, std::uintptr_t state_offset_or_mem_ptr, bool target_is_memory,
                   std::uint64_t bit_index, bool unsigned_memory_index, std::uint32_t size_bits,
                   BitTestOp op);
void BitScan(LsxMachineImage& state, std::size_t state_offset, std::uint64_t value,
             bool reverse, std::uint32_t size_bits);
void Xadd(LsxMachineImage& state, std::size_t dst_state_offset, std::size_t src_state_offset,
          std::uint32_t size_bits);
void XaddMem(LsxMachineImage& state, void* target, std::size_t src_state_offset, std::uint32_t size_bits);
void XaddMemLocked(LsxMachineImage& state, void* target, std::size_t src_state_offset,
                   std::uint32_t size_bits);
void Popcnt(LsxMachineImage& state, std::size_t state_offset, std::uint64_t value,
            std::uint32_t src_size_bits, std::uint32_t dst_size_bits);
void Lzcnt(LsxMachineImage& state, std::size_t state_offset, std::uint64_t value,
           std::uint32_t src_size_bits, std::uint32_t dst_size_bits);
void Tzcnt(LsxMachineImage& state, std::size_t state_offset, std::uint64_t value,
           std::uint32_t src_size_bits, std::uint32_t dst_size_bits);
void Bswap(LsxMachineImage& state, std::size_t state_offset, std::uint32_t size_bits);
void NotMem(void* target, std::uint32_t size_bits);
void Imul(LsxMachineImage& state, std::size_t state_offset, std::uint64_t lhs, std::uint64_t rhs,
          std::uint32_t size_bits);
void VMov(void* dst, const void* src, std::size_t size, bool zero_upper_ymm);
void Movdq(void* vector, void* other, std::size_t size, std::uint32_t mode);
void Movlhps64(void* dst, const void* old_dst_or_src, const void* src, bool high_qword,
               std::uint32_t mode);
void Movlhlps(void* dst, const void* lhs, const void* rhs, bool high_selector,
              bool full_vector);
void SseLogic(std::array<std::uint8_t, 16>& dst, const std::array<std::uint8_t, 16>& src,
              VectorLogicOp op);
void VLogic(void* dst, const void* lhs, const void* rhs, VectorLogicOp op, std::size_t size);
void VPackedTest(LsxMachineImage& state, const void* lhs, const void* rhs, std::size_t size);
void VMoveSingleDuplicate(void* dst, const void* src, bool high_source_elements, std::size_t size);
void VMoveDoubleDuplicate(void* dst, const void* src, std::size_t size, bool zero_upper_ymm);
void VBroadcastScalar(void* dst, const void* src, std::size_t element_size, std::size_t total_size);
void VBroadcast128(void* dst, const void* src);
std::uint64_t VMoveMask(const void* src, std::size_t lane_bytes, std::size_t total_size);
void VExtractDword(void* dst, const void* src, std::uint32_t index, bool store_qword);
void VExtractElement(void* dst, const void* src, std::uint32_t index, bool store_qword,
                     std::size_t element_size);
void VInsertPs(void* dst, const void* old_dst, const void* src, std::uint32_t imm,
               bool src_is_memory);
void VPinsr(void* dst, const void* old_dst, std::uint64_t value, std::uint32_t index,
            std::size_t element_size);
void VInsertF128(void* dst, const void* old_dst, const void* src, std::uint32_t half_index);
void VExtractF128(void* dst, const void* src, std::uint32_t half_index, bool zero_upper_ymm);
void VPerm2F128(void* dst, const void* lhs, const void* rhs, std::uint32_t imm);
void VBlendImm(void* dst, const void* lhs, const void* rhs, std::uint32_t imm,
               std::size_t lane_bytes, std::size_t total_size,
               bool repeat_mask_per_128 = false);
void VBlendVar(void* dst, const void* lhs, const void* rhs, const void* mask,
               std::size_t lane_bytes, std::size_t total_size);
void VPBlendVB(void* dst, const void* lhs, const void* rhs, const void* mask,
               std::size_t total_size);
void VPBlendW(void* dst, const void* lhs, const void* rhs, std::uint32_t imm,
              std::size_t total_size);
void Pshufb(void* dst, const void* src, const void* mask, std::size_t total_size);
void Punpck(void* dst, const void* lhs, const void* rhs, std::size_t element_bytes,
            bool high_half, std::size_t total_size);
void VPunpck(void* dst, const void* lhs, const void* rhs, std::size_t element_bytes,
             bool high_half, std::size_t total_size);
void VMovScalarMerge(void* dst, const void* old_dst, const void* scalar_src,
                     std::size_t scalar_size);
void Pshufd(void* dst, const void* src, std::uint8_t imm);
void VPshufw(void* dst, const void* src, std::uint8_t imm, std::size_t total_size,
             bool high_half);
void Shufps(void* dst, const void* lhs, const void* rhs, std::uint8_t imm);
void VShuf(void* dst, const void* lhs, const void* rhs, std::uint8_t imm,
           std::size_t total_size, bool double_lane);
void VPackedZeroExtend(void* dst, const void* src, std::size_t src_element_size,
                       std::size_t dst_element_size, std::size_t total_size);
void VPackedSignExtend(void* dst, const void* src, std::size_t src_element_size,
                       std::size_t dst_element_size, std::size_t total_size);
void VCvtsi2Scalar(void* dst, const void* old_dst, std::uint64_t value,
                   std::uint32_t source_size_bits, bool integer_to_double);
void VCvtScalarToInt(LsxMachineImage& state, std::size_t state_offset, const void* scalar_src,
                     std::uint32_t dst_size_bits, bool source_is_double,
                     bool use_current_rounding);
void Comi(LsxMachineImage& state, const void* lhs, const void* rhs, bool compare_double);
void VScalarArith(void* dst, const void* lhs_old_dst, const void* rhs, ScalarArithOp op,
                  bool operate_double);
void VScalarSqrt(void* dst, const void* old_dst, const void* src, bool operate_double);
void VScalarRound(void* dst, const void* old_dst, const void* src, std::uint32_t imm,
                  bool operate_double, std::uint32_t mxcsr);
void VPackedAddSub(void* dst, const void* lhs, const void* rhs, std::size_t element_size,
                   bool subtract, std::size_t total_size);
void VPackuswb(void* dst, const void* lhs, const void* rhs, std::size_t total_size);
void VPackusdw(void* dst, const void* lhs, const void* rhs, std::size_t total_size);
void VPackedQShift(void* dst, const void* src, std::uint64_t count, bool left,
                   std::size_t total_size);
void VPackedWordShift(void* dst, const void* src, std::uint64_t count,
                      std::size_t element_bytes, bool left, bool arithmetic_right,
                      std::size_t total_size);
void VPackedByteShift(void* dst, const void* src, std::size_t shift, bool left,
                      std::size_t total_size);
void PsImm(void* dst, const void* src, std::uint64_t count, std::size_t element_bytes,
           bool left, bool arithmetic_right, std::size_t total_size);
void VPsImm(void* dst, const void* src, std::uint64_t count, std::size_t element_bytes,
            bool left, bool arithmetic_right, std::size_t total_size);
void VPackedMultiply(void* dst, const void* lhs, const void* rhs, std::uint32_t mode,
                     std::size_t total_size);
void VPackedMinMax(void* dst, const void* lhs, const void* rhs, std::size_t element_size,
                   bool compare_signed, bool select_max, std::size_t total_size);
void VPackedCompare(void* dst, const void* lhs, const void* rhs, std::size_t element_size,
                    bool signed_greater_than, std::size_t total_size);
void VPackedFloatArith(void* dst, const void* lhs, const void* rhs, ScalarArithOp op,
                       bool operate_double, std::size_t total_size);
void VPackedSqrt(void* dst, const void* src, bool operate_double, std::size_t total_size,
                 bool zero_upper_ymm);
void VPackedFloatCompare(void* dst, const void* lhs, const void* rhs, std::uint32_t predicate,
                         bool operate_double, std::size_t total_size);
void VPackedReciprocalEstimate(void* dst, const void* src, bool reciprocal_sqrt,
                               std::size_t total_size, bool zero_upper_ymm);
void VPackedRound(void* dst, const void* src, std::uint32_t imm, bool operate_double,
                  std::size_t total_size, bool zero_upper_ymm, std::uint32_t mxcsr);
void VPackedHorizontalAddFloat(void* dst, const void* lhs, const void* rhs, bool double_lane,
                               bool subtract, std::size_t total_size, bool zero_upper_ymm);
void VPhaddd(void* dst, const void* lhs, const void* rhs, std::size_t total_size);
void VPhminposuw(void* dst, const void* src, bool zero_upper_ymm);
void VDpps(void* dst, const void* lhs, const void* rhs, std::uint8_t imm,
           std::size_t total_size, bool zero_upper_ymm);
void VPackedDqPsConvert(void* dst, const void* src, bool int_to_float, bool truncate,
                        std::size_t total_size, std::uint32_t mxcsr);
void VScalarCompare(void* dst, const void* old_dst, const void* lhs, const void* rhs,
                    std::uint32_t predicate, bool operate_double);
void VPermil(void* dst, const void* src, const void* control, std::uint8_t imm,
             std::size_t total_size, bool double_lane, bool immediate_control);
void VMaskMov(void* vector, const void* mask, void* other, std::size_t total_size,
              std::size_t lane_bytes, bool store_direction);
void VcvtScalarPrecision(std::array<std::uint8_t, 32>& dst,
                         const std::array<std::uint8_t, 16>& old_dst,
                         const void* src_scalar, bool single_to_double);
void FxsaveRestore(LsxMachineImage& state, std::uint64_t address, bool save,
                   bool is_64_bit_image = false);
void Cmpxchg16bLocked(LsxMachineImage& state, std::uint64_t address);

} // namespace Executor::Jit
