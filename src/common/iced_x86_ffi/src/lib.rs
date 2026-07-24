// SPDX-FileCopyrightText: Copyright 2026 PS4Run Project
// SPDX-License-Identifier: GPL-2.0-or-later

use core::ffi::{c_char, c_void};
use core::{ptr, slice};
use iced_x86::{
    Code, Decoder, DecoderOptions, Formatter, Instruction, InstructionInfoFactory, IntelFormatter,
    Mnemonic, OpAccess, OpKind, Register,
};

const STATUS_OK: i32 = 0;
const STATUS_INVALID_ARGUMENT: i32 = -1;
const STATUS_DECODE_FAILED: i32 = -2;

const ATTR_HAS_LOCK: u64 = 1 << 0;
const ATTR_HAS_REP: u64 = 1 << 1;
const ATTR_HAS_REPE: u64 = 1 << 2;
const ATTR_HAS_REPNE: u64 = 1 << 3;

const OPERAND_UNUSED: u8 = 0;
const OPERAND_REGISTER: u8 = 1;
const OPERAND_MEMORY: u8 = 2;
const OPERAND_POINTER: u8 = 3;
const OPERAND_IMMEDIATE: u8 = 4;

const ACTION_READ: u8 = 1;
const ACTION_WRITE: u8 = 2;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct X86RegisterOperand {
    value: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct X86Displacement {
    value: i64,
    offset: u8,
    size: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct X86MemoryOperand {
    kind: u32,
    segment: u32,
    base: u32,
    index: u32,
    scale: u8,
    disp: X86Displacement,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct X86PointerOperand {
    segment: u16,
    offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union X86ImmediateValue {
    u: u64,
    s: i64,
}

impl Default for X86ImmediateValue {
    fn default() -> Self {
        Self { u: 0 }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct X86ImmediateOperand {
    is_signed: u8,
    is_relative: u8,
    value: X86ImmediateValue,
    offset: u8,
    size: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union X86OperandValue {
    reg: X86RegisterOperand,
    mem: X86MemoryOperand,
    ptr: X86PointerOperand,
    imm: X86ImmediateOperand,
}

impl Default for X86OperandValue {
    fn default() -> Self {
        Self {
            mem: X86MemoryOperand::default(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct X86DecodedOperand {
    id: u8,
    visibility: u8,
    actions: u8,
    encoding: u8,
    size: u16,
    element_type: u16,
    element_size: u16,
    element_count: u16,
    attributes: u8,
    operand_type: u8,
    value: X86OperandValue,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct X86DecodedInstruction {
    attributes: u64,
    mnemonic: u32,
    length: u8,
    encoding: u8,
    address_width: u8,
    operand_width: u8,
    operand_count: u8,
    operand_count_visible: u8,
    bytes: [u8; 15],
}

fn access_mask(access: OpAccess) -> u8 {
    match access {
        OpAccess::Read | OpAccess::CondRead => ACTION_READ,
        OpAccess::Write | OpAccess::CondWrite => ACTION_WRITE,
        OpAccess::ReadWrite | OpAccess::ReadCondWrite => ACTION_READ | ACTION_WRITE,
        _ => 0,
    }
}

fn immediate_value(instruction: &Instruction, kind: OpKind) -> (u64, bool, u8) {
    match kind {
        OpKind::Immediate8 => (instruction.immediate8() as u64, false, 8),
        OpKind::Immediate8_2nd => (instruction.immediate8_2nd() as u64, false, 8),
        OpKind::Immediate16 => (instruction.immediate16() as u64, false, 16),
        OpKind::Immediate32 => (instruction.immediate32() as u64, false, 32),
        OpKind::Immediate64 => (instruction.immediate64(), false, 64),
        OpKind::Immediate8to16 => (instruction.immediate8to16() as i64 as u64, true, 8),
        OpKind::Immediate8to32 => (instruction.immediate8to32() as i64 as u64, true, 8),
        OpKind::Immediate8to64 => (instruction.immediate8to64() as u64, true, 8),
        OpKind::Immediate32to64 => (instruction.immediate32to64() as u64, true, 32),
        _ => (0, false, 0),
    }
}

fn register_width(register: Register) -> u16 {
    register.info().size().saturating_mul(8) as u16
}

fn operand_width(instruction: &Instruction) -> u8 {
    let declared_width = instruction.op_code().operand_size();
    if matches!(declared_width, 8 | 16 | 32 | 64) {
        return declared_width as u8;
    }
    let mut width = 0u16;
    for index in 0..instruction.op_count() {
        let candidate = match instruction.op_kind(index) {
            OpKind::Register => register_width(instruction.op_register(index)),
            OpKind::Memory => instruction.memory_size().info().size().saturating_mul(8) as u16,
            kind => immediate_value(instruction, kind).2 as u16,
        };
        if candidate <= 64 {
            width = width.max(candidate);
        }
    }
    width.max(8).min(64) as u8
}

fn relative_target_delta(instruction: &Instruction, kind: OpKind) -> i64 {
    let next_ip = instruction.next_ip();
    let target = match kind {
        OpKind::NearBranch16 => instruction.near_branch16() as u64,
        OpKind::NearBranch32 => instruction.near_branch32() as u64,
        OpKind::NearBranch64 => instruction.near_branch64(),
        _ => next_ip,
    };
    target.wrapping_sub(next_ip) as i64
}

fn normalized_mnemonic(mnemonic: Mnemonic) -> Mnemonic {
    match mnemonic {
        Mnemonic::Pcmpestri64 => Mnemonic::Pcmpestri,
        Mnemonic::Pcmpestrm64 => Mnemonic::Pcmpestrm,
        Mnemonic::Vpcmpestri64 => Mnemonic::Vpcmpestri,
        Mnemonic::Vpcmpestrm64 => Mnemonic::Vpcmpestrm,
        Mnemonic::Sysexitq => Mnemonic::Sysexit,
        Mnemonic::Sysretq => Mnemonic::Sysret,
        Mnemonic::Retf => Mnemonic::Ret,
        Mnemonic::Sal => Mnemonic::Shl,
        Mnemonic::Fclex => Mnemonic::Fnclex,
        Mnemonic::Finit => Mnemonic::Fninit,
        Mnemonic::Fsave => Mnemonic::Fnsave,
        Mnemonic::Fstcw => Mnemonic::Fnstcw,
        Mnemonic::Fstenv => Mnemonic::Fnstenv,
        Mnemonic::Fstsw => Mnemonic::Fnstsw,
        _ => mnemonic,
    }
}

fn public_operand_start(instruction: &Instruction) -> u32 {
    if matches!(
        instruction.mnemonic(),
        Mnemonic::Maskmovdqu | Mnemonic::Vmaskmovdqu
    ) && instruction.op_count() == 3
        && matches!(
            instruction.op_kind(0),
            OpKind::Memory
                | OpKind::MemorySegDI
                | OpKind::MemorySegEDI
                | OpKind::MemorySegRDI
                | OpKind::MemoryESDI
                | OpKind::MemoryESEDI
                | OpKind::MemoryESRDI
        )
    {
        1
    } else {
        0
    }
}

fn has_address_size_override(bytes: &[u8]) -> bool {
    for byte in bytes {
        match *byte {
            0x67 => return true,
            0x26 | 0x2e | 0x36 | 0x3e | 0x64 | 0x65 | 0x66 | 0xf0 | 0xf2 | 0xf3 => {}
            0x40..=0x4f => {}
            _ => return false,
        }
    }
    false
}

fn fill_operand(
    instruction: &Instruction,
    index: u32,
    access: OpAccess,
    displacement_offset: u8,
    displacement_size: u8,
    immediate_offset: u8,
    immediate_size: u8,
    immediate_offset2: u8,
    immediate_size2: u8,
) -> X86DecodedOperand {
    let kind = instruction.op_kind(index);
    let mut output = X86DecodedOperand {
        id: index as u8,
        visibility: 1,
        actions: access_mask(access),
        ..X86DecodedOperand::default()
    };
    #[allow(unreachable_patterns)]
    match kind {
        OpKind::Register => {
            let register = instruction.op_register(index);
            output.operand_type = OPERAND_REGISTER;
            output.size = register_width(register);
            output.value = X86OperandValue {
                reg: X86RegisterOperand {
                    value: register as u32,
                },
            };
        }
        OpKind::Memory
        | OpKind::MemorySegSI
        | OpKind::MemorySegESI
        | OpKind::MemorySegRSI
        | OpKind::MemorySegDI
        | OpKind::MemorySegEDI
        | OpKind::MemorySegRDI
        | OpKind::MemoryESDI
        | OpKind::MemoryESEDI
        | OpKind::MemoryESRDI => {
            let mut base = instruction.memory_base();
            let mut index_register = instruction.memory_index();
            match kind {
                OpKind::MemorySegSI => base = Register::SI,
                OpKind::MemorySegESI => base = Register::ESI,
                OpKind::MemorySegRSI => base = Register::RSI,
                OpKind::MemorySegDI | OpKind::MemoryESDI => base = Register::DI,
                OpKind::MemorySegEDI | OpKind::MemoryESEDI => base = Register::EDI,
                OpKind::MemorySegRDI | OpKind::MemoryESRDI => base = Register::RDI,
                _ => {}
            }
            if kind != OpKind::Memory {
                index_register = Register::None;
            }
            let next_ip = instruction.next_ip();
            let displacement = if base == Register::RIP || base == Register::EIP {
                instruction.ip_rel_memory_address().wrapping_sub(next_ip) as i64
            } else {
                instruction.memory_displacement64() as i64
            };
            let memory_info = instruction.memory_size().info();
            let size = memory_info.size().saturating_mul(8) as u16;
            output.operand_type = OPERAND_MEMORY;
            output.size = size;
            output.element_size = memory_info.element_size().saturating_mul(8) as u16;
            output.element_count = if memory_info.element_size() == 0 {
                1
            } else {
                (memory_info.size() / memory_info.element_size()).max(1) as u16
            };
            output.value = X86OperandValue {
                mem: X86MemoryOperand {
                    kind: 0,
                    segment: instruction.memory_segment() as u32,
                    base: base as u32,
                    index: index_register as u32,
                    scale: instruction.memory_index_scale() as u8,
                    disp: X86Displacement {
                        value: displacement,
                        offset: displacement_offset,
                        size: displacement_size.saturating_mul(8),
                    },
                },
            };
        }
        OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64 => {
            let width = match kind {
                OpKind::NearBranch16 => 16,
                OpKind::NearBranch32 => 32,
                _ => 64,
            };
            output.operand_type = OPERAND_IMMEDIATE;
            output.size = width;
            output.value = X86OperandValue {
                imm: X86ImmediateOperand {
                    is_signed: 1,
                    is_relative: 1,
                    value: X86ImmediateValue {
                        s: relative_target_delta(instruction, kind),
                    },
                    offset: immediate_offset,
                    size: immediate_size.saturating_mul(8),
                },
            };
        }
        OpKind::FarBranch16 | OpKind::FarBranch32 => {
            output.operand_type = OPERAND_POINTER;
            output.size = if kind == OpKind::FarBranch16 { 16 } else { 32 };
            output.value = X86OperandValue {
                ptr: X86PointerOperand {
                    segment: instruction.far_branch_selector(),
                    offset: instruction.far_branch32(),
                },
            };
        }
        OpKind::Immediate8
        | OpKind::Immediate8_2nd
        | OpKind::Immediate16
        | OpKind::Immediate32
        | OpKind::Immediate64
        | OpKind::Immediate8to16
        | OpKind::Immediate8to32
        | OpKind::Immediate8to64
        | OpKind::Immediate32to64 => {
            let (value, signed, encoded_bits) = immediate_value(instruction, kind);
            let semantic_bits = match kind {
                OpKind::Immediate8to16 => 16,
                OpKind::Immediate8to32 => 32,
                OpKind::Immediate8to64 | OpKind::Immediate32to64 => 64,
                _ => encoded_bits,
            };
            let second = kind == OpKind::Immediate8_2nd;
            output.operand_type = OPERAND_IMMEDIATE;
            output.size = semantic_bits as u16;
            output.value = X86OperandValue {
                imm: X86ImmediateOperand {
                    is_signed: signed as u8,
                    is_relative: 0,
                    value: X86ImmediateValue { u: value },
                    offset: if second {
                        immediate_offset2
                    } else {
                        immediate_offset
                    },
                    size: if second {
                        immediate_size2.saturating_mul(8)
                    } else {
                        immediate_size.saturating_mul(8)
                    },
                },
            };
        }
        _ => {
            output.operand_type = OPERAND_UNUSED;
        }
    }
    output
}

#[no_mangle]
pub unsafe extern "C" fn lsx_iced_decode(
    data: *const u8,
    size: usize,
    instruction_out: *mut X86DecodedInstruction,
    operands_out: *mut X86DecodedOperand,
    operand_capacity: usize,
) -> i32 {
    if data.is_null() || instruction_out.is_null() || size == 0 {
        return STATUS_INVALID_ARGUMENT;
    }
    let bytes = slice::from_raw_parts(data, size.min(15));
    let mut decoder = Decoder::with_ip(64, bytes, 0, DecoderOptions::NONE);
    let instruction = decoder.decode();
    if instruction.code() == Code::INVALID || instruction.len() == 0 {
        return STATUS_DECODE_FAILED;
    }
    let constants = decoder.get_constant_offsets(&instruction);
    let mut attributes = 0u64;
    if instruction.has_lock_prefix() {
        attributes |= ATTR_HAS_LOCK;
    }
    if instruction.has_rep_prefix() {
        attributes |= ATTR_HAS_REP | ATTR_HAS_REPE;
    }
    if instruction.has_repne_prefix() {
        attributes |= ATTR_HAS_REPNE;
    }
    let mut bytes_copy = [0u8; 15];
    bytes_copy[..instruction.len()].copy_from_slice(&bytes[..instruction.len()]);
    let operand_start = public_operand_start(&instruction);
    let op_count = instruction.op_count().saturating_sub(operand_start).min(5);
    ptr::write(
        instruction_out,
        X86DecodedInstruction {
            attributes,
            mnemonic: normalized_mnemonic(instruction.mnemonic()) as u32,
            length: instruction.len() as u8,
            encoding: instruction.encoding() as u8,
            address_width: if instruction.code_size() == iced_x86::CodeSize::Code64 {
                if has_address_size_override(&bytes_copy[..instruction.len()]) {
                    32
                } else {
                    64
                }
            } else {
                instruction.code_size() as u8
            },
            operand_width: operand_width(&instruction),
            operand_count: op_count as u8,
            operand_count_visible: op_count as u8,
            bytes: bytes_copy,
        },
    );
    if !operands_out.is_null() {
        for index in 0..operand_capacity {
            ptr::write(operands_out.add(index), X86DecodedOperand::default());
        }
        let mut info_factory = InstructionInfoFactory::new();
        let info = info_factory.info(&instruction);
        let count = (op_count as usize).min(operand_capacity);
        for index in 0..count {
            let source_index = index as u32 + operand_start;
            let mut operand = fill_operand(
                &instruction,
                source_index,
                info.op_access(source_index),
                constants.displacement_offset() as u8,
                constants.displacement_size() as u8,
                constants.immediate_offset() as u8,
                constants.immediate_size() as u8,
                constants.immediate_offset2() as u8,
                constants.immediate_size2() as u8,
            );
            operand.id = index as u8;
            ptr::write(
                operands_out.add(index),
                operand,
            );
        }
    }
    STATUS_OK
}

fn copy_text(text: &str, output: *mut c_char, capacity: usize) -> i32 {
    if output.is_null() || capacity == 0 {
        return STATUS_INVALID_ARGUMENT;
    }
    let bytes = text.as_bytes();
    let count = bytes.len().min(capacity - 1);
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), output.cast::<u8>(), count);
        *output.add(count) = 0;
    }
    count as i32
}

#[no_mangle]
pub unsafe extern "C" fn lsx_iced_format(
    data: *const u8,
    size: usize,
    runtime_address: u64,
    output: *mut c_char,
    capacity: usize,
) -> i32 {
    if data.is_null() || size == 0 {
        return STATUS_INVALID_ARGUMENT;
    }
    let bytes = slice::from_raw_parts(data, size.min(15));
    let mut decoder = Decoder::with_ip(64, bytes, runtime_address, DecoderOptions::NONE);
    let instruction = decoder.decode();
    if instruction.code() == Code::INVALID {
        return STATUS_DECODE_FAILED;
    }
    let mut formatter = IntelFormatter::new();
    let mut text = String::new();
    formatter.format(&instruction, &mut text);
    copy_text(&text, output, capacity)
}

#[no_mangle]
pub extern "C" fn lsx_iced_decoder_contract_version() -> u32 {
    0x0001_1501
}

#[no_mangle]
pub extern "C" fn lsx_iced_uses_vendor_decoder(_reserved: *const c_void) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    fn decode(bytes: &[u8]) -> (X86DecodedInstruction, [X86DecodedOperand; 10]) {
        let mut instruction = X86DecodedInstruction::default();
        let mut operands = [X86DecodedOperand::default(); 10];
        let status = unsafe {
            lsx_iced_decode(
                bytes.as_ptr(),
                bytes.len(),
                &mut instruction,
                operands.as_mut_ptr(),
                operands.len(),
            )
        };
        assert_eq!(status, STATUS_OK);
        (instruction, operands)
    }

    #[test]
    fn c_abi_layout_matches_cpp_contract() {
        assert_eq!(size_of::<X86Displacement>(), 16);
        assert_eq!(size_of::<X86MemoryOperand>(), 40);
        assert_eq!(size_of::<X86ImmediateOperand>(), 24);
        assert_eq!(size_of::<X86DecodedOperand>(), 56);
        assert_eq!(size_of::<X86DecodedInstruction>(), 40);
    }

    #[test]
    fn rip_relative_memory_preserves_signed_delta() {
        let (instruction, operands) = decode(&[0x48, 0x8b, 0x05, 0xf0, 0xff, 0xff, 0xff]);
        assert_eq!(instruction.mnemonic, Mnemonic::Mov as u32);
        assert_eq!(instruction.length, 7);
        assert_eq!(operands[0].operand_type, OPERAND_REGISTER);
        assert_eq!(unsafe { operands[0].value.reg.value }, Register::RAX as u32);
        assert_eq!(operands[1].operand_type, OPERAND_MEMORY);
        assert_eq!(unsafe { operands[1].value.mem.base }, Register::RIP as u32);
        assert_eq!(unsafe { operands[1].value.mem.disp.value }, -16);
        assert_eq!(operands[1].size, 64);
    }

    #[test]
    fn indirect_call_with_disp8_preserves_its_full_length() {
        let (instruction, operands) = decode(&[0xff, 0x50, 0x18]);
        assert_eq!(instruction.mnemonic, Mnemonic::Call as u32);
        assert_eq!(instruction.length, 3);
        assert_eq!(instruction.operand_count, 1);
        assert_eq!(operands[0].operand_type, OPERAND_MEMORY);
        assert_eq!(unsafe { operands[0].value.mem.base }, Register::RAX as u32);
        assert_eq!(unsafe { operands[0].value.mem.disp.value }, 0x18);
        assert_eq!(operands[0].size, 64);
    }

    #[test]
    fn sign_extended_immediate_keeps_semantic_and_encoded_widths() {
        let (instruction, operands) = decode(&[0x48, 0x83, 0xc0, 0xff]);
        assert_eq!(instruction.mnemonic, Mnemonic::Add as u32);
        assert_eq!(instruction.operand_width, 64);
        assert_eq!(operands[1].operand_type, OPERAND_IMMEDIATE);
        assert_eq!(operands[1].size, 64);
        assert_eq!(unsafe { operands[1].value.imm.size }, 8);
        assert_eq!(unsafe { operands[1].value.imm.is_signed }, 1);
        assert_eq!(unsafe { operands[1].value.imm.value.s }, -1);
    }

    #[test]
    fn explicit_length_string_compare_observes_rex_w() {
        let (instruction, operands) = decode(&[0x66, 0x48, 0x0f, 0x3a, 0x61, 0xc1, 0x00]);
        assert_eq!(instruction.mnemonic, Mnemonic::Pcmpestri as u32);
        assert_eq!(instruction.operand_width, 64);
        assert_eq!(operands[0].size, 128);
        assert_eq!(operands[1].size, 128);
        assert_eq!(operands[2].operand_type, OPERAND_IMMEDIATE);
    }

    #[test]
    fn vmaskmovdqu_exposes_two_xmm_sources_and_implicit_rdi_destination() {
        let (instruction, operands) = decode(&[0xc5, 0xf9, 0xf7, 0xc1]);
        assert_eq!(instruction.mnemonic, Mnemonic::Vmaskmovdqu as u32);
        assert_eq!(instruction.operand_count, 2);
        assert_eq!(instruction.operand_count_visible, 2);
        assert_eq!(instruction.address_width, 64);
        assert_eq!(operands[0].operand_type, OPERAND_REGISTER);
        assert_eq!(operands[1].operand_type, OPERAND_REGISTER);
        assert_eq!(unsafe { operands[0].value.reg.value }, Register::XMM0 as u32);
        assert_eq!(unsafe { operands[1].value.reg.value }, Register::XMM1 as u32);
        assert_eq!(operands[0].size, 128);
        assert_eq!(operands[1].size, 128);
    }

    #[test]
    fn legacy_maskmovdqu_uses_the_same_public_operand_contract() {
        let (instruction, operands) = decode(&[0x66, 0x0f, 0xf7, 0xc1]);
        assert_eq!(instruction.mnemonic, Mnemonic::Maskmovdqu as u32);
        assert_eq!(instruction.operand_count, 2);
        assert_eq!(instruction.operand_count_visible, 2);
        assert_eq!(operands[0].operand_type, OPERAND_REGISTER);
        assert_eq!(operands[1].operand_type, OPERAND_REGISTER);
        assert_eq!(unsafe { operands[0].value.reg.value }, Register::XMM0 as u32);
        assert_eq!(unsafe { operands[1].value.reg.value }, Register::XMM1 as u32);
        assert_eq!(operands[0].size, 128);
        assert_eq!(operands[1].size, 128);
    }

    #[test]
    fn fxch_preserves_iced_explicit_st0_and_stack_register_operands() {
        let (instruction, operands) = decode(&[0xd9, 0xc9]);
        assert_eq!(instruction.mnemonic, Mnemonic::Fxch as u32);
        assert_eq!(instruction.operand_count, 2);
        assert_eq!(operands[0].operand_type, OPERAND_REGISTER);
        assert_eq!(operands[1].operand_type, OPERAND_REGISTER);
        assert_eq!(unsafe { operands[0].value.reg.value }, Register::ST0 as u32);
        assert_eq!(unsafe { operands[1].value.reg.value }, Register::ST1 as u32);
    }

    #[test]
    fn f16c_float_to_half_exposes_destination_source_and_rounding_control() {
        let (instruction, operands) = decode(&[0xc4, 0xe3, 0x79, 0x1d, 0xe4, 0x00]);
        assert_eq!(instruction.mnemonic, Mnemonic::Vcvtps2ph as u32);
        assert_eq!(instruction.operand_count, 3);
        assert_eq!(unsafe { operands[0].value.reg.value }, Register::XMM4 as u32);
        assert_eq!(unsafe { operands[1].value.reg.value }, Register::XMM4 as u32);
        assert_eq!(operands[0].size, 128);
        assert_eq!(operands[1].size, 128);
        assert_eq!(operands[2].operand_type, OPERAND_IMMEDIATE);
        assert_eq!(unsafe { operands[2].value.imm.value.u }, 0);
    }
}
