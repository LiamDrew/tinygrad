#!/usr/bin/env python3
"""tinyasm2 — Apple G15/M3 GPU assembler: kernel text -> runnable Metal binary archive. Syntax in README."""
import struct, hashlib, sys
from tinygrad.runtime.support.agx import air

from tinygrad.renderer.agx import dsl

def reg(tok):
    assert tok[0]=='r'; return int(tok[1:])
def i_load(dstreg, slot, first, more_follow): return dsl.LOAD.encode(first=int(first), more=int(more_follow), dst=dstreg, slot=slot)
def i_wait(): return dsl.WAIT.encode()
def i_falu_imm(dst,a,imm,mul=0):
    b,neg=dsl.e4m3(imm)
    return dsl.FALU.encode(srcb=b, mul=mul, mod=(dsl.FADD_NEG if neg else dsl.FADD_ADD_IMM)>>1, srca=dsl.reg_operand(a), bmode=0x80)
def i_falu_reg(dst,a,bb,mul=0): return dsl.FALU.encode(srcb=dsl.reg_operand(bb), mul=mul, mod=dsl.FADD_NEG>>1, srca=dsl.reg_operand(a))
def i_store(srcreg, slot, is_result):
    src=dsl.STORE_RESULT if is_result else (0x56 if srcreg==0 else dsl.STORE_RESULT+srcreg*2)
    return dsl.STORE.encode(src=src, slot=slot)
def movimm_bytes(dst, f): return dsl.MOVIMM.encode(dst=dst, **dsl.movimm_fields(f))

PREAMBLE_IDX = bytes.fromhex('030007000200000060000e000000')  # thread-index preamble
EPILOGUE     = dsl.BARRIER.encode() + dsl.STOP.encode()

def assemble_text(program):
    """Return (text_bytes, N, written slots). Builds preamble(0..0x40) + main + epilogue."""
    N=0; main=bytearray(); load_i=0; written=set()
    main+=dsl.GET_TID.encode()                            # thread-index prologue
    lines=[l.split(';')[0].strip() for l in program.splitlines()]  # ';' comments only ('#'=imm)
    lines=[l for l in lines if l]
    body=[]
    for l in lines:
        p=l.replace(',',' ').split()
        op=p[0]
        if op=='.buffers': N=int(p[1]); continue
        body.append(p)
    total_loads=sum(1 for p in body if p[0]=='load')
    fadd_wrote_result=False
    for p in body:
        op=p[0]
        if op=='load':
            d=reg(p[1]); slot=int(p[2]); N=max(N,slot+1)
            main+=i_load(d,slot, load_i==0, load_i<total_loads-1); load_i+=1
        elif op=='wait':
            main+=i_wait()
        elif op in ('fadd','fmul'):
            d=reg(p[1]); a=reg(p[2]); mul=int(op=='fmul')
            if p[3].startswith('#'):
                main+=i_falu_imm(d,a,float(p[3][1:]),mul)
            else:
                main+=i_falu_reg(d,a,reg(p[3]),mul)
            fadd_wrote_result=True
        elif op=='movimm':
            d=reg(p[1]); main+=movimm_bytes(d,float(p[2].lstrip('#')))
        elif op=='store':
            s=reg(p[1]); slot=int(p[2]); N=max(N,slot+1); written.add(slot)
            main+=i_store(s,slot, fadd_wrote_result)
        else:
            raise ValueError(f"unknown op {op}")
    main+=EPILOGUE
    text=bytearray(PREAMBLE_IDX)
    while len(text)<0x40: text+=dsl.NOP.encode()
    text+=main
    return bytes(text), N, written

# ==== __GPU_METADATA FlatBuffer for N device buffers (see README "Metadata") ====
def flatbuffer(objs, root):
    """Minimal FlatBuffer emitter. objs: name -> ('tbl', {slot: ('u8'|'u32', v) | ('ref', name)}) | ('vec', [names]) | ('str', s) | ('raw', bytes).
    Apple's layout: each vtable glued before its table; tables, vectors and strings 4-aligned; u32 fields before u8."""
    def al(x): return (x + 3) & ~3
    pos = {}; cur = 4; blobs = {}                       # name -> (vtable | None, body, {body offset: ref target})
    for name, (kind, v) in objs.items():
        if kind == 'tbl':
            off = {}; p = 4
            for slot, (t, _) in sorted(v.items(), key=lambda kv: (kv[1][0] == 'u8', kv[0])):
                off[slot] = p; p += 1 if t == 'u8' else 4
            tb = al(p); nslots = max(v) + 1 if v else 0
            vt = struct.pack('<HH', 4 + 2 * nslots, tb) + b''.join(struct.pack('<H', off.get(i, 0)) for i in range(nslots))
            body = bytearray(tb)
            for slot, (t, val) in v.items():
                if t == 'u8': body[off[slot]] = val
                elif t == 'u32': struct.pack_into('<I', body, off[slot], val)
            cur = al(cur + len(vt)); pos[name] = cur; cur += tb
            blobs[name] = (vt, body, {off[slot]: val for slot, (t, val) in v.items() if t == 'ref'})
        else:
            if kind == 'vec': body = struct.pack('<I', len(v)) + bytes(4 * len(v)); refs = {4 + 4 * i: n for i, n in enumerate(v)}
            elif kind == 'str': body = struct.pack('<I', len(v)) + v.encode() + b'\0'; refs = {}
            else: body = v; refs = {}
            cur = al(cur); pos[name] = cur; cur += len(body); blobs[name] = (None, bytearray(body), refs)
    out = bytearray(al(cur)); struct.pack_into('<I', out, 0, pos[root])
    for name, (vt, body, refs) in blobs.items():
        base = pos[name]
        if vt is not None: out[base - len(vt):base] = vt; struct.pack_into('<i', body, 0, len(vt))   # soffset table -> vtable
        for o, tgt in refs.items(): struct.pack_into('<I', body, o, pos[tgt] - (base + o))           # uoffset, self-relative
        out[base:base + len(body)] = body
    return bytes(out)

def gen_metadata(N, written=None):
    """Metadata for N buffers with an explicit written set (default {N-1}). Every object here is dereferenced by the
    AGX driver (ProgramBindingRemap segfaults without it); hardware-verified N=1..3."""
    written = {N - 1} if written is None else written
    spans = [dict(f0=3, f2=2 * N)] + ([dict(f0=6, f2=2, f3=2 * N)] if N % 2 else [])   # uniform-register spans: 2N address words + pad to 4
    o = {'root':   ('tbl', {0: ('ref', 'fn'), 3: ('ref', 'nameh')}),
         'nameh':  ('tbl', {1: ('ref', 'name')}),  'name': ('str', 'agc.main'),
         'fn':     ('tbl', {2: ('ref', 'spans'), 3: ('u32', 8 * N), 4: ('ref', 'descs'), 13: ('ref', 'zeros'), 26: ('ref', 'cpoolv'),
                            **{s: ('ref', 'empty') for s in (6, 8, 10, 12, 27)}}),
         'cpoolv': ('vec', ['cpool']),
         'cpool':  ('tbl', {0: ('ref', 'cpools'), 1: ('u8', 3), 2: ('u8', 1), 3: ('u32', 16)}),  'cpools': ('str', 'agc.main.constant_program'),
         'zeros':  ('raw', struct.pack('<I', 8 * N) + bytes(8 * N)),
         'empty':  ('vec', []),
         'descs':  ('vec', [f'd{i}' for i in range(N)]),
         'spans':  ('vec', [f's{i}' for i in range(len(spans))])}
    for i, sp in enumerate(spans):
        o[f's{i}'] = ('tbl', {0: ('u8', sp['f0']), 2: ('u32', sp['f2']), **({3: ('u32', sp['f3'])} if 'f3' in sp else {})})
    for i in range(N):                                   # buffer descriptor: kind 5, index, address-register offset 2i, written flag
        o[f'd{i}'] = ('tbl', {0: ('u8', 5), **({1: ('u32', i), 2: ('u32', 2 * i)} if i else {}), **({3: ('u8', 1)} if i in written else {})})
    return flatbuffer(o, 'root')

# ==== applegpu GPU-exec slice + fat archive (see README "Shell") ====
# ---- __AIR_DATA note-record template (recovered once from the reference archive; computed fields patched in) ----
AIRDATA = bytes.fromhex('f0030000000000000100000000000000180400000000000001000000000000004004000000000000010000000000000078040000000000000100000000000000000000000000000000000000000000009c040000000000000100000000000000a4040000000000000200000000000000b4050000000000000400000000000000bd3743188816301cb981f15b5b26d029000e00000000000040100000000000000000000000000000c28abc9c2a512016e08b92885a13087771ea04e10f93b5cf816f8502424ea1e30000000000000000793d28d9d268537bfe254b3bbe5faced0589e2d8a9142dbcfc1ca3f7bdac1555a00c0000000000005c0100000000000010000000000000000000000000000000f0060000b0050000c00500002c010000c00500002c0100000000000001000000000000000100000000000000c28abc9c2a512016e08b92885a13087771ea04e10f93b5cf816f8502424ea1e30000000000000000000000000000000000000000000000000000000000000000e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855000000000204000003040000000000c000000000010000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008040000000000000000000004000040f0060000b0050000c00500002c010000000000000000000000000000')

# ---- inner __compute Mach-O: an object file with our __text, __GPU_METADATA, symtab and the static __GPU_*_MD ----
GPU_LD_MD      = bytes.fromhex('14000000000000000c000c0008000000000004000c00000014000000580000000c00080000000000000007000c000000000000014000200000000000180000000000140010001c0000000000000000000000000000000000000000000f00000000000000000000000000000000000000000004004000000001000000000000000000000140000000140000001800000004000000010000006b000000010000000000000007000000636f6d7075746500')
GPU_ARCH_LD_MD = bytes.fromhex('100000000c000e0008000000000004000c0000001000000020000000000006000800040006000000040000000000000008000800000007000800000000000001')
GPU_STATS_MD   = bytes(96)                  # compile statistics; zeroed, driver ignores it (LD_MD/ARCH_LD_MD it parses: zeroing segfaults)
STRTAB = b'\0_agc.main\0_agc.main.constant_program\0'.ljust(40, b'\0')

def _sect_hdr(sectname, segname, addr, size, off, align, flags=0):
    return (sectname.ljust(16, b'\x00') + segname.ljust(16, b'\x00')
            + struct.pack('<QQIIIIIIII', addr, size, off, align, 0, 0, flags, 0, 0, 0))

def build_inner(text, md):
    """Object Mach-O: header + segment(6 sections) + LC_SYMTAB + LC_DYSYMTAB, then the section bodies."""
    def al(x, a): return (x + a - 1) & ~(a - 1)
    hdr_size = 32 + 72 + 6 * 80 + 24 + 80                  # header, segment + sections, symtab, dysymtab = 688
    text_off = hdr_size
    md_off = al(text_off + len(text), 8); md_addr = md_off - text_off
    symoff = al(md_off + len(md), 4) + 4                    # two nlist entries follow a 4-byte pad
    stroff = symoff + 32
    tail_off = stroff + len(STRTAB); tail_addr = md_addr + len(md) + 4       # Apple's addr for the trailing sections
    ld_off = tail_off; arch_off = ld_off + len(GPU_LD_MD); stats_off = arch_off + len(GPU_ARCH_LD_MD)
    total = stats_off + len(GPU_STATS_MD)

    b = bytearray(struct.pack('<IiiIIIII', 0xfeedfacf, 0x1000013, 0x1a3, 1, 3, hdr_size - 32, 0, 0))
    seg_size = total - 760                                  # Apple's value; not a span the loader checks
    b += struct.pack('<II', 0x19, 72 + 6 * 80) + bytes(16) + struct.pack('<QQQQiiII', 0, seg_size, text_off, seg_size, 7, 7, 6, 0)
    b += _sect_hdr(b'__text', b'__TEXT', 0, len(text), text_off, 6, 0x80000400)
    b += _sect_hdr(b'__compute', b'__GPU_METADATA', md_addr, len(md), md_off, 3)
    b += _sect_hdr(b'__compute', b'__GPU_REMARKS_MD', tail_addr, 0, ld_off, 3)
    b += _sect_hdr(b'__compute', b'__GPU_LD_MD', tail_addr, len(GPU_LD_MD), ld_off, 3)
    b += _sect_hdr(b'__compute', b'__GPU_ARCH_LD_MD', tail_addr, len(GPU_ARCH_LD_MD), arch_off, 3)
    b += _sect_hdr(b'__compute', b'__GPU_STATS_MD', tail_addr, len(GPU_STATS_MD), stats_off, 3)
    b += struct.pack('<IIIIII', 0x2, 24, symoff, 2, stroff, len(STRTAB))
    b += struct.pack('<II', 0xb, 80) + struct.pack('<18I', 0, 0, 0, 2, 2, *[0] * 13)
    assert len(b) == text_off
    b += text; b += bytes(md_off - len(b)); b += md; b += bytes(symoff - len(b))
    b += struct.pack('<IBBHQ', 1, 0x0f, 1, 0, 0x40)         # _agc.main @ 0x40 in __text (after the preamble)
    b += struct.pack('<IBBHQ', 11, 0x0f, 1, 0, 0)           # _agc.main.constant_program
    b += STRTAB + GPU_LD_MD + GPU_ARCH_LD_MD + GPU_STATS_MD
    assert len(b) == total
    return bytes(b)

# ---- outer slice0: two-level Mach-O + __AIR_DATA note records ----
_NOTE_OWNERS = [b'AIR_METALLIB', b'AIR_MODULE', b'AIR_DESCRIPTOR', b'AIR_OBJECT',
                b'AIR_OBJECT_INDEX', b'AIR_PIPELINE', b'AIR_HASHES', b'AIR_STRTABLE']
_LC32 = bytes.fromhex('0100000000070f0000000000020000000704000000ff177d0304000000000200')

def build_slice0(inner, N, fnhash):
    R = 16                                  # __reflection: 16 zero bytes; Metal needs a nonzero-size section, not its contents
    compute_off = 0x5c0 + R
    total = compute_off + len(inner)
    desc_off = total                        # __descriptor and __metallib: size-0 headers, contents dropped
    ml_off = desc_off + 352

    b = bytearray()
    b += struct.pack('<IiiIIIII', 0xfeedfacf, 0x1000013, 0x1a3, 0xd, 12, 848, 0, 0)
    b += struct.pack('<II', 0x19, 72) + b'__AIR_DATA'.ljust(16, b'\x00')
    b += struct.pack('<QQQQiiII', 0, 0x5c0, 0, 0x5c0, 1, 1, 0, 0)
    for i, owner in enumerate(_NOTE_OWNERS):
        b += struct.pack('<II', 0x31, 40) + owner.ljust(16, b'\x00')
        b += struct.pack('<QQ', 0x370 + 16 * i, 16)
    tsz = total - 0x5c0
    b += struct.pack('<II', 0x19, 392) + b'__TEXT'.ljust(16, b'\x00')
    b += struct.pack('<QQQQiiII', 0x5c0, tsz, 0x5c0, tsz, 1, 1, 4, 0)
    b += _sect_hdr(b'__reflection', b'__TEXT', 0x5c0, R, 0x5c0, 4)
    b += _sect_hdr(b'__compute', b'__TEXT', compute_off, len(inner), compute_off, 4)
    b += _sect_hdr(b'__descriptor', b'__TEXT', desc_off, 0, desc_off, 4)
    b += _sect_hdr(b'__metallib', b'__TEXT', ml_off, 0, ml_off, 4)
    b += struct.pack('<II', 0x32, 40) + _LC32
    b += struct.pack('<II', 0x1b, 24) + fnhash[:16]                  # LC_UUID = fnhash[:16]
    assert len(b) == 0x370, len(b)
    ad = bytearray(AIRDATA)
    def p32(o, v): struct.pack_into('<I', ad, o - 0x370, v)
    def p64(o, v): struct.pack_into('<Q', ad, o - 0x370, v)
    def pb(o, v): ad[o - 0x370:o - 0x370 + len(v)] = v
    # AIR_METALLIB record @0x3f0 stays as the template's: a zeroed record segfaults Metal, past-EOF is skipped.
    pb(0x418, fnhash)                # AIR_MODULE hash
    p64(0x460, desc_off)             # AIR_DESCRIPTOR offset
    p32(0x480, compute_off); p32(0x484, len(inner))
    p32(0x488, 0x5c0); p32(0x48c, R)                     # __reflection off/size
    p32(0x490, 0x5c0); p32(0x494, R)
    pb(0x4ac, fnhash)                # AIR_PIPELINE hash
    p32(0x5a4, compute_off); p32(0x5a8, len(inner))
    p32(0x5ac, 0x5c0); p32(0x5b0, R)
    b += ad
    assert len(b) == 0x5c0
    return bytes(b) + bytes(R) + inner

# ---- fat archive: [applegpu slice, air64 metallib slice] + NBUF trailer ----
def build_archive(new_text, new_md, N):
    """Return the complete archive bytes for our __text/__GPU_METADATA and N buffers."""
    srcname = 'k_' + hashlib.sha256(new_text + new_md).hexdigest()[:16]   # per-program AIR -> unique archive key
    ml = air.gen_mtlb(air.gen_air(N, srcname))
    fnhash = hashlib.sha256(ml[231:]).digest()      # SHA-256(wrapper+bitcode+pad) keys both slices
    s0 = build_slice0(build_inner(new_text, new_md), N, fnhash)
    off0 = 48
    off1 = (off0 + len(s0) + 15) & ~15
    out = bytearray(struct.pack('>II', 0xcafebabe, 2))
    out += struct.pack('>iiIII', 0x1000013, 0x1a3, off0, len(s0), 4)
    out += struct.pack('>iiIII', 0x1000017, 0x0b, off1, len(ml), 4)
    out += bytes(off0 - len(out)) + s0
    out += bytes(off1 - len(out)) + ml
    out += b'NBUF' + struct.pack('<I', N)
    return bytes(out)

def build(program):
    """Assemble `program` and return (archive_bytes, N)."""
    text,N,written=assemble_text(program)
    md=gen_metadata(N, written)
    return build_archive(text, md, N), N

if __name__=='__main__':
    out=sys.argv[2] if len(sys.argv)>2 else 'out.bin'
    archive,N=build(open(sys.argv[1]).read())
    open(out,'wb').write(archive)
    print(f"{out}: N={N} archive={len(archive)}B")
