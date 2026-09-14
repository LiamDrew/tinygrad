#!/usr/bin/env python3
"""AIR bitcode for the N-buffer empty-body kernel 'k', and its metallib. See README "AIR module"."""
import struct, hashlib

# ==== LLVM bitstream reader/writer that preserves the exact encoding path (byte-exact round trip) ====
END_BLOCK=0; ENTER_SUBBLOCK=1; DEFINE_ABBREV=2; UNABBREV_RECORD=3; FIRST_APP_ABBREV=4
ENC_FIXED=1; ENC_VBR=2; ENC_ARRAY=3; ENC_CHAR6=4; ENC_BLOB=5

class BitWriter:
    def __init__(s): s.bits=bytearray(); s.bitpos=0; s.poss={}
    def write(s,v,n):
        for i in range(n):
            if s.bitpos>=len(s.bits)*8: s.bits.append(0)
            if (v>>i)&1: s.bits[s.bitpos>>3]|=1<<(s.bitpos&7)
            s.bitpos+=1
    def vbr(s,v,n):
        hi=1<<(n-1); mask=hi-1
        while True:
            if v<hi: s.write(v,n); return
            s.write((v&mask)|hi,n); v>>=(n-1)
    def align32(s):
        while s.bitpos&31: s.write(0,1)
    def patch32(s,pos,v):
        for i in range(32):
            p=pos+i
            if (v>>i)&1: s.bits[p>>3]|=1<<(p&7)
            else: s.bits[p>>3]&=~(1<<(p&7))

class Abbrev:
    def __init__(s,ops): s.ops=ops

def w_abbrev_def(w,ab):
    w.vbr(len(ab.ops),5)
    for enc,val in ab.ops:
        if enc=='literal': w.write(1,1); w.vbr(val,8)
        else:
            w.write(0,1); w.write(enc,3)
            if enc in (ENC_FIXED,ENC_VBR): w.vbr(val,5)

def w_abbrev_rec(w,ab,vals):
    i=0; vi=0
    while i<len(ab.ops):
        enc,val=ab.ops[i]
        if enc=='literal': vi+=1
        elif enc==ENC_FIXED:
            if val: w.write(vals[vi],val)
            vi+=1
        elif enc==ENC_VBR: w.vbr(vals[vi],val); vi+=1
        elif enc==ENC_CHAR6: w.write(vals[vi],6); vi+=1
        elif enc==ENC_ARRAY:
            arr=vals[vi]; w.vbr(len(arr),6); ee,ev=ab.ops[i+1]
            for x in arr:
                if ee=='literal': pass
                elif ee==ENC_FIXED:
                    if ev: w.write(x,ev)
                elif ee==ENC_VBR: w.vbr(x,ev)
                elif ee==ENC_CHAR6: w.write(x,6)
            vi+=1; i+=1
        elif enc==ENC_BLOB:
            blob=vals[vi]; w.vbr(len(blob),6); w.align32()
            for b in blob: w.write(b,8)
            w.align32(); vi+=1
        i+=1

# Tree items: ('abbrevrec', abbrev_id, vals) | ('unabbrev', code, ops) | ('def', Abbrev) | ('block', Block)
class Block:
    def __init__(s,bid,width): s.bid=bid; s.width=width; s.items=[]

cur_bi=[None]   # BLOCKINFO SETBID being defined (abbrevs in block 0 register to this bid)
def write_subblock(w,width,sub,binfo):
    w.poss.setdefault(sub.bid,w.bitpos)          # enter bit of each block id (first occurrence)
    w.write(ENTER_SUBBLOCK,width); w.vbr(sub.bid,8); w.vbr(sub.width,4); w.align32()
    lenpos=w.bitpos; w.write(0,32)
    write_block_body(w,sub,binfo)
    w.patch32(lenpos,(w.bitpos-lenpos-32)//32)

def write_block_body(w,blk,binfo):
    saved=cur_bi[0]
    if blk.bid==0: cur_bi[0]=None
    abbrevs=list(binfo.get(blk.bid,[]))
    for it in blk.items:
        if it[0]=='block': write_subblock(w,blk.width,it[1],binfo)
        elif it[0]=='def':
            w.write(DEFINE_ABBREV,blk.width); w_abbrev_def(w,it[1])
            if blk.bid!=0: abbrevs.append(it[1])
        elif it[0]=='unabbrev':
            w.write(UNABBREV_RECORD,blk.width); w.vbr(it[1],6); w.vbr(len(it[2]),6)
            for o in it[2]: w.vbr(o,6)
            if blk.bid==0 and it[1]==1: cur_bi[0]=it[2][0]
        elif it[0]=='abbrevrec':
            aid=it[1]; w.write(aid,blk.width); w_abbrev_rec(w,abbrevs[aid-FIRST_APP_ABBREV],it[2])
    w.write(END_BLOCK,blk.width); w.align32()
    cur_bi[0]=saved

def emit(top,binfo):
    """Emit magic + top-level blocks. Returns (bytes, {bid: enter bit})."""
    w=BitWriter()
    for c in (0x42,0x43,0xc0,0xde): w.write(c,8)
    cur_bi[0]=None
    for blk in top: write_subblock(w,2,blk,binfo)
    return bytes(w.bits), w.poss

# ==== AIR module tree ====

# ---- tree constructors: D(A(...)) abbrev def, U(code,ops) unabbrev rec, R(aid,vals) abbrev rec ----
ENC_FIXED, ENC_VBR, ENC_ARRAY, ENC_CHAR6, ENC_BLOB = 1, 2, 3, 4, 5
def L(v): return ('literal', v)
def F(n): return (ENC_FIXED, n)
def V(n): return (ENC_VBR, n)
ARR = (ENC_ARRAY, None); C6 = (ENC_CHAR6, None); BLOB = (ENC_BLOB, None)
def A(*ops): return Abbrev(list(ops))
def D(ab):   return ('def', ab)
def U(code, ops): return ('unabbrev', code, list(ops))
def R(aid, *vals): return ('abbrevrec', aid, list(vals))
def blk(bid, width, *items):
    b = Block(bid, width); b.items = list(items); return b
def B(b): return ('block', b)
def _s(txt):  return [ord(c) for c in txt]
def _z(txt):  return _s(txt) + [0]

def _grp(gid, pidx, *attrs):
    """PARAMATTR group entry: int = enum attr, str = string attr, (k,v) = key=value attr."""
    ops = [gid, pidx]
    for a in attrs:
        if isinstance(a, int):                  ops += [0, a]
        elif isinstance(a, tuple) and len(a) == 2: ops += [4] + _z(a[0]) + _z(a[1])
        else:                                   ops += [3] + _z(a[0] if isinstance(a, tuple) else a)
    return U(3, ops)

NOCAPTURE, NOUNDEF = 11, 68     # LLVM ATTR_KIND codes (names plausible, values exact)

MD_STRINGS = [                  # METADATA string slots for N=1; base(N) inserts b1..b{N-1} before air.thread_position_in_grid
    'SDK Version', 'wchar_size', 'frame-pointer',
    'air.max_device_buffers', 'air.max_constant_buffers',
    'air.max_threadgroup_buffers', 'air.max_textures',
    'air.max_read_write_textures', 'air.max_samplers',
    'Apple metal version 32023.620 (metalfe-32023.620)',
    'Metal',
    'air.compile.denorms_disable', 'air.compile.fast_math_enable',
    'air.compile.framebuffer_fetch_enable',
    'air.buffer', 'air.location_index', 'air.read_write',
    'air.address_space', 'air.arg_type_size', 'air.arg_type_align_size',
    'air.arg_type_name', 'float', 'air.arg_name', 'b0', 'air.arg_unused',
    'air.thread_position_in_grid', 'uint', 't',
]

FN_NAME  = 'k'
PRODUCER = '32023.620'
TRIPLE   = 'air64-apple-macosx15.0.0'
DATALAYOUT = ('e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64'
              '-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64'
              '-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256'
              '-v512:512:512-v1024:1024:1024-n8:16:32')
STRTAB_BLOB = (FN_NAME + PRODUCER + TRIPLE).encode()   # offsets: k@0, producer@1, triple@10

# LLVM irsymtab blob; string refs are (offset,len) into STRTAB. Not load-bearing, reproduced exactly.
SYMTAB_BLOB = struct.pack('<28I',
    3,                  # irsymtab version 3
    1, 9,               # producer strref
    0x4c,               # module table offset
    1, 0x58, 0,         # unexplained
    0x58, 1, 0x70, 0,   # unexplained (0x58,0x70 = blob offsets)
    10, 0x18,           # triple strref
    0,                  # unexplained
    1, 1, 0,            # unexplained
    0x70, 0, 0,         # unexplained
    1, 0, 0,            # unexplained
    1, 0,               # unexplained
    1, 0xffffffff,      # unexplained
    0x2400,             # symbol flags for 'k' (unexplained)
)

def _md_strings_rec(strings):
    """MD STRINGS record [35, count, lentab-bytes, blob]: VBR6 length table (32-bit aligned) + string bytes."""
    w = BitWriter()
    for x in strings: w.vbr(len(x), 6)
    w.align32()
    lentab = bytes(w.bits)
    return R(8, 35, len(strings), len(lentab), lentab + ''.join(strings).encode('latin1'))

def vbr6bits(v):
    c = 1
    while v >= 32: v >>= 5; c += 1
    return 6 * c

def rec_bits(width, it):
    """Bit size of an UNABBREV record as emitted."""
    return width + vbr6bits(it[1]) + vbr6bits(len(it[2])) + sum(vbr6bits(o) for o in it[2])

def _c6(txt):
    """LLVM char6 encoding: a-z 0-25, A-Z 26-51, 0-9 52-61, '.' 62, '_' 63."""
    return ['abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._'.index(c) for c in txt]

def base(N, srcname='k'):
    """Return (top_blocks, blockinfo): the module tree for N buffers. Record order is positional; do not reorder."""
    blockinfo_block = blk(0, 2,                  # abbrevs registered for blocks 14, 11, 12 (ids 4..)
        U(1, [14]),                              # SETBID VALUE_SYMTAB
        D(A(F(3), V(8), ARR, F(8))),             # 4: VST entry
        D(A(L(1), V(8), ARR, F(7))),             # 5: VST_CODE_ENTRY
        D(A(L(1), V(8), ARR, C6)),               # 6: VST_CODE_ENTRY char6
        D(A(L(2), V(8), ARR, C6)),               # 7: VST_CODE_BBENTRY char6
        U(1, [11]),                              # SETBID CONSTANTS
        D(A(L(1), F(4))),                        # 4: SETTYPE
        D(A(L(4), V(8))),                        # 5: CST_INTEGER (signed VBR: 2*v)
        D(A(L(11), F(4), F(4), V(8))),           # 6: CST_CODE_CE_CAST(?)
        D(A(L(2))),                              # 7: CST_NULL
        U(1, [12]),                              # SETBID FUNCTION
        D(A(L(20), V(6), F(4), V(4), F(1))),     # 4: INST_LOAD
        D(A(L(56), V(6), F(4))),                 # 5: unexplained inst
        D(A(L(56), V(6), F(4), F(8))),           # 6: unexplained inst
        D(A(L(2), V(6), V(6), F(4))),            # 7: INST_BINOP
        D(A(L(2), V(6), V(6), F(4), F(8))),      # 8: INST_BINOP + flags
        D(A(L(3), V(6), F(4), F(4))),            # 9: INST_CAST
        D(A(L(10))),                             # 10: INST_RET (void)
        D(A(L(10), V(6))),                       # 11: INST_RET (value)
        D(A(L(15))),                             # 12: INST_UNREACHABLE
        D(A(L(43), F(1), F(4), ARR, V(6))),      # 13: INST_GEP
    )

    # ---- integer constants: module values after fn 'k' (value 0); i32 5..N are appended when N >= 5 ----
    ints = [2, 1, 4, 7, 31, 128, 8, 16, 0, 3] + list(range(5, N + 1))
    vidx = {v: k + 1 for k, v in enumerate(ints)}          # value index of i32 v
    T7 = len(ints) + 1                                     # value index of the type-7 SDK constant

    # ---- METADATA slots: strings, then one slot per VALUE/NODE record ----
    P = MD_STRINGS.index('air.thread_position_in_grid')
    strings = MD_STRINGS[:P] + [f'b{i}' for i in range(1, N)] + MD_STRINGS[P:]
    st = {x: k for k, x in enumerate(strings)}
    slot = [len(strings)]                                  # next free slot
    def nxt(): slot[0] += 1; return slot[0] - 1
    values = []                                            # (name, record)
    def val(name, rec): values.append(rec); return nxt()
    v = {}                                                 # name -> slot
    v['i2'] = val('i2', U(2, [3, vidx[2]]))
    v['sdk'] = val('sdk', U(2, [7, T7]))
    for n in (1, 4, 7, 31, 128, 8, 16, 0, 3): v[f'i{n}'] = val(n, U(2, [3, vidx[n]]))
    v['fn'] = val('fn', U(2, [5, 0]))                      # fn-ptr value 0 = kernel 'k'
    for n in range(5, N + 1): v[f'i{n}'] = val(n, U(2, [3, vidx[n]]))
    r = lambda name: v[name] + 1                           # NODE operand = slot + 1 (0 = null)
    S = lambda name: st[name] + 1
    nodes = []
    def node(name, ops): nodes.append(U(3, ops)); v[name] = nxt()
    node('f.sdk',   [r('i2'), S('SDK Version'), r('sdk')])
    node('f.wchar', [r('i1'), S('wchar_size'), r('i4')])
    node('f.fp',    [r('i7'), S('frame-pointer'), r('i2')])
    node('f.dev',   [r('i7'), S('air.max_device_buffers'), r('i31')])
    node('f.cst',   [r('i7'), S('air.max_constant_buffers'), r('i31')])
    node('f.tg',    [r('i7'), S('air.max_threadgroup_buffers'), r('i31')])
    node('f.tex',   [r('i7'), S('air.max_textures'), r('i128')])
    node('f.rwtex', [r('i7'), S('air.max_read_write_textures'), r('i8')])
    node('f.samp',  [r('i7'), S('air.max_samplers'), r('i16')])
    node('ident',   [S('Apple metal version 32023.620 (metalfe-32023.620)')])
    node('version', [r('i2'), r('i7'), r('i0')])
    node('lang',    [S('Metal'), r('i3'), r('i2'), r('i0')])
    node('opt.den', [S('air.compile.denorms_disable')])
    node('opt.fm',  [S('air.compile.fast_math_enable')])
    node('opt.fb',  [S('air.compile.framebuffer_fetch_enable')])
    node('empty',   [])
    for i in range(N):                                     # air.buffer descriptor per buffer, all read_write
        node(f'b{i}', [r(f'i{i}'), S('air.buffer'), S('air.location_index'), r(f'i{i}'), r('i1'),
                       S('air.read_write'), S('air.address_space'), r('i1'), S('air.arg_type_size'), r('i4'),
                       S('air.arg_type_align_size'), r('i4'), S('air.arg_type_name'), S('float'),
                       S('air.arg_name'), S(f'b{i}'), S('air.arg_unused')])
    node('tpos',    [r(f'i{N}'), S('air.thread_position_in_grid'), S('air.arg_type_name'), S('uint'),
                     S('air.arg_name'), S('t'), S('air.arg_unused')])
    node('args',    [r(f'b{i}') for i in range(N)] + [r('tpos')])
    node('kernel',  [r('fn'), r('empty'), r('args')])
    recs = values + nodes
    sizes = [rec_bits(4, it) for it in recs]

    module = blk(8, 3,                           # MODULE, abbrev width 3
        U(1, [2]),                               # VERSION 2
        B(blockinfo_block),

        B(blk(17, 4,                             # TYPE table: 0=void 1=float 2=float*(as1) 3=i32 4=fn 5=fn* 6,7 unexplained
            D(A(L(8),  F(4), L(0))),             # 4: POINTER(elem-type, as literal 0)
            D(A(L(25), L(0))),                   # 5: unexplained
            D(A(L(21), F(1), ARR, F(4))),        # 6: FUNCTION(vararg, [ret, params...])
            D(A(L(18), F(1), ARR, F(4))),        # 7: STRUCT_NAMED
            D(A(L(19), ARR, C6)),                # 8: STRUCT_NAME
            D(A(L(20), F(1), ARR, F(4))),        # 9: STRUCT_ANON(?)
            D(A(L(11), V(8), F(4))),             # 10: ARRAY
            U(1, [8]),                           # NUMENTRY = 8 types
            U(2, []),                            #  VOID
            U(3, []),                            #  FLOAT
            U(8, [1, 1]),                        #  POINTER(float, addrspace 1 = device)
            U(7, [32]),                          #  INTEGER 32
            R(6, 21, 0, [0] + [2] * N + [3]),    #  FUNCTION: ret void, params [float* x N, i32]
            R(4, 8, 4, 0),                       #  POINTER(fn-type, as 0)
            U(16, []),                           #  unexplained (METADATA type?)
            R(10, 11, 2, 3),                     #  ARRAY? unexplained
        )),

        # ---- PARAMATTR_GROUP (8,10): group 1 = function attrs (enum attrs + string key=values from `metal`),
        # groups 2..N+1 = per-buffer nocapture + "air-buffer-no-alias" + noundef, group N+2 = t: noundef.
        B(blk(10, 3,
            _grp(1, 0xffffffff, 70, 62, 48, 63, 18, 20, 61,
                 ('approx-func-fp-math', 'true'), ('frame-pointer', 'all'),
                 ('min-legal-vector-width', '0'), ('no-builtins',),
                 ('no-infs-fp-math', 'true'), ('no-nans-fp-math', 'true'),
                 ('no-signed-zeros-fp-math', 'true'),
                 ('no-trapping-math', 'true'),
                 ('stack-protector-buffer-size', '8'),
                 ('unsafe-fp-math', 'true')),
            *[_grp(i + 2, i + 1, NOCAPTURE, NOUNDEF, 'air-buffer-no-alias') for i in range(N)],
            _grp(N + 2, N + 1, NOUNDEF),
        )),
        B(blk(9, 3,                              # PARAMATTR: fn 'k' uses all groups
            U(2, list(range(1, N + 3))),
        )),

        U(2, _s(TRIPLE)),                        # MODULE_CODE_TRIPLE
        U(3, _s(DATALAYOUT)),                    # MODULE_CODE_DATALAYOUT
        D(A(L(16), ARR, C6)),                    # 4: SOURCE_FILENAME abbrev
        R(4, 16, _c6(srcname)),                  # SOURCE_FILENAME; a per-program name keeps the function hash (=archive key) unique
        U(8, [0, 1, 4, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0]),  # FUNCTION 'k': strtab (0,1), type 4, paramattr 1; tail constant
        D(A(L(13), F(32))),                      # 5: VSTOFFSET abbrev
        R(5, 13, 0),                             # VSTOFFSET word offset of block 14; filled by emit_fixed

        B(blk(11, 4,                             # CONSTANTS
            D(A(L(7),  ARR, F(4))),              # 4/8: unused
            D(A(L(8),  ARR, F(8))),              # 9: CST_STRING(?)
            D(A(L(9),  ARR, F(7))),              # 10: CST_CSTRING(?)
            D(A(L(9),  ARR, C6)),                # 11: CST_CSTRING char6
            R(4, 1, 3),                          # SETTYPE i32
            *[U(2, []) if n == 0 else R(5, 4, 2 * n) for n in ints],   # i32 constants (0 = CST_NULL)
            R(4, 1, 7),                          # SETTYPE type 7
            U(22, [15, 5]),                      # SDK-version constant, unexplained encoding
        )),

        B(blk(22, 3, *[U(6, [i] + _s(nm)) for i, nm in enumerate([   # METADATA_KIND: fixed LLVM table
            'dbg', 'tbaa', 'prof', 'fpmath', 'range', 'tbaa.struct',
            'invariant.load', 'alias.scope', 'noalias', 'nontemporal',
            'llvm.mem.parallel_loop_access', 'nonnull', 'dereferenceable',
            'dereferenceable_or_null', 'make.implicit', 'unpredictable',
            'invariant.group', 'align', 'llvm.loop', 'type', 'section_prefix',
            'absolute_symbol', 'associated', 'callees', 'irr_loop',
            'llvm.access.group', 'callback', 'llvm.preserve.access.index',
            'vcall_visibility', 'noundef', 'annotation', 'heapallocsite',
            'air.function_groups'])])),

        B(blk(15, 4,                             # METADATA
            D(A(L(7),  F(1), V(6), V(8), V(6), V(6), F(1))),  # 4: unused
            D(A(L(12), F(1), V(6), F(1), V(6), ARR, V(6))),   # 5: unused
            D(A(L(38), F(32), F(32))),                        # 6: INDEX_OFFSET
            D(A(L(39), ARR, V(6))),                           # 7: INDEX
            D(A(L(35), V(6), V(6), BLOB)),                    # 8: STRINGS
            _md_strings_rec(strings),
            R(6, 38, sum(sizes), 0),             # INDEX_OFFSET: bits from here to the INDEX record
            *recs,
            R(7, 39, [0] + sizes[:-1]),          # INDEX: per-record bit sizes
            D(A(L(4), ARR, F(8))),               # 9: NAME abbrev
            R(9, 4, _s('llvm.module.flags')),
            U(10, [v[k] for k in ('f.sdk', 'f.wchar', 'f.fp', 'f.dev', 'f.cst', 'f.tg', 'f.tex', 'f.rwtex', 'f.samp')]),
            R(9, 4, _s('llvm.ident')),           U(10, [v['ident']]),
            R(9, 4, _s('air.version')),          U(10, [v['version']]),
            R(9, 4, _s('air.language_version')), U(10, [v['lang']]),
            R(9, 4, _s('air.compile_options')),  U(10, [v['opt.den'], v['opt.fm'], v['opt.fb']]),
            R(9, 4, _s('air.kernel')),           U(10, [v['kernel']]),
        )),

        B(blk(21, 3, *[U(1, _s(t)) for t in [    # OPERAND_BUNDLE_TAGS: fixed LLVM table
            'deopt', 'funclet', 'gc-transition', 'cfguardtarget',
            'preallocated', 'gc-live', 'clang.arc.attachedcall', 'ptrauth']])),

        B(blk(26, 2,                             # SYNC_SCOPE_NAMES
            U(1, _s('singlethread')),
            U(1, []),
        )),

        B(blk(12, 4,                             # FUNCTION body: `ret void`
            U(1, [1]),                           # DECLAREBLOCKS 1
            R(10, 10),                           # INST_RET
        )),

        B(blk(14, 4,                             # VALUE_SYMTAB
            D(A(L(3), V(8), V(8))),              # 4: FNENTRY [value, word-offset]
            R(8, 3, 0, 0),                       # fn 'k' word offset; filled by emit_fixed
        )),
    )

    top = [
        blk(13, 5,                               # IDENTIFICATION
            D(A(L(1), ARR, C6)),
            R(4, 1, []),                         # producer ""
            D(A(L(2), V(6))),
            R(5, 2, 0),                          # epoch 0
        ),
        module,
        blk(25, 3, D(A(L(1), BLOB)), R(4, 1, SYMTAB_BLOB)),   # SYMTAB
        blk(23, 3, D(A(L(1), BLOB)), R(4, 1, STRTAB_BLOB)),   # STRTAB
    ]
    binfo = {}; cur = None                       # replay BLOCKINFO: {bid: [Abbrev,...]}
    for it in blockinfo_block.items:
        if it[0] == 'unabbrev' and it[1] == 1: cur = it[2][0]
        elif it[0] == 'def': binfo.setdefault(cur, []).append(it[1])
    return top, binfo

def gen_air(N, srcname='k'):
    """AIR bitcode for `kernel void k(device float* b0..b{N-1}, uint t)` with an empty body.
    `srcname` (SOURCE_FILENAME, char6 alphabet [a-zA-Z0-9._]) is the only per-program input: two programs with the
    same N and srcname hash to the same archive key and Metal's in-process pipeline cache would serve one for the other."""
    top, binfo = base(N, srcname)
    mod = top[1]
    vstoff = next(it for it in mod.items if it[0] == 'abbrevrec' and it[2][0] == 13)
    fnentry = next(it[1] for it in mod.items if it[0] == 'block' and it[1].bid == 14).items[-1]
    for _ in range(10):                          # VSTOFFSET/FNENTRY hold word offsets of blocks 14/12: iterate to a fixpoint
        out, poss = emit(top, binfo)
        w14, w12 = poss[14] // 32, poss[12] // 32
        if vstoff[2][1] == w14 and fnentry[2][2] == w12: return out
        vstoff[2][1] = w14; fnentry[2][2] = w12
    raise RuntimeError('offset fixpoint did not converge')

# ---- metallib (MTLB) container around the bitcode; single function 'k' ----
MODULE_OFF = 231

def gen_mtlb(bitcode):
    """Wrap unpadded bitcode in a byte-exact MTLB with a self-computed SHA-256 HASH."""
    module = struct.pack('<IIIII', 0x0B17C0DE, 0, 0x14, len(bitcode), 0xFFFFFFFF) + bitcode
    if len(module) % 16: module += bytes(16 - (len(module) % 16))
    mdsz = len(module)
    b = bytearray(b'MTLB' + bytes.fromhex('0180020007000081') + struct.pack('<I', 15))
    b += struct.pack('<Q', MODULE_OFF + mdsz)                          # filesize
    for v in (88, 119, 215, 8, 223, 8, MODULE_OFF, mdsz): b += struct.pack('<Q', v)
    b += struct.pack('<II', 1, 119)
    b += b'NAME' + struct.pack('<H', 2) + b'k\x00'
    b += b'TYPE' + struct.pack('<H', 1) + bytes([0x02])
    b += b'HASH' + struct.pack('<H', 32) + hashlib.sha256(module).digest()
    b += b'MDSZ' + struct.pack('<H', 8) + struct.pack('<Q', mdsz)
    b += b'OFFT' + struct.pack('<H', 24) + bytes(24)
    b += b'VERS' + struct.pack('<H', 8) + bytes.fromhex('0200070003000200')
    b += bytes.fromhex('454e4454454e445404000000454e445404000000454e4454')   # ENDT tail
    assert len(b) == MODULE_OFF, len(b)
    return bytes(b) + module
