#include <lsxintrin.h>
#include <stdio.h>

#include "vector-common.h"

typedef union {
   __m128i v;
   vec128_data d;
} vec128;

static void print_case1(const char* name, const vec128* a, const vec128* got)
{
   printf("insn %s:\n", name);
   printf("  v_arg1   = ");
   print_u64x2(a->d.u64);
   printf("  v_result = ");
   print_u64x2(got->d.u64);
}

static void print_case2(const char* name, const vec128* a,
                        const vec128* b, const vec128* got)
{
   printf("insn %s:\n", name);
   printf("  v_arg1   = ");
   print_u64x2(a->d.u64);
   printf("  v_arg2   = ");
   print_u64x2(b->d.u64);
   printf("  v_result = ");
   print_u64x2(got->d.u64);
}

static void print_case2_imm(const char* name, const vec128* a,
                            uint64_t imm, const vec128* got)
{
   printf("insn %s:\n", name);
   printf("  v_arg1   = ");
   print_u64x2(a->d.u64);
   printf("  imm      = %016llx\n", (unsigned long long)imm);
   printf("  v_result = ");
   print_u64x2(got->d.u64);
}

static void print_case3(const char* name, const vec128* a, const vec128* b,
                        const vec128* c, const vec128* got)
{
   printf("insn %s:\n", name);
   printf("  v_arg1   = ");
   print_u64x2(a->d.u64);
   printf("  v_arg2   = ");
   print_u64x2(b->d.u64);
   printf("  v_arg3   = ");
   print_u64x2(c->d.u64);
   printf("  v_result = ");
   print_u64x2(got->d.u64);
}

#define DO_UN128(name, model_stmt, expr) \
   do { \
      model_stmt; \
      got.v = (expr); \
      print_case1((name), &a, &got); \
      check_bytes(tst, (name), got.d.u8, exp.d.u8, sizeof(exp.d.u8)); \
   } while (0)

#define DO_BIN128(name, model_stmt, expr) \
   do { \
      model_stmt; \
      got.v = (expr); \
      print_case2((name), &a, &b, &got); \
      check_bytes(tst, (name), got.d.u8, exp.d.u8, sizeof(exp.d.u8)); \
   } while (0)

#define DO_IMM128(name, immv, model_stmt, expr) \
   do { \
      model_stmt; \
      got.v = (expr); \
      print_case2_imm((name), &a, (immv), &got); \
      check_bytes(tst, (name), got.d.u8, exp.d.u8, sizeof(exp.d.u8)); \
   } while (0)

#define DO_TRI128(name, model_stmt, expr) \
   do { \
      model_stmt; \
      got.v = (expr); \
      print_case3((name), &a, &b, &c, &got); \
      check_bytes(tst, (name), got.d.u8, exp.d.u8, sizeof(exp.d.u8)); \
   } while (0)

#define DO_TRI128_ACC(name, model_stmt, expr) \
   do { \
      model_stmt; \
      got.v = (expr); \
      print_case3((name), &acc, &a, &b, &got); \
      check_bytes(tst, (name), got.d.u8, exp.d.u8, sizeof(exp.d.u8)); \
   } while (0)

static void test_basic(test_state* tst)
{
   vec128 a = {.d.u8 = {
      0x81, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78,
      0x89, 0x9a, 0xab, 0xbc, 0xcd, 0xde, 0xef, 0x10}};
   vec128 b = {.d.u8 = {
      0x11, 0xf2, 0xe3, 0xd4, 0xc5, 0xb6, 0xa7, 0x98,
      0x79, 0x6a, 0x5b, 0x4c, 0x3d, 0x2e, 0x1f, 0xf0}};
   vec128 c = {.d.u8 = {
      3, 5, 7, 9, 11, 13, 15, 17,
      19, 21, 23, 25, 27, 29, 31, 33}};
   vec128 got, exp;

   memset(&exp, 0, sizeof(exp));
   DO_BIN128("vadd.b", model_addsub(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 0), __lsx_vadd_b(a.v, b.v));
   DO_BIN128("vadd.h", model_addsub(exp.d.u8, a.d.u8, b.d.u8, 16, 8, 0), __lsx_vadd_h(a.v, b.v));
   DO_BIN128("vadd.w", model_addsub(exp.d.u8, a.d.u8, b.d.u8, 32, 4, 0), __lsx_vadd_w(a.v, b.v));
   DO_BIN128("vadd.d", model_addsub(exp.d.u8, a.d.u8, b.d.u8, 64, 2, 0), __lsx_vadd_d(a.v, b.v));
   DO_BIN128("vadd.q", model_q_addsub(exp.d.u8, a.d.u8, b.d.u8, 1, 0), __lsx_vadd_q(a.v, b.v));
   DO_BIN128("vsub.b", model_addsub(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 1), __lsx_vsub_b(a.v, b.v));
   DO_BIN128("vsub.h", model_addsub(exp.d.u8, a.d.u8, b.d.u8, 16, 8, 1), __lsx_vsub_h(a.v, b.v));
   DO_BIN128("vsub.w", model_addsub(exp.d.u8, a.d.u8, b.d.u8, 32, 4, 1), __lsx_vsub_w(a.v, b.v));
   DO_BIN128("vsub.d", model_addsub(exp.d.u8, a.d.u8, b.d.u8, 64, 2, 1), __lsx_vsub_d(a.v, b.v));
   DO_BIN128("vsub.q", model_q_addsub(exp.d.u8, a.d.u8, b.d.u8, 1, 1), __lsx_vsub_q(a.v, b.v));
   DO_IMM128("vaddi.bu", 7, model_addi(exp.d.u8, a.d.u8, 8, 16, 7, 0), __lsx_vaddi_bu(a.v, 7));
   DO_IMM128("vaddi.hu", 9, model_addi(exp.d.u8, a.d.u8, 16, 8, 9, 0), __lsx_vaddi_hu(a.v, 9));
   DO_IMM128("vaddi.wu", 11, model_addi(exp.d.u8, a.d.u8, 32, 4, 11, 0), __lsx_vaddi_wu(a.v, 11));
   DO_IMM128("vaddi.du", 13, model_addi(exp.d.u8, a.d.u8, 64, 2, 13, 0), __lsx_vaddi_du(a.v, 13));
   DO_IMM128("vsubi.bu", 5, model_addi(exp.d.u8, a.d.u8, 8, 16, 5, 1), __lsx_vsubi_bu(a.v, 5));
   DO_IMM128("vsubi.hu", 6, model_addi(exp.d.u8, a.d.u8, 16, 8, 6, 1), __lsx_vsubi_hu(a.v, 6));
   DO_IMM128("vsubi.wu", 7, model_addi(exp.d.u8, a.d.u8, 32, 4, 7, 1), __lsx_vsubi_wu(a.v, 7));
   DO_IMM128("vsubi.du", 8, model_addi(exp.d.u8, a.d.u8, 64, 2, 8, 1), __lsx_vsubi_du(a.v, 8));
   DO_UN128("vclo.b", model_clo(exp.d.u8, a.d.u8, 8, 16), __lsx_vclo_b(a.v));
   DO_UN128("vclz.h", model_clz(exp.d.u8, a.d.u8, 16, 8), __lsx_vclz_h(a.v));
   DO_UN128("vpcnt.w", model_pcnt(exp.d.u8, a.d.u8, 32, 4), __lsx_vpcnt_w(a.v));
   DO_UN128("vneg.d", model_neg(exp.d.u8, a.d.u8, 64, 2), __lsx_vneg_d(a.v));
   DO_BIN128("vsadd.b", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 1, 0), __lsx_vsadd_b(a.v, b.v));
   DO_BIN128("vsadd.hu", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 16, 8, 0, 0), __lsx_vsadd_hu(a.v, b.v));
   DO_BIN128("vssub.w", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 32, 4, 1, 1), __lsx_vssub_w(a.v, b.v));
   DO_BIN128("vssub.du", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 64, 2, 0, 1), __lsx_vssub_du(a.v, b.v));
   DO_BIN128("vavgr.b", model_avgr(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 1), __lsx_vavgr_b(a.v, b.v));
   DO_BIN128("vavgr.hu", model_avgr(exp.d.u8, a.d.u8, b.d.u8, 16, 8, 0), __lsx_vavgr_hu(a.v, b.v));
   DO_BIN128("vadda.w", model_adda(exp.d.u8, a.d.u8, b.d.u8, 32, 4), __lsx_vadda_w(a.v, b.v));
   DO_BIN128("vabsd.h", model_absd(exp.d.u8, a.d.u8, b.d.u8, 16, 8, 1), __lsx_vabsd_h(a.v, b.v));
   DO_BIN128("vabsd.du", model_absd(exp.d.u8, a.d.u8, b.d.u8, 64, 2, 0), __lsx_vabsd_du(a.v, b.v));
   DO_BIN128("vmax.b", model_minmax(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 1, 0), __lsx_vmax_b(a.v, b.v));
   DO_BIN128("vmin.hu", model_minmax(exp.d.u8, a.d.u8, b.d.u8, 16, 8, 0, 1), __lsx_vmin_hu(a.v, b.v));
   DO_IMM128("vmaxi.w", 7, model_minmaxi(exp.d.u8, a.d.u8, 32, 4, 7, 1, 0), __lsx_vmaxi_w(a.v, 7));
   DO_IMM128("vmini.du", 19, model_minmaxi(exp.d.u8, a.d.u8, 64, 2, 19, 0, 1), __lsx_vmini_du(a.v, 19));
   DO_BIN128("vmul.h", model_mullo(exp.d.u8, a.d.u8, c.d.u8, 16, 8), __lsx_vmul_h(a.v, c.v));
   DO_BIN128("vmuh.w", model_mulhi(exp.d.u8, a.d.u8, c.d.u8, 32, 4, 1), __lsx_vmuh_w(a.v, c.v));
   DO_BIN128("vmuh.hu", model_mulhi(exp.d.u8, a.d.u8, c.d.u8, 16, 8, 0), __lsx_vmuh_hu(a.v, c.v));
   DO_TRI128("vmadd.b", model_maddmsub(exp.d.u8, a.d.u8, b.d.u8, c.d.u8, 8, 16, 0), __lsx_vmadd_b(a.v, b.v, c.v));
   DO_TRI128("vmsub.w", model_maddmsub(exp.d.u8, a.d.u8, b.d.u8, c.d.u8, 32, 4, 1), __lsx_vmsub_w(a.v, b.v, c.v));
   DO_BIN128("vdiv.b", model_divmod(exp.d.u8, a.d.u8, c.d.u8, 8, 16, 1, 0), __lsx_vdiv_b(a.v, c.v));
   DO_BIN128("vmod.hu", model_divmod(exp.d.u8, a.d.u8, c.d.u8, 16, 8, 0, 1), __lsx_vmod_hu(a.v, c.v));
   DO_BIN128("vsigncov.d", model_signcov(exp.d.u8, a.d.u8, b.d.u8, 64, 2), __lsx_vsigncov_d(a.v, b.v));
}

static void test_widen(test_state* tst)
{
   vec128 a = {.d.u8 = {
      1, 0xfe, 3, 0xfc, 5, 0xfa, 7, 0xf8,
      9, 0xf6, 11, 0xf4, 13, 0xf2, 15, 0xf0}};
   vec128 b = {.d.u8 = {
      2, 3, 4, 5, 6, 7, 8, 9,
      10, 11, 12, 13, 14, 15, 16, 17}};
   vec128 acc = {.d.u16 = {100, 200, 300, 400, 500, 600, 700, 800}};
   vec128 got, exp;

   DO_BIN128("vmulwev.h.b", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 0, 1, 1), __lsx_vmulwev_h_b(a.v, b.v));
   DO_BIN128("vmulwod.h.bu.b", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 1, 0, 1), __lsx_vmulwod_h_bu_b(a.v, b.v));
   DO_TRI128_ACC("vmaddwev.h.b", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 0, 1, 1), __lsx_vmaddwev_h_b(acc.v, a.v, b.v));
   DO_TRI128_ACC("vmaddwod.h.bu.b", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 1, 0, 1), __lsx_vmaddwod_h_bu_b(acc.v, a.v, b.v));
}

int main(void)
{
   test_state tst = {0, 0};

   test_basic(&tst);
   test_widen(&tst);

   return tst.fails != 0;
}
