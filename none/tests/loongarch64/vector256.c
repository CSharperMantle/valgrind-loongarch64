#include <lasxintrin.h>
#include <stdio.h>

#include "vector-common.h"

typedef union {
   __m256i v;
   vec256_data d;
} vec256;

static void print_case1(const char* name, const vec256* a, const vec256* got)
{
   printf("insn %s:\n", name);
   printf("  v_arg1   = ");
   print_u64x4(a->d.u64);
   printf("  v_result = ");
   print_u64x4(got->d.u64);
}

static void print_case2(const char* name, const vec256* a,
                        const vec256* b, const vec256* got)
{
   printf("insn %s:\n", name);
   printf("  v_arg1   = ");
   print_u64x4(a->d.u64);
   printf("  v_arg2   = ");
   print_u64x4(b->d.u64);
   printf("  v_result = ");
   print_u64x4(got->d.u64);
}

static void print_case2_imm(const char* name, const vec256* a,
                            uint64_t imm, const vec256* got)
{
   printf("insn %s:\n", name);
   printf("  v_arg1   = ");
   print_u64x4(a->d.u64);
   printf("  imm      = %016llx\n", (unsigned long long)imm);
   printf("  v_result = ");
   print_u64x4(got->d.u64);
}

static void print_case3(const char* name, const vec256* a, const vec256* b,
                        const vec256* c, const vec256* got)
{
   printf("insn %s:\n", name);
   printf("  v_arg1   = ");
   print_u64x4(a->d.u64);
   printf("  v_arg2   = ");
   print_u64x4(b->d.u64);
   printf("  v_arg3   = ");
   print_u64x4(c->d.u64);
   printf("  v_result = ");
   print_u64x4(got->d.u64);
}

#define DO_UN256(name, model_stmt, expr) \
   do { \
      model_stmt; \
      got.v = (expr); \
      print_case1((name), &a, &got); \
      check_bytes(tst, (name), got.d.u8, exp.d.u8, sizeof(exp.d.u8)); \
   } while (0)

#define DO_BIN256(name, model_stmt, expr) \
   do { \
      model_stmt; \
      got.v = (expr); \
      print_case2((name), &a, &b, &got); \
      check_bytes(tst, (name), got.d.u8, exp.d.u8, sizeof(exp.d.u8)); \
   } while (0)

#define DO_IMM256(name, immv, model_stmt, expr) \
   do { \
      model_stmt; \
      got.v = (expr); \
      print_case2_imm((name), &a, (immv), &got); \
      check_bytes(tst, (name), got.d.u8, exp.d.u8, sizeof(exp.d.u8)); \
   } while (0)

#define DO_TRI256(name, model_stmt, expr) \
   do { \
      model_stmt; \
      got.v = (expr); \
      print_case3((name), &a, &b, &c, &got); \
      check_bytes(tst, (name), got.d.u8, exp.d.u8, sizeof(exp.d.u8)); \
   } while (0)

#define DO_TRI256_ACC(name, model_stmt, expr) \
   do { \
      model_stmt; \
      got.v = (expr); \
      print_case3((name), &acc, &a, &b, &got); \
      check_bytes(tst, (name), got.d.u8, exp.d.u8, sizeof(exp.d.u8)); \
   } while (0)

static void test_basic(test_state* tst)
{
   vec256 a = {.d.u8 = {
      0x81, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78,
      0x89, 0x9a, 0xab, 0xbc, 0xcd, 0xde, 0xef, 0x10,
      0x20, 0x31, 0x42, 0x53, 0x64, 0x75, 0x86, 0x97,
      0xa8, 0xb9, 0xca, 0xdb, 0xec, 0xfd, 0x0e, 0x1f}};
   vec256 b = {.d.u8 = {
      0x11, 0xf2, 0xe3, 0xd4, 0xc5, 0xb6, 0xa7, 0x98,
      0x79, 0x6a, 0x5b, 0x4c, 0x3d, 0x2e, 0x1f, 0xf0,
      0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78,
      0x69, 0x5a, 0x4b, 0x3c, 0x2d, 0x1e, 0x0f, 0xff}};
   vec256 c = {.d.u8 = {
      3, 5, 7, 9, 11, 13, 15, 17,
      19, 21, 23, 25, 27, 29, 31, 33,
      35, 37, 39, 41, 43, 45, 47, 49,
      51, 53, 55, 57, 59, 61, 63, 65}};
   vec256 got, exp;

   DO_BIN256("xvadd.b", model_addsub(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 0), __lasx_xvadd_b(a.v, b.v));
   DO_BIN256("xvadd.h", model_addsub(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 0), __lasx_xvadd_h(a.v, b.v));
   DO_BIN256("xvsub.d", model_addsub(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 1), __lasx_xvsub_d(a.v, b.v));
   DO_BIN256("xvadd.q", model_q_addsub(exp.d.u8, a.d.u8, b.d.u8, 2, 0), __lasx_xvadd_q(a.v, b.v));
   DO_IMM256("xvaddi.wu", 11, model_addi(exp.d.u8, a.d.u8, 32, 8, 11, 0), __lasx_xvaddi_wu(a.v, 11));
   DO_IMM256("xvsubi.du", 8, model_addi(exp.d.u8, a.d.u8, 64, 4, 8, 1), __lasx_xvsubi_du(a.v, 8));
   DO_UN256("xvclo.b", model_clo(exp.d.u8, a.d.u8, 8, 32), __lasx_xvclo_b(a.v));
   DO_UN256("xvclz.w", model_clz(exp.d.u8, a.d.u8, 32, 8), __lasx_xvclz_w(a.v));
   DO_UN256("xvpcnt.d", model_pcnt(exp.d.u8, a.d.u8, 64, 4), __lasx_xvpcnt_d(a.v));
   DO_UN256("xvneg.h", model_neg(exp.d.u8, a.d.u8, 16, 16), __lasx_xvneg_h(a.v));
   DO_BIN256("xvsadd.b", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 1, 0), __lasx_xvsadd_b(a.v, b.v));
   DO_BIN256("xvssub.du", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 0, 1), __lasx_xvssub_du(a.v, b.v));
   DO_BIN256("xvavgr.h", model_avgr(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 1), __lasx_xvavgr_h(a.v, b.v));
   DO_BIN256("xvavgr.wu", model_avgr(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 0), __lasx_xvavgr_wu(a.v, b.v));
   DO_BIN256("xvadda.d", model_adda(exp.d.u8, a.d.u8, b.d.u8, 64, 4), __lasx_xvadda_d(a.v, b.v));
   DO_BIN256("xvabsd.b", model_absd(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 1), __lasx_xvabsd_b(a.v, b.v));
   DO_BIN256("xvabsd.du", model_absd(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 0), __lasx_xvabsd_du(a.v, b.v));
   DO_BIN256("xvmax.h", model_minmax(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 1, 0), __lasx_xvmax_h(a.v, b.v));
   DO_BIN256("xvmin.wu", model_minmax(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 0, 1), __lasx_xvmin_wu(a.v, b.v));
   DO_IMM256("xvmaxi.b", 9, model_minmaxi(exp.d.u8, a.d.u8, 8, 32, 9, 1, 0), __lasx_xvmaxi_b(a.v, 9));
   DO_IMM256("xvmini.du", 17, model_minmaxi(exp.d.u8, a.d.u8, 64, 4, 17, 0, 1), __lasx_xvmini_du(a.v, 17));
   DO_BIN256("xvmul.w", model_mullo(exp.d.u8, a.d.u8, c.d.u8, 32, 8), __lasx_xvmul_w(a.v, c.v));
   DO_BIN256("xvmuh.h", model_mulhi(exp.d.u8, a.d.u8, c.d.u8, 16, 16, 1), __lasx_xvmuh_h(a.v, c.v));
   DO_BIN256("xvmuh.du", model_mulhi(exp.d.u8, a.d.u8, c.d.u8, 64, 4, 0), __lasx_xvmuh_du(a.v, c.v));
   DO_TRI256("xvmadd.h", model_maddmsub(exp.d.u8, a.d.u8, b.d.u8, c.d.u8, 16, 16, 0), __lasx_xvmadd_h(a.v, b.v, c.v));
   DO_TRI256("xvmsub.d", model_maddmsub(exp.d.u8, a.d.u8, b.d.u8, c.d.u8, 64, 4, 1), __lasx_xvmsub_d(a.v, b.v, c.v));
   DO_BIN256("xvdiv.h", model_divmod(exp.d.u8, a.d.u8, c.d.u8, 16, 16, 1, 0), __lasx_xvdiv_h(a.v, c.v));
   DO_BIN256("xvmod.wu", model_divmod(exp.d.u8, a.d.u8, c.d.u8, 32, 8, 0, 1), __lasx_xvmod_wu(a.v, c.v));
   DO_BIN256("xvsigncov.w", model_signcov(exp.d.u8, a.d.u8, b.d.u8, 32, 8), __lasx_xvsigncov_w(a.v, b.v));
}

static void test_widen(test_state* tst)
{
   vec256 a = {.d.u8 = {
      1, 0xfe, 3, 0xfc, 5, 0xfa, 7, 0xf8,
      9, 0xf6, 11, 0xf4, 13, 0xf2, 15, 0xf0,
      17, 0xee, 19, 0xec, 21, 0xea, 23, 0xe8,
      25, 0xe6, 27, 0xe4, 29, 0xe2, 31, 0xe0}};
   vec256 b = {.d.u8 = {
      2, 3, 4, 5, 6, 7, 8, 9,
      10, 11, 12, 13, 14, 15, 16, 17,
      18, 19, 20, 21, 22, 23, 24, 25,
      26, 27, 28, 29, 30, 31, 32, 33}};
   vec256 acc = {.d.u16 = {
      100, 200, 300, 400, 500, 600, 700, 800,
      900, 1000, 1100, 1200, 1300, 1400, 1500, 1600}};
   vec256 got, exp;

   DO_BIN256("xvmulwev.h.b", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 0, 1, 1), __lasx_xvmulwev_h_b(a.v, b.v));
   DO_BIN256("xvmulwod.h.bu.b", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 1, 0, 1), __lasx_xvmulwod_h_bu_b(a.v, b.v));
   DO_TRI256_ACC("xvmaddwev.h.b", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 0, 1, 1), __lasx_xvmaddwev_h_b(acc.v, a.v, b.v));
   DO_TRI256_ACC("xvmaddwod.h.bu.b", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 1, 0, 1), __lasx_xvmaddwod_h_bu_b(acc.v, a.v, b.v));
}

int main(void)
{
   test_state tst = {0, 0};

   test_basic(&tst);
   test_widen(&tst);

   return tst.fails != 0;
}
