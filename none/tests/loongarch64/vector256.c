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

static void print_case2v_imm256(const char* name, const vec256* a,
                                const vec256* b, uint64_t imm,
                                const vec256* got)
{
   printf("insn %s:\n", name);
   printf("  v_arg1   = ");
   print_u64x4(a->d.u64);
   printf("  v_arg2   = ");
   print_u64x4(b->d.u64);
   printf("  imm      = %016llx\n", (unsigned long long)imm);
   printf("  v_result = ");
   print_u64x4(got->d.u64);
}

static void print_case_vs_ri256(const char* name, const vec256* a, uint64_t r,
                                uint64_t imm)
{
   printf("insn %s:\n", name);
   printf("  v_arg1   = ");
   print_u64x4(a->d.u64);
   printf("  r_arg2   = %016llx\n", (unsigned long long)r);
   printf("  imm      = %016llx\n", (unsigned long long)imm);
}

static void print_case_pick256(const char* name, const vec256* a, uint64_t imm,
                               uint64_t got)
{
   printf("insn %s:\n", name);
   printf("  v_arg1   = ");
   print_u64x4(a->d.u64);
   printf("  imm      = %016llx\n", (unsigned long long)imm);
   printf("  r_result = %016llx\n", (unsigned long long)got);
}

static void print_case_rv256(const char* name, uint64_t r, const vec256* got)
{
   printf("insn %s:\n", name);
   printf("  r_arg1   = %016llx\n", (unsigned long long)r);
   printf("  v_result = ");
   print_u64x4(got->d.u64);
}

static void print_case_set256(const char* name, const vec256* a, uint64_t got)
{
   printf("insn %s:\n", name);
   printf("  v_arg1   = ");
   print_u64x4(a->d.u64);
   printf("  r_result = %016llx\n", (unsigned long long)got);
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

#define DO_INS256(name, immv, rval, model_stmt, expr) \
   do { \
      model_stmt; \
      got.v = (expr); \
      print_case_vs_ri256((name), &a, (rval), (immv)); \
      printf("  v_result = "); \
      print_u64x4(got.d.u64); \
      check_bytes(tst, (name), got.d.u8, exp.d.u8, sizeof(exp.d.u8)); \
   } while (0)

#define DO_PICK256(name, immv, model_expr, expr) \
   do { \
      uint64_t got_r = (uint64_t)(expr); \
      uint64_t exp_r = (model_expr); \
      print_case_pick256((name), &a, (immv), got_r); \
      check_u64(tst, (name), got_r, exp_r); \
   } while (0)

#define DO_RV256(name, rval, model_stmt, expr) \
   do { \
      model_stmt; \
      got.v = (expr); \
      print_case_rv256((name), (rval), &got); \
      check_bytes(tst, (name), got.d.u8, exp.d.u8, sizeof(exp.d.u8)); \
   } while (0)

#define DO_SET256(name, vecv, model_expr, expr) \
   do { \
      uint64_t got_r = (uint64_t)(expr); \
      uint64_t exp_r = (model_expr); \
      print_case_set256((name), &(vecv), got_r); \
      check_u64(tst, (name), got_r, exp_r); \
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
   DO_BIN256("xvavg.b", model_avg(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 1), __lasx_xvavg_b(a.v, b.v));
   DO_BIN256("xvavg.hu", model_avg(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 0), __lasx_xvavg_hu(a.v, b.v));
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
   DO_BIN256("xvadda.b", model_adda(exp.d.u8, a.d.u8, b.d.u8, 8, 32), __lasx_xvadda_b(a.v, b.v));
   DO_BIN256("xvadda.h", model_adda(exp.d.u8, a.d.u8, b.d.u8, 16, 16), __lasx_xvadda_h(a.v, b.v));
   DO_BIN256("xvadda.w", model_adda(exp.d.u8, a.d.u8, b.d.u8, 32, 8), __lasx_xvadda_w(a.v, b.v));
   DO_BIN256("xvabsd.h", model_absd(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 1), __lasx_xvabsd_h(a.v, b.v));
   DO_BIN256("xvabsd.w", model_absd(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 1), __lasx_xvabsd_w(a.v, b.v));
   DO_BIN256("xvabsd.hu", model_absd(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 0), __lasx_xvabsd_hu(a.v, b.v));
   DO_BIN256("xvabsd.wu", model_absd(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 0), __lasx_xvabsd_wu(a.v, b.v));
   DO_BIN256("xvavg.h", model_avg(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 1), __lasx_xvavg_h(a.v, b.v));
   DO_BIN256("xvavg.w", model_avg(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 1), __lasx_xvavg_w(a.v, b.v));
   DO_BIN256("xvavg.d", model_avg(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 1), __lasx_xvavg_d(a.v, b.v));
   DO_BIN256("xvavg.bu", model_avg(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 0), __lasx_xvavg_bu(a.v, b.v));
   DO_BIN256("xvavg.wu", model_avg(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 0), __lasx_xvavg_wu(a.v, b.v));
   DO_BIN256("xvavg.du", model_avg(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 0), __lasx_xvavg_du(a.v, b.v));
   DO_BIN256("xvavgr.b", model_avgr(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 1), __lasx_xvavgr_b(a.v, b.v));
   DO_BIN256("xvavgr.w", model_avgr(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 1), __lasx_xvavgr_w(a.v, b.v));
   DO_BIN256("xvavgr.d", model_avgr(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 1), __lasx_xvavgr_d(a.v, b.v));
   DO_BIN256("xvavgr.bu", model_avgr(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 0), __lasx_xvavgr_bu(a.v, b.v));
   DO_BIN256("xvavgr.du", model_avgr(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 0), __lasx_xvavgr_du(a.v, b.v));
   DO_BIN256("xvmax.b", model_minmax(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 1, 0), __lasx_xvmax_b(a.v, b.v));
   DO_BIN256("xvmax.w", model_minmax(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 1, 0), __lasx_xvmax_w(a.v, b.v));
   DO_BIN256("xvmax.d", model_minmax(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 1, 0), __lasx_xvmax_d(a.v, b.v));
   DO_BIN256("xvmin.b", model_minmax(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 1, 1), __lasx_xvmin_b(a.v, b.v));
   DO_BIN256("xvmin.w", model_minmax(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 1, 1), __lasx_xvmin_w(a.v, b.v));
   DO_BIN256("xvmin.d", model_minmax(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 1, 1), __lasx_xvmin_d(a.v, b.v));
   DO_IMM256("xvmaxi.h", -5, model_minmaxi(exp.d.u8, a.d.u8, 16, 16, -5, 1, 0), __lasx_xvmaxi_h(a.v, -5));
   DO_IMM256("xvmaxi.w", 7, model_minmaxi(exp.d.u8, a.d.u8, 32, 8, 7, 1, 0), __lasx_xvmaxi_w(a.v, 7));
   DO_IMM256("xvmaxi.d", 11, model_minmaxi(exp.d.u8, a.d.u8, 64, 4, 11, 1, 0), __lasx_xvmaxi_d(a.v, 11));
   DO_IMM256("xvmini.b", -3, model_minmaxi(exp.d.u8, a.d.u8, 8, 32, -3, 1, 1), __lasx_xvmini_b(a.v, -3));
   DO_IMM256("xvmini.h", 5, model_minmaxi(exp.d.u8, a.d.u8, 16, 16, 5, 1, 1), __lasx_xvmini_h(a.v, 5));
   DO_IMM256("xvmini.w", -7, model_minmaxi(exp.d.u8, a.d.u8, 32, 8, -7, 1, 1), __lasx_xvmini_w(a.v, -7));
   DO_IMM256("xvmaxi.bu", 9, model_minmaxi(exp.d.u8, a.d.u8, 8, 32, 9, 0, 0), __lasx_xvmaxi_bu(a.v, 9));
   DO_IMM256("xvmaxi.du", 7, model_minmaxi(exp.d.u8, a.d.u8, 64, 4, 7, 0, 0), __lasx_xvmaxi_du(a.v, 7));
   DO_IMM256("xvmini.bu", 23, model_minmaxi(exp.d.u8, a.d.u8, 8, 32, 23, 0, 1), __lasx_xvmini_bu(a.v, 23));
   DO_IMM256("xvmini.hu", 13, model_minmaxi(exp.d.u8, a.d.u8, 16, 16, 13, 0, 1), __lasx_xvmini_hu(a.v, 13));
   DO_BIN256("xvmul.b", model_mullo(exp.d.u8, a.d.u8, c.d.u8, 8, 32), __lasx_xvmul_b(a.v, c.v));
   DO_BIN256("xvmul.h", model_mullo(exp.d.u8, a.d.u8, c.d.u8, 16, 16), __lasx_xvmul_h(a.v, c.v));
   DO_BIN256("xvmul.d", model_mullo(exp.d.u8, a.d.u8, c.d.u8, 64, 4), __lasx_xvmul_d(a.v, c.v));
   DO_BIN256("xvmuh.b", model_mulhi(exp.d.u8, a.d.u8, c.d.u8, 8, 32, 1), __lasx_xvmuh_b(a.v, c.v));
   DO_BIN256("xvmuh.w", model_mulhi(exp.d.u8, a.d.u8, c.d.u8, 32, 8, 1), __lasx_xvmuh_w(a.v, c.v));
   DO_TRI256("xvmadd.b", model_maddmsub(exp.d.u8, a.d.u8, b.d.u8, c.d.u8, 8, 32, 0), __lasx_xvmadd_b(a.v, b.v, c.v));
   DO_TRI256("xvmadd.w", model_maddmsub(exp.d.u8, a.d.u8, b.d.u8, c.d.u8, 32, 8, 0), __lasx_xvmadd_w(a.v, b.v, c.v));
   DO_TRI256("xvmadd.d", model_maddmsub(exp.d.u8, a.d.u8, b.d.u8, c.d.u8, 64, 4, 0), __lasx_xvmadd_d(a.v, b.v, c.v));
   DO_TRI256("xvmsub.b", model_maddmsub(exp.d.u8, a.d.u8, b.d.u8, c.d.u8, 8, 32, 1), __lasx_xvmsub_b(a.v, b.v, c.v));
   DO_TRI256("xvmsub.h", model_maddmsub(exp.d.u8, a.d.u8, b.d.u8, c.d.u8, 16, 16, 1), __lasx_xvmsub_h(a.v, b.v, c.v));
   DO_TRI256("xvmsub.w", model_maddmsub(exp.d.u8, a.d.u8, b.d.u8, c.d.u8, 32, 8, 1), __lasx_xvmsub_w(a.v, b.v, c.v));
   DO_BIN256("xvsigncov.b", model_signcov(exp.d.u8, a.d.u8, b.d.u8, 8, 32), __lasx_xvsigncov_b(a.v, b.v));
   DO_BIN256("xvsigncov.d", model_signcov(exp.d.u8, a.d.u8, b.d.u8, 64, 4), __lasx_xvsigncov_d(a.v, b.v));
}

static void test_sat(test_state* tst)
{
   vec256 a, b, got, exp;

   a = (vec256){.d.i8 = {
      -120, -20, -5, -4, -3, -1, 0, 1, 2, 3, 4, 5, 20, 100, 110, 120,
      -128, -64, -7, -6, -5, -4, -2, -1, 6, 7, 8, 9, 40, 70, 90, 127}};
   b = (vec256){.d.i8 = {
      -120, -120, -120, -2, -1, 1, 1, 2, 120, 120, 120, 4, 20, 40, 60, 80,
      -1, -80, -7, -3, -2, 2, 3, 4, 7, 8, 9, 10, 50, 60, 70, 1}};
   DO_BIN256("xvsadd.b", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 1, 0), __lasx_xvsadd_b(a.v, b.v));
   DO_BIN256("xvssub.b", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 1, 1), __lasx_xvssub_b(a.v, b.v));
   DO_IMM256("xvsat.b", 2, model_sat_imm(exp.d.u8, a.d.u8, 8, 32, 2, 1), __lasx_xvsat_b(a.v, 2));

   a = (vec256){.d.u8 = {
      0, 1, 2, 3, 4, 5, 6, 7, 15, 31, 63, 95, 127, 191, 223, 255,
      8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 48, 64, 96, 128, 192, 250}};
   b = (vec256){.d.u8 = {
      0, 1, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8,
      16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1}};
   DO_BIN256("xvsadd.bu", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 0, 0), __lasx_xvsadd_bu(a.v, b.v));
   DO_BIN256("xvssub.bu", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 0, 1), __lasx_xvssub_bu(a.v, b.v));
   DO_IMM256("xvsat.bu", 2, model_sat_imm(exp.d.u8, a.d.u8, 8, 32, 2, 0), __lasx_xvsat_bu(a.v, 2));

   a = (vec256){.d.i16 = {-2000, -40, -33, -32, -31, 31, 32, 2000, -32768, -200, -33, -1, 0, 1, 200, 32767}};
   b = (vec256){.d.i16 = {-2000, -70, -1, 1, 31, 32, 2000, 2000, -1, -400, 33, 1, 1, 200, 1000, 1}};
   DO_BIN256("xvsadd.h", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 1, 0), __lasx_xvsadd_h(a.v, b.v));
   DO_BIN256("xvssub.h", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 1, 1), __lasx_xvssub_h(a.v, b.v));
   DO_IMM256("xvsat.h", 5, model_sat_imm(exp.d.u8, a.d.u8, 16, 16, 5, 1), __lasx_xvsat_h(a.v, 5));

   a = (vec256){.d.u16 = {0, 1, 31, 32, 33, 255, 511, 60000, 2, 63, 64, 65, 1023, 4095, 32768, 65535}};
   b = (vec256){.d.u16 = {0, 1, 32, 33, 34, 1000, 1000, 60000, 4, 8, 16, 32, 64, 128, 256, 512}};
   DO_BIN256("xvsadd.hu", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 0, 0), __lasx_xvsadd_hu(a.v, b.v));
   DO_BIN256("xvssub.hu", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 0, 1), __lasx_xvssub_hu(a.v, b.v));
   DO_IMM256("xvsat.hu", 5, model_sat_imm(exp.d.u8, a.d.u8, 16, 16, 5, 0), __lasx_xvsat_hu(a.v, 5));

   a = (vec256){.d.i32 = {-200000, -600, -513, -512, -511, 511, 512, 200000}};
   b = (vec256){.d.i32 = {-200000, -600, 1024, 200000, -1, 1, 600, 200000}};
   DO_BIN256("xvsadd.w", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 1, 0), __lasx_xvsadd_w(a.v, b.v));
   DO_BIN256("xvssub.w", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 1, 1), __lasx_xvssub_w(a.v, b.v));
   DO_IMM256("xvsat.w", 9, model_sat_imm(exp.d.u8, a.d.u8, 32, 8, 9, 1), __lasx_xvsat_w(a.v, 9));

   a = (vec256){.d.u32 = {0, 1, 1023, 1024, 1025, 65535, 262143, 4000000000u}};
   b = (vec256){.d.u32 = {0, 1024, 2048, 4096, 8192, 16384, 32768, 4000000000u}};
   DO_BIN256("xvsadd.wu", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 0, 0), __lasx_xvsadd_wu(a.v, b.v));
   DO_BIN256("xvssub.wu", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 0, 1), __lasx_xvssub_wu(a.v, b.v));
   DO_IMM256("xvsat.wu", 9, model_sat_imm(exp.d.u8, a.d.u8, 32, 8, 9, 0), __lasx_xvsat_wu(a.v, 9));

   a = (vec256){.d.i64 = {-500000000000LL, -131073, -131072, 500000000000LL}};
   b = (vec256){.d.i64 = {-500000000000LL, -200000, 200000, 500000000000LL}};
   DO_BIN256("xvsadd.d", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 1, 0), __lasx_xvsadd_d(a.v, b.v));
   DO_BIN256("xvssub.d", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 1, 1), __lasx_xvssub_d(a.v, b.v));
   DO_IMM256("xvsat.d", 17, model_sat_imm(exp.d.u8, a.d.u8, 64, 4, 17, 1), __lasx_xvsat_d(a.v, 17));

   a = (vec256){.d.u64 = {0, 1, 262143, UINT64_MAX}};
   b = (vec256){.d.u64 = {1, 2, 262144, UINT64_MAX}};
   DO_BIN256("xvsadd.du", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 0, 0), __lasx_xvsadd_du(a.v, b.v));
   DO_BIN256("xvssub.du", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 0, 1), __lasx_xvssub_du(a.v, b.v));
   DO_IMM256("xvsat.du", 17, model_sat_imm(exp.d.u8, a.d.u8, 64, 4, 17, 0), __lasx_xvsat_du(a.v, 17));
}

static void test_cmpmask(test_state* tst)
{
   vec256 a = {.d.i16 = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
   vec256 b = {.d.i16 = {-5, -3, -4, -2, 0, 0, 2, 1, 4, 4, 4, 7, 6, 9, 9, 9}};
   vec256 aw = {.d.i32 = {-100, -50, 0, 50, 100, -200, -300, 400}};
   vec256 ad = {.d.i64 = {-1000, 1000, -2000, 2000}};
   vec256 got, exp;

   DO_BIN256("xvseq.h", model_cmp(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 1, 0), __lasx_xvseq_h(a.v, b.v));
   DO_BIN256("xvseq.b", model_cmp(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 1, 0), __lasx_xvseq_b(a.v, b.v));
   DO_BIN256("xvseq.w", model_cmp(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 1, 0), __lasx_xvseq_w(a.v, b.v));
   DO_BIN256("xvseq.d", model_cmp(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 1, 0), __lasx_xvseq_d(a.v, b.v));
   DO_IMM256("xvseqi.w", -1, model_cmp(exp.d.u8, a.d.u8, (vec256){.d.i32 = {-1, -1, -1, -1, -1, -1, -1, -1}}.d.u8, 32, 8, 1, 0), __lasx_xvseqi_w(a.v, -1));
   DO_IMM256("xvseqi.b", 1, model_cmp(exp.d.u8, a.d.u8, (vec256){.d.i8 = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}}.d.u8, 8, 32, 1, 0), __lasx_xvseqi_b(a.v, 1));
   a = (vec256){.d.i32 = {-100, -50, 0, 50, 100, -200, -300, 400}};
   DO_IMM256("xvseqi.h", -5, model_cmp(exp.d.u8, a.d.u8, (vec256){.d.i16 = {-5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5}}.d.u8, 16, 16, 1, 0), __lasx_xvseqi_h(a.v, -5));
   a = (vec256){.d.i64 = {-1000, 1000, -2000, 2000}};
   DO_IMM256("xvseqi.d", 7, model_cmp(exp.d.u8, a.d.u8, (vec256){.d.i64 = {7, 7, 7, 7}}.d.u8, 64, 4, 1, 0), __lasx_xvseqi_d(a.v, 7));
   a = (vec256){.d.i16 = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
   DO_IMM256("xvslei.w", 2, model_cmp(exp.d.u8, a.d.u8, (vec256){.d.i32 = {2, 2, 2, 2, 2, 2, 2, 2}}.d.u8, 32, 8, 1, 1), __lasx_xvslei_w(a.v, 2));
   DO_IMM256("xvslei.b", 3, model_cmp(exp.d.u8, a.d.u8, (vec256){.d.i8 = {3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3}}.d.u8, 8, 32, 1, 1), __lasx_xvslei_b(a.v, 3));
   a = (vec256){.d.i32 = {-100, -50, 0, 50, 100, -200, -300, 400}};
   DO_IMM256("xvslei.h", -3, model_cmp(exp.d.u8, a.d.u8, (vec256){.d.i16 = {-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3}}.d.u8, 16, 16, 1, 1), __lasx_xvslei_h(a.v, -3));
   a = (vec256){.d.i64 = {-1000, 1000, -2000, 2000}};
   DO_IMM256("xvslei.d", -5, model_cmp(exp.d.u8, a.d.u8, (vec256){.d.i64 = {-5, -5, -5, -5}}.d.u8, 64, 4, 1, 1), __lasx_xvslei_d(a.v, -5));
   a = (vec256){.d.u16 = {0, 1, 100, 500, 1000, 5000, 60000, 65000,
                          200, 300, 400, 800, 2000, 7000, 50000, 60000}};
   DO_IMM256("xvsleiu.b", 3, model_cmp(exp.d.u8, a.d.u8, (vec256){.d.u8 = {3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3}}.d.u8, 8, 32, 0, 1), __lasx_xvslei_bu(a.v, 3));
   a = (vec256){.d.u32 = {0, 1, 1000, 4000000000u, 500, 600, 700, 800}};
   DO_IMM256("xvsleiu.h", 7, model_cmp(exp.d.u8, a.d.u8, (vec256){.d.u16 = {7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7}}.d.u8, 16, 16, 0, 1), __lasx_xvslei_hu(a.v, 7));
   a = (vec256){.d.u64 = {0, 100000, 200000, 300000}};
   DO_IMM256("xvsleiu.w", 31, model_cmp(exp.d.u8, a.d.u8, (vec256){.d.u32 = {31, 31, 31, 31, 31, 31, 31, 31}}.d.u8, 32, 8, 0, 1), __lasx_xvslei_wu(a.v, 31));
   a = (vec256){.d.u16 = {0, 1, 100, 500, 1000, 5000, 60000, 65000,
                          200, 300, 400, 800, 2000, 7000, 50000, 60000}};
   DO_IMM256("xvsleiu.d", 31, model_cmp(exp.d.u8, a.d.u8, (vec256){.d.u64 = {31, 31, 31, 31}}.d.u8, 64, 4, 0, 1), __lasx_xvslei_du(a.v, 31));
   a = (vec256){.d.i16 = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
   DO_IMM256("xvslti.wu", 7, model_cmp(exp.d.u8, a.d.u8, (vec256){.d.u32 = {7, 7, 7, 7, 7, 7, 7, 7}}.d.u8, 32, 8, 0, 2), __lasx_xvslti_wu(a.v, 7));
   DO_IMM256("xvslti.b", 1, model_cmp(exp.d.u8, a.d.u8, (vec256){.d.i8 = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}}.d.u8, 8, 32, 1, 2), __lasx_xvslti_b(a.v, 1));
   a = (vec256){.d.i32 = {-100, -50, 0, 50, 100, -200, -300, 400}};
   DO_IMM256("xvslti.h", -3, model_cmp(exp.d.u8, a.d.u8, (vec256){.d.i16 = {-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3}}.d.u8, 16, 16, 1, 2), __lasx_xvslti_h(a.v, -3));
   a = (vec256){.d.i64 = {-1000, 1000, -2000, 2000}};
   DO_IMM256("xvslti.d", -5, model_cmp(exp.d.u8, a.d.u8, (vec256){.d.i64 = {-5, -5, -5, -5}}.d.u8, 64, 4, 1, 2), __lasx_xvslti_d(a.v, -5));
   a = (vec256){.d.u16 = {0, 1, 100, 500, 1000, 5000, 60000, 65000,
                          200, 300, 400, 800, 2000, 7000, 50000, 60000}};
   DO_IMM256("xvsltiu.b", 1, model_cmp(exp.d.u8, a.d.u8, (vec256){.d.u8 = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}}.d.u8, 8, 32, 0, 2), __lasx_xvslti_bu(a.v, 1));
   a = (vec256){.d.u32 = {0, 1, 1000, 4000000000u, 500, 600, 700, 800}};
   DO_IMM256("xvsltiu.h", 7, model_cmp(exp.d.u8, a.d.u8, (vec256){.d.u16 = {7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7}}.d.u8, 16, 16, 0, 2), __lasx_xvslti_hu(a.v, 7));
   a = (vec256){.d.u64 = {0, 100000, 200000, 300000}};
   DO_IMM256("xvsltiu.w", 31, model_cmp(exp.d.u8, a.d.u8, (vec256){.d.u32 = {31, 31, 31, 31, 31, 31, 31, 31}}.d.u8, 32, 8, 0, 2), __lasx_xvslti_wu(a.v, 31));
   a = (vec256){.d.u16 = {0, 1, 100, 500, 1000, 5000, 60000, 65000,
                          200, 300, 400, 800, 2000, 7000, 50000, 60000}};
   DO_IMM256("xvsltiu.d", 31, model_cmp(exp.d.u8, a.d.u8, (vec256){.d.u64 = {31, 31, 31, 31}}.d.u8, 64, 4, 0, 2), __lasx_xvslti_du(a.v, 31));
   DO_UN256("xvmskltz.h", model_vmsk(exp.d.u8, a.d.u8, 16, sizeof(exp.d.u8), 0), __lasx_xvmskltz_h(a.v));
   DO_UN256("xvmskgez.b", model_vmsk(exp.d.u8, a.d.u8, 8, sizeof(exp.d.u8), 1), __lasx_xvmskgez_b(a.v));
   DO_UN256("xvmsknz.b", model_vmsk(exp.d.u8, a.d.u8, 8, sizeof(exp.d.u8), 2), __lasx_xvmsknz_b(a.v));
   a = aw;
   DO_UN256("xvmskltz.b", model_vmsk(exp.d.u8, a.d.u8, 8, sizeof(exp.d.u8), 0), __lasx_xvmskltz_b(a.v));
   a = ad;
   DO_UN256("xvmskltz.w", model_vmsk(exp.d.u8, a.d.u8, 32, sizeof(exp.d.u8), 0), __lasx_xvmskltz_w(a.v));
   a = (vec256){.d.u64 = {-1000, 1000, -2000, 2000}};
   DO_UN256("xvmskltz.d", model_vmsk(exp.d.u8, a.d.u8, 64, sizeof(exp.d.u8), 0), __lasx_xvmskltz_d(a.v));
}

static void test_vset(test_state* tst)
{
   vec256 z = {.d.u64 = {0, 0, 0, 0}};
   vec256 bh = {.d.u64 = {
      0x1122334455667788ULL, 0x00000000aabbccddULL,
      0x0102030405060708ULL, 0x1112131415161718ULL}};
   vec256 w = {.d.u32 = {1, 2, 0, 4, 5, 6, 7, 8}};
   vec256 d = {.d.u64 = {
      0x0102030405060708ULL, 0, 0x1112131415161718ULL, 0x2122232425262728ULL}};
   vec256 nz = {.d.u64 = {
      0x0102030405060708ULL, 0x1112131415161718ULL,
      0x2122232425262728ULL, 0x3132333435363738ULL}};

   DO_SET256("xbz.v.z", z, model_vset(z.d.u8, 8, sizeof(z.d.u8), 0), __lasx_xbz_v(z.v));
   DO_SET256("xbz.v.nz", nz, model_vset(nz.d.u8, 8, sizeof(nz.d.u8), 0), __lasx_xbz_v(nz.v));
   DO_SET256("xbnz.v.z", z, model_vset(z.d.u8, 8, sizeof(z.d.u8), 1), __lasx_xbnz_v(z.v));
   DO_SET256("xbnz.v.nz", nz, model_vset(nz.d.u8, 8, sizeof(nz.d.u8), 1), __lasx_xbnz_v(nz.v));

   DO_SET256("xbz.b.haszero", bh, model_vset(bh.d.u8, 8, 32, 2), __lasx_xbz_b(bh.v));
   DO_SET256("xbz.b.nz", nz, model_vset(nz.d.u8, 8, 32, 2), __lasx_xbz_b(nz.v));
   DO_SET256("xbz.h.haszero", bh, model_vset(bh.d.u8, 16, 16, 2), __lasx_xbz_h(bh.v));
   DO_SET256("xbz.h.nz", nz, model_vset(nz.d.u8, 16, 16, 2), __lasx_xbz_h(nz.v));
   DO_SET256("xbz.w.haszero", w, model_vset(w.d.u8, 32, 8, 2), __lasx_xbz_w(w.v));
   DO_SET256("xbz.w.nz", nz, model_vset(nz.d.u8, 32, 8, 2), __lasx_xbz_w(nz.v));
   DO_SET256("xbz.d.haszero", d, model_vset(d.d.u8, 64, 4, 2), __lasx_xbz_d(d.v));
   DO_SET256("xbz.d.nz", nz, model_vset(nz.d.u8, 64, 4, 2), __lasx_xbz_d(nz.v));

   DO_SET256("xbnz.b.haszero", bh, model_vset(bh.d.u8, 8, 32, 3), __lasx_xbnz_b(bh.v));
   DO_SET256("xbnz.b.nz", nz, model_vset(nz.d.u8, 8, 32, 3), __lasx_xbnz_b(nz.v));
   DO_SET256("xbnz.h.haszero", bh, model_vset(bh.d.u8, 16, 16, 3), __lasx_xbnz_h(bh.v));
   DO_SET256("xbnz.h.nz", nz, model_vset(nz.d.u8, 16, 16, 3), __lasx_xbnz_h(nz.v));
   DO_SET256("xbnz.w.haszero", w, model_vset(w.d.u8, 32, 8, 3), __lasx_xbnz_w(w.v));
   DO_SET256("xbnz.w.nz", nz, model_vset(nz.d.u8, 32, 8, 3), __lasx_xbnz_w(nz.v));
   DO_SET256("xbnz.d.haszero", d, model_vset(d.d.u8, 64, 4, 3), __lasx_xbnz_d(d.v));
   DO_SET256("xbnz.d.nz", nz, model_vset(nz.d.u8, 64, 4, 3), __lasx_xbnz_d(nz.v));
}

static void test_exth(test_state* tst)
{
   vec256 a = {.d.u8 = {
      0x81, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78,
      0x89, 0x9a, 0xab, 0xbc, 0xcd, 0xde, 0xef, 0x10,
      0x20, 0x31, 0x42, 0x53, 0x64, 0x75, 0x86, 0x97,
      0xa8, 0xb9, 0xca, 0xdb, 0xec, 0xfd, 0x0e, 0x1f}};
   vec256 got, exp;

   DO_UN256("xvexth.h.b", model_xvexth(exp.d.u8, a.d.u8, 8, 16, 8, 0), __lasx_xvexth_h_b(a.v));
   DO_UN256("xvexth.w.h", model_xvexth(exp.d.u8, a.d.u8, 16, 32, 4, 0), __lasx_xvexth_w_h(a.v));
   DO_UN256("xvexth.d.w", model_xvexth(exp.d.u8, a.d.u8, 32, 64, 2, 0), __lasx_xvexth_d_w(a.v));
   DO_UN256("xvexth.q.d", model_xvexth(exp.d.u8, a.d.u8, 64, 128, 1, 0), __lasx_xvexth_q_d(a.v));
   DO_UN256("xvexth.hu.bu", model_xvexth(exp.d.u8, a.d.u8, 8, 16, 8, 1), __lasx_xvexth_hu_bu(a.v));
   DO_UN256("xvexth.wu.hu", model_xvexth(exp.d.u8, a.d.u8, 16, 32, 4, 1), __lasx_xvexth_wu_hu(a.v));
   DO_UN256("xvexth.du.wu", model_xvexth(exp.d.u8, a.d.u8, 32, 64, 2, 1), __lasx_xvexth_du_wu(a.v));
   DO_UN256("xvexth.qu.du", model_xvexth(exp.d.u8, a.d.u8, 64, 128, 1, 1), __lasx_xvexth_qu_du(a.v));

   DO_UN256("vext2xv.h.b", model_vext2xv(exp.d.u8, a.d.u8, 8, 16, 16, 0), __lasx_vext2xv_h_b(a.v));
   DO_UN256("vext2xv.w.b", model_vext2xv(exp.d.u8, a.d.u8, 8, 32, 8, 0), __lasx_vext2xv_w_b(a.v));
   DO_UN256("vext2xv.d.b", model_vext2xv(exp.d.u8, a.d.u8, 8, 64, 4, 0), __lasx_vext2xv_d_b(a.v));
   DO_UN256("vext2xv.w.h", model_vext2xv(exp.d.u8, a.d.u8, 16, 32, 8, 0), __lasx_vext2xv_w_h(a.v));
   DO_UN256("vext2xv.d.h", model_vext2xv(exp.d.u8, a.d.u8, 16, 64, 4, 0), __lasx_vext2xv_d_h(a.v));
   DO_UN256("vext2xv.d.w", model_vext2xv(exp.d.u8, a.d.u8, 32, 64, 4, 0), __lasx_vext2xv_d_w(a.v));
   DO_UN256("vext2xv.hu.bu", model_vext2xv(exp.d.u8, a.d.u8, 8, 16, 16, 1), __lasx_vext2xv_hu_bu(a.v));
   DO_UN256("vext2xv.wu.bu", model_vext2xv(exp.d.u8, a.d.u8, 8, 32, 8, 1), __lasx_vext2xv_wu_bu(a.v));
   DO_UN256("vext2xv.du.bu", model_vext2xv(exp.d.u8, a.d.u8, 8, 64, 4, 1), __lasx_vext2xv_du_bu(a.v));
   DO_UN256("vext2xv.wu.hu", model_vext2xv(exp.d.u8, a.d.u8, 16, 32, 8, 1), __lasx_vext2xv_wu_hu(a.v));
   DO_UN256("vext2xv.du.hu", model_vext2xv(exp.d.u8, a.d.u8, 16, 64, 4, 1), __lasx_vext2xv_du_hu(a.v));
   DO_UN256("vext2xv.du.wu", model_vext2xv(exp.d.u8, a.d.u8, 32, 64, 4, 1), __lasx_vext2xv_du_wu(a.v));
}

static void test_unary_more(test_state* tst)
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
   vec256 got, exp;

   DO_UN256("xvclo.d", model_clo(exp.d.u8, a.d.u8, 64, 4), __lasx_xvclo_d(a.v));
   DO_UN256("xvclz.b", model_clz(exp.d.u8, a.d.u8, 8, 32), __lasx_xvclz_b(a.v));
   DO_UN256("xvpcnt.h", model_pcnt(exp.d.u8, a.d.u8, 16, 16), __lasx_xvpcnt_h(a.v));
   DO_UN256("xvneg.b", model_neg(exp.d.u8, a.d.u8, 8, 32), __lasx_xvneg_b(a.v));
   DO_UN256("xvneg.w", model_neg(exp.d.u8, a.d.u8, 32, 8), __lasx_xvneg_w(a.v));
   DO_UN256("xvneg.d", model_neg(exp.d.u8, a.d.u8, 64, 4), __lasx_xvneg_d(a.v));
   DO_BIN256("xvsigncov.h", model_signcov(exp.d.u8, a.d.u8, b.d.u8, 16, 16), __lasx_xvsigncov_h(a.v, b.v));
   DO_BIN256("xvsigncov.d", model_signcov(exp.d.u8, a.d.u8, b.d.u8, 64, 4), __lasx_xvsigncov_d(a.v, b.v));
}

static void test_vshuf(test_state* tst)
{
   vec256 j = {.d.u8 = {
      0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
      0x20, 0x21, 0x22, 0x23, 0x30, 0x31, 0x32, 0x33,
      0x40, 0x41, 0x42, 0x43, 0x50, 0x51, 0x52, 0x53,
      0x60, 0x61, 0x62, 0x63, 0x70, 0x71, 0x72, 0x73}};
   vec256 k = {.d.u8 = {
      0x80, 0x81, 0x82, 0x83, 0x90, 0x91, 0x92, 0x93,
      0xa0, 0xa1, 0xa2, 0xa3, 0xb0, 0xb1, 0xb2, 0xb3,
      0xc0, 0xc1, 0xc2, 0xc3, 0xd0, 0xd1, 0xd2, 0xd3,
      0xe0, 0xe1, 0xe2, 0xe3, 0xf0, 0xf1, 0xf2, 0xf3}};
   vec256 selb = {.d.u8 = {
      0x00, 0x01, 0x08, 0x09, 0x10, 0x11, 0x18, 0x19,
      0x80, 0x81, 0x88, 0x89, 0xc0, 0xc1, 0xc8, 0xc9,
      0x00, 0x01, 0x08, 0x09, 0x10, 0x11, 0x18, 0x19,
      0x80, 0x81, 0x88, 0x89, 0xc0, 0xc1, 0xc8, 0xc9}};
   // FIXME: Manual says high selector bits for xvshuf.h/w/d are disputed
   // across LoongArch variants. Keep tests to portable low-bit selectors.
   vec256 selh = {.d.u16 = {
      0, 1, 2, 3, 4, 5, 6, 7,
      0, 1, 2, 3, 4, 5, 6, 7}};
   vec256 selw = {.d.u32 = {0, 1, 4, 7, 0, 1, 4, 7}};
   vec256 seld = {.d.u64 = {0, 3, 0, 3}};
   vec256 got, exp;

   model_xvshuf_b(exp.d.u8, j.d.u8, k.d.u8, selb.d.u8);
   got.v = __lasx_xvshuf_b(j.v, k.v, selb.v);
   print_case3("xvshuf.b", &j, &k, &selb, &got);
   check_bytes(tst, "xvshuf.b", got.d.u8, exp.d.u8, sizeof(exp.d.u8));

   model_xvshuf_hwd(exp.d.u8, selh.d.u8, j.d.u8, k.d.u8, 16);
   got.v = __lasx_xvshuf_h(selh.v, j.v, k.v);
   print_case3("xvshuf.h", &selh, &j, &k, &got);
   check_bytes(tst, "xvshuf.h", got.d.u8, exp.d.u8, sizeof(exp.d.u8));

   model_xvshuf_hwd(exp.d.u8, selw.d.u8, j.d.u8, k.d.u8, 32);
   got.v = __lasx_xvshuf_w(selw.v, j.v, k.v);
   print_case3("xvshuf.w", &selw, &j, &k, &got);
   check_bytes(tst, "xvshuf.w", got.d.u8, exp.d.u8, sizeof(exp.d.u8));

   model_xvshuf_hwd(exp.d.u8, seld.d.u8, j.d.u8, k.d.u8, 64);
   got.v = __lasx_xvshuf_d(seld.v, j.v, k.v);
   print_case3("xvshuf.d", &seld, &j, &k, &got);
   check_bytes(tst, "xvshuf.d", got.d.u8, exp.d.u8, sizeof(exp.d.u8));

   model_xvperm_w(exp.d.u8, j.d.u8, selw.d.u8);
   got.v = __lasx_xvperm_w(j.v, selw.v);
   print_case2("xvperm.w", &j, &selw, &got);
   check_bytes(tst, "xvperm.w", got.d.u8, exp.d.u8, sizeof(exp.d.u8));
}

static void test_shift(test_state* tst)
{
   vec256 a = {.d.u8 = {
      0x81, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78,
      0x89, 0x9a, 0xab, 0xbc, 0xcd, 0xde, 0xef, 0x10,
      0x20, 0x31, 0x42, 0x53, 0x64, 0x75, 0x86, 0x97,
      0xa8, 0xb9, 0xca, 0xdb, 0xec, 0xfd, 0x0e, 0x1f}};
   vec256 zero = {.d.u64 = {0, 0, 0, 0}};
   vec256 mix = {.d.u64 = {0, 63, 64, 65}};
   vec256 mixb = {.d.u8 = {7,8,9,15,1,2,3,4,5,6,7,8,9,10,11,12,
                           7,8,9,15,1,2,3,4,5,6,7,8,9,10,11,12}};
   vec256 mix16 = {.d.u16 = {0,1,8,15,0,1,8,15,0,1,8,15,0,1,8,15}};
   vec256 mix32 = {.d.u32 = {0,16,31,8,0,16,31,8}};
   vec256 got, exp, b;

   DO_IMM256("xvslli.w.0", 0, model_shift_imm(exp.d.u8, a.d.u8, 32, 8, 0, 0, 0), __lasx_xvslli_w(a.v, 0));
   DO_IMM256("xvslli.w.31", 31, model_shift_imm(exp.d.u8, a.d.u8, 32, 8, 31, 0, 0), __lasx_xvslli_w(a.v, 31));
   DO_IMM256("xvslli.b.0", 0, model_shift_imm(exp.d.u8, a.d.u8, 8, 32, 0, 0, 0), __lasx_xvslli_b(a.v, 0));
   DO_IMM256("xvslli.b.7", 7, model_shift_imm(exp.d.u8, a.d.u8, 8, 32, 7, 0, 0), __lasx_xvslli_b(a.v, 7));
   DO_IMM256("xvslli.h.0", 0, model_shift_imm(exp.d.u8, a.d.u8, 16, 16, 0, 0, 0), __lasx_xvslli_h(a.v, 0));
   DO_IMM256("xvslli.h.15", 15, model_shift_imm(exp.d.u8, a.d.u8, 16, 16, 15, 0, 0), __lasx_xvslli_h(a.v, 15));
   DO_IMM256("xvslli.d.0", 0, model_shift_imm(exp.d.u8, a.d.u8, 64, 4, 0, 0, 0), __lasx_xvslli_d(a.v, 0));
   DO_IMM256("xvslli.d.63", 63, model_shift_imm(exp.d.u8, a.d.u8, 64, 4, 63, 0, 0), __lasx_xvslli_d(a.v, 63));
   DO_IMM256("xvsrli.d.0", 0, model_shift_imm(exp.d.u8, a.d.u8, 64, 4, 0, 1, 0), __lasx_xvsrli_d(a.v, 0));
   DO_IMM256("xvsrli.d.63", 63, model_shift_imm(exp.d.u8, a.d.u8, 64, 4, 63, 1, 0), __lasx_xvsrli_d(a.v, 63));
   DO_IMM256("xvsrli.b.0", 0, model_shift_imm(exp.d.u8, a.d.u8, 8, 32, 0, 1, 0), __lasx_xvsrli_b(a.v, 0));
   DO_IMM256("xvsrli.b.7", 7, model_shift_imm(exp.d.u8, a.d.u8, 8, 32, 7, 1, 0), __lasx_xvsrli_b(a.v, 7));
   DO_IMM256("xvsrli.h.0", 0, model_shift_imm(exp.d.u8, a.d.u8, 16, 16, 0, 1, 0), __lasx_xvsrli_h(a.v, 0));
   DO_IMM256("xvsrli.h.15", 15, model_shift_imm(exp.d.u8, a.d.u8, 16, 16, 15, 1, 0), __lasx_xvsrli_h(a.v, 15));
   DO_IMM256("xvsrli.w.0", 0, model_shift_imm(exp.d.u8, a.d.u8, 32, 8, 0, 1, 0), __lasx_xvsrli_w(a.v, 0));
   DO_IMM256("xvsrli.w.31", 31, model_shift_imm(exp.d.u8, a.d.u8, 32, 8, 31, 1, 0), __lasx_xvsrli_w(a.v, 31));
   DO_IMM256("xvsrai.h.0", 0, model_shift_imm(exp.d.u8, a.d.u8, 16, 16, 0, 1, 1), __lasx_xvsrai_h(a.v, 0));
   DO_IMM256("xvsrai.h.15", 15, model_shift_imm(exp.d.u8, a.d.u8, 16, 16, 15, 1, 1), __lasx_xvsrai_h(a.v, 15));
   DO_IMM256("xvsrai.b.0", 0, model_shift_imm(exp.d.u8, a.d.u8, 8, 32, 0, 1, 1), __lasx_xvsrai_b(a.v, 0));
   DO_IMM256("xvsrai.b.7", 7, model_shift_imm(exp.d.u8, a.d.u8, 8, 32, 7, 1, 1), __lasx_xvsrai_b(a.v, 7));
   DO_IMM256("xvsrai.w.0", 0, model_shift_imm(exp.d.u8, a.d.u8, 32, 8, 0, 1, 1), __lasx_xvsrai_w(a.v, 0));
   DO_IMM256("xvsrai.w.31", 31, model_shift_imm(exp.d.u8, a.d.u8, 32, 8, 31, 1, 1), __lasx_xvsrai_w(a.v, 31));
   DO_IMM256("xvsrai.d.0", 0, model_shift_imm(exp.d.u8, a.d.u8, 64, 4, 0, 1, 1), __lasx_xvsrai_d(a.v, 0));
   DO_IMM256("xvsrai.d.63", 63, model_shift_imm(exp.d.u8, a.d.u8, 64, 4, 63, 1, 1), __lasx_xvsrai_d(a.v, 63));

   b = mix;
   DO_BIN256("xvsll.d.var", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 0, 0), __lasx_xvsll_d(a.v, b.v));
   b = zero;
   DO_BIN256("xvsll.d.zero", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 0, 0), __lasx_xvsll_d(a.v, b.v));
   b = mix;
   DO_BIN256("xvsrl.d.var", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 1, 0), __lasx_xvsrl_d(a.v, b.v));
   b = zero;
   DO_BIN256("xvsrl.d.zero", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 1, 0), __lasx_xvsrl_d(a.v, b.v));
   b = mix;
   DO_BIN256("xvsra.d.var", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 1, 1), __lasx_xvsra_d(a.v, b.v));
   b = zero;
   DO_BIN256("xvsra.d.zero", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 1, 1), __lasx_xvsra_d(a.v, b.v));
   b = mixb;
   DO_BIN256("xvsll.b.var", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 0, 0), __lasx_xvsll_b(a.v, b.v));
   b = zero;
   DO_BIN256("xvsll.b.zero", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 0, 0), __lasx_xvsll_b(a.v, b.v));
   b = mixb;
   DO_BIN256("xvsrl.b.var", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 1, 0), __lasx_xvsrl_b(a.v, b.v));
   b = zero;
   DO_BIN256("xvsrl.b.zero", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 1, 0), __lasx_xvsrl_b(a.v, b.v));
   b = mixb;
   DO_BIN256("xvsra.b.var", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 1, 1), __lasx_xvsra_b(a.v, b.v));
   b = zero;
   DO_BIN256("xvsra.b.zero", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 1, 1), __lasx_xvsra_b(a.v, b.v));
   b = mix16;
   DO_BIN256("xvsll.h.var", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 0, 0), __lasx_xvsll_h(a.v, b.v));
   b = zero;
   DO_BIN256("xvsll.h.zero", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 0, 0), __lasx_xvsll_h(a.v, b.v));
   b = mix16;
   DO_BIN256("xvsrl.h.var", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 1, 0), __lasx_xvsrl_h(a.v, b.v));
   b = zero;
   DO_BIN256("xvsrl.h.zero", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 1, 0), __lasx_xvsrl_h(a.v, b.v));
   b = mix16;
   DO_BIN256("xvsra.h.var", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 1, 1), __lasx_xvsra_h(a.v, b.v));
   b = zero;
   DO_BIN256("xvsra.h.zero", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 1, 1), __lasx_xvsra_h(a.v, b.v));
   b = mix32;
   DO_BIN256("xvsll.w.var", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 0, 0), __lasx_xvsll_w(a.v, b.v));
   b = zero;
   DO_BIN256("xvsll.w.zero", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 0, 0), __lasx_xvsll_w(a.v, b.v));
   b = mix32;
   DO_BIN256("xvsrl.w.var", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 1, 0), __lasx_xvsrl_w(a.v, b.v));
   b = zero;
   DO_BIN256("xvsrl.w.zero", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 1, 0), __lasx_xvsrl_w(a.v, b.v));
   b = mix32;
   DO_BIN256("xvsra.w.var", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 1, 1), __lasx_xvsra_w(a.v, b.v));
   b = zero;
   DO_BIN256("xvsra.w.zero", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 1, 1), __lasx_xvsra_w(a.v, b.v));
}

static void test_shift_round(test_state* tst)
{
   vec256 a = {.d.u8 = {
      0x81, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78,
      0x89, 0x9a, 0xab, 0xbc, 0xcd, 0xde, 0xef, 0x10,
      0x20, 0x31, 0x42, 0x53, 0x64, 0x75, 0x86, 0x97,
      0xa8, 0xb9, 0xca, 0xdb, 0xec, 0xfd, 0x0e, 0x1f}};
   vec256 zero = {.d.u64 = {0, 0, 0, 0}};
   vec256 mix = {.d.u64 = {0, 63, 64, 65}};
   vec256 mixb = {.d.u8 = {7,8,9,15,1,2,3,4,5,6,7,8,9,10,11,12,
                           7,8,9,15,1,2,3,4,5,6,7,8,9,10,11,12}};
   vec256 mix16 = {.d.u16 = {0,1,8,15,0,1,8,15,0,1,8,15,0,1,8,15}};
   vec256 mix32 = {.d.u32 = {0,16,31,8,0,16,31,8}};
   vec256 got, exp, b;

   b = mixb;
   DO_BIN256("xvsrlr.b.var", model_shift_round_var(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 0), __lasx_xvsrlr_b(a.v, b.v));
   b = zero;
   DO_BIN256("xvsrlr.b.zero", model_shift_round_var(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 0), __lasx_xvsrlr_b(a.v, b.v));
   b = mixb;
   DO_BIN256("xvsrar.b.var", model_shift_round_var(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 1), __lasx_xvsrar_b(a.v, b.v));
   b = zero;
   DO_BIN256("xvsrar.b.zero", model_shift_round_var(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 1), __lasx_xvsrar_b(a.v, b.v));
   b = mix16;
   DO_BIN256("xvsrlr.h.var", model_shift_round_var(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 0), __lasx_xvsrlr_h(a.v, b.v));
   b = zero;
   DO_BIN256("xvsrlr.h.zero", model_shift_round_var(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 0), __lasx_xvsrlr_h(a.v, b.v));
   b = mix16;
   DO_BIN256("xvsrar.h.var", model_shift_round_var(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 1), __lasx_xvsrar_h(a.v, b.v));
   b = zero;
   DO_BIN256("xvsrar.h.zero", model_shift_round_var(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 1), __lasx_xvsrar_h(a.v, b.v));
   b = mix32;
   DO_BIN256("xvsrlr.w.var", model_shift_round_var(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 0), __lasx_xvsrlr_w(a.v, b.v));
   b = zero;
   DO_BIN256("xvsrlr.w.zero", model_shift_round_var(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 0), __lasx_xvsrlr_w(a.v, b.v));
   b = mix32;
   DO_BIN256("xvsrar.w.var", model_shift_round_var(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 1), __lasx_xvsrar_w(a.v, b.v));
   b = zero;
   DO_BIN256("xvsrar.w.zero", model_shift_round_var(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 1), __lasx_xvsrar_w(a.v, b.v));
   b = mix;
   DO_BIN256("xvsrlr.d.var", model_shift_round_var(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 0), __lasx_xvsrlr_d(a.v, b.v));
   b = zero;
   DO_BIN256("xvsrlr.d.zero", model_shift_round_var(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 0), __lasx_xvsrlr_d(a.v, b.v));
   b = mix;
   DO_BIN256("xvsrar.d.var", model_shift_round_var(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 1), __lasx_xvsrar_d(a.v, b.v));
   b = zero;
   DO_BIN256("xvsrar.d.zero", model_shift_round_var(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 1), __lasx_xvsrar_d(a.v, b.v));
}

static void test_shift_narrow(test_state* tst)
{
   vec256 a = {.d.u32 = {
      0x34231281, 0x78675645, 0xbcab9a89, 0x10efdecd,
      0x53423120, 0x97867564, 0xdbcab9a8, 0x1f0efdec}};
   vec256 a16 = {.d.u16 = {
      0x8123, 0x3456, 0x9ab8, 0xdead, 0xbeef, 0x1234, 0x5678, 0x90ab,
      0xcdef, 0x2468, 0x1357, 0xfedc, 0xba98, 0x7654, 0x3210, 0xffff}};
   vec256 zero = {.d.u32 = {0, 0, 0, 0, 0, 0, 0, 0}};
   vec256 mix = {.d.u32 = {0, 1, 32, 33, 0, 1, 32, 33}};
   vec256 mix16 = {.d.u16 = {0, 8, 15, 1, 0, 8, 15, 1,
                             0, 8, 15, 1, 0, 8, 15, 1}};
   vec256 got, exp, b;

   b = mix16;
   DO_BIN256("xvsrln.b.h.var", model_narrow_shift(exp.d.u8, a16.d.u8, b.d.u8, 16, 8, 8, 0, sizeof(exp.d.u8)), __lasx_xvsrln_b_h(a16.v, b.v));
   b = zero;
   DO_BIN256("xvsrln.b.h.zero", model_narrow_shift(exp.d.u8, a16.d.u8, b.d.u8, 16, 8, 8, 0, sizeof(exp.d.u8)), __lasx_xvsrln_b_h(a16.v, b.v));
   b = mix;
   DO_BIN256("xvsrln.h.w.var", model_narrow_shift(exp.d.u8, a.d.u8, b.d.u8, 32, 16, 4, 0, sizeof(exp.d.u8)), __lasx_xvsrln_h_w(a.v, b.v));
   b = zero;
   DO_BIN256("xvsrln.h.w.zero", model_narrow_shift(exp.d.u8, a.d.u8, b.d.u8, 32, 16, 4, 0, sizeof(exp.d.u8)), __lasx_xvsrln_h_w(a.v, b.v));
   b = mix;
   DO_BIN256("xvsrln.w.d.var", model_narrow_shift(exp.d.u8, a.d.u8, b.d.u8, 64, 32, 2, 0, sizeof(exp.d.u8)), __lasx_xvsrln_w_d(a.v, b.v));
   b = zero;
   DO_BIN256("xvsrln.w.d.zero", model_narrow_shift(exp.d.u8, a.d.u8, b.d.u8, 64, 32, 2, 0, sizeof(exp.d.u8)), __lasx_xvsrln_w_d(a.v, b.v));
   b = mix16;
   DO_BIN256("xvsran.b.h.var", model_narrow_shift(exp.d.u8, a16.d.u8, b.d.u8, 16, 8, 8, 1, sizeof(exp.d.u8)), __lasx_xvsran_b_h(a16.v, b.v));
   b = zero;
   DO_BIN256("xvsran.b.h.zero", model_narrow_shift(exp.d.u8, a16.d.u8, b.d.u8, 16, 8, 8, 1, sizeof(exp.d.u8)), __lasx_xvsran_b_h(a16.v, b.v));
   b = mix;
   DO_BIN256("xvsran.h.w.var", model_narrow_shift(exp.d.u8, a.d.u8, b.d.u8, 32, 16, 4, 1, sizeof(exp.d.u8)), __lasx_xvsran_h_w(a.v, b.v));
   b = zero;
   DO_BIN256("xvsran.h.w.zero", model_narrow_shift(exp.d.u8, a.d.u8, b.d.u8, 32, 16, 4, 1, sizeof(exp.d.u8)), __lasx_xvsran_h_w(a.v, b.v));
   b = mix;
   DO_BIN256("xvsran.w.d.var", model_narrow_shift(exp.d.u8, a.d.u8, b.d.u8, 64, 32, 2, 1, sizeof(exp.d.u8)), __lasx_xvsran_w_d(a.v, b.v));
   b = zero;
   DO_BIN256("xvsran.w.d.zero", model_narrow_shift(exp.d.u8, a.d.u8, b.d.u8, 64, 32, 2, 1, sizeof(exp.d.u8)), __lasx_xvsran_w_d(a.v, b.v));
}

static void test_shift_narrow_round(test_state* tst)
{
   vec256 a = {.d.u32 = {
      0x34231281, 0x78675645, 0xbcab9a89, 0x10efdecd,
      0x53423120, 0x97867564, 0xdbcab9a8, 0x1f0efdec}};
   vec256 a16 = {.d.u16 = {
      0x8123, 0x3456, 0x9ab8, 0xdead, 0xbeef, 0x1234, 0x5678, 0x90ab,
      0xcdef, 0x2468, 0x1357, 0xfedc, 0xba98, 0x7654, 0x3210, 0xffff}};
   vec256 zero = {.d.u32 = {0, 0, 0, 0, 0, 0, 0, 0}};
   vec256 mix = {.d.u32 = {0, 1, 32, 33, 0, 1, 32, 33}};
   vec256 mix16 = {.d.u16 = {0, 8, 15, 1, 0, 8, 15, 1,
                             0, 8, 15, 1, 0, 8, 15, 1}};
   vec256 got, exp, b;

   b = mix16;
   DO_BIN256("xvsrlrn.b.h.var", model_narrow_shift_round(exp.d.u8, a16.d.u8, b.d.u8, 16, 8, 8, 0, sizeof(exp.d.u8)), __lasx_xvsrlrn_b_h(a16.v, b.v));
   b = zero;
   DO_BIN256("xvsrlrn.b.h.zero", model_narrow_shift_round(exp.d.u8, a16.d.u8, b.d.u8, 16, 8, 8, 0, sizeof(exp.d.u8)), __lasx_xvsrlrn_b_h(a16.v, b.v));
   b = mix;
   DO_BIN256("xvsrlrn.h.w.var", model_narrow_shift_round(exp.d.u8, a.d.u8, b.d.u8, 32, 16, 4, 0, sizeof(exp.d.u8)), __lasx_xvsrlrn_h_w(a.v, b.v));
   b = zero;
   DO_BIN256("xvsrlrn.h.w.zero", model_narrow_shift_round(exp.d.u8, a.d.u8, b.d.u8, 32, 16, 4, 0, sizeof(exp.d.u8)), __lasx_xvsrlrn_h_w(a.v, b.v));
   b = mix;
   DO_BIN256("xvsrlrn.w.d.var", model_narrow_shift_round(exp.d.u8, a.d.u8, b.d.u8, 64, 32, 2, 0, sizeof(exp.d.u8)), __lasx_xvsrlrn_w_d(a.v, b.v));
   b = zero;
   DO_BIN256("xvsrlrn.w.d.zero", model_narrow_shift_round(exp.d.u8, a.d.u8, b.d.u8, 64, 32, 2, 0, sizeof(exp.d.u8)), __lasx_xvsrlrn_w_d(a.v, b.v));
   b = mix16;
   DO_BIN256("xvsrarn.b.h.var", model_narrow_shift_round(exp.d.u8, a16.d.u8, b.d.u8, 16, 8, 8, 1, sizeof(exp.d.u8)), __lasx_xvsrarn_b_h(a16.v, b.v));
   b = zero;
   DO_BIN256("xvsrarn.b.h.zero", model_narrow_shift_round(exp.d.u8, a16.d.u8, b.d.u8, 16, 8, 8, 1, sizeof(exp.d.u8)), __lasx_xvsrarn_b_h(a16.v, b.v));
   b = mix;
   DO_BIN256("xvsrarn.h.w.var", model_narrow_shift_round(exp.d.u8, a.d.u8, b.d.u8, 32, 16, 4, 1, sizeof(exp.d.u8)), __lasx_xvsrarn_h_w(a.v, b.v));
   b = zero;
   DO_BIN256("xvsrarn.h.w.zero", model_narrow_shift_round(exp.d.u8, a.d.u8, b.d.u8, 32, 16, 4, 1, sizeof(exp.d.u8)), __lasx_xvsrarn_h_w(a.v, b.v));
   b = mix;
   DO_BIN256("xvsrarn.w.d.var", model_narrow_shift_round(exp.d.u8, a.d.u8, b.d.u8, 64, 32, 2, 1, sizeof(exp.d.u8)), __lasx_xvsrarn_w_d(a.v, b.v));
   b = zero;
   DO_BIN256("xvsrarn.w.d.zero", model_narrow_shift_round(exp.d.u8, a.d.u8, b.d.u8, 64, 32, 2, 1, sizeof(exp.d.u8)), __lasx_xvsrarn_w_d(a.v, b.v));
}

static void test_rotate(test_state* tst)
{
   vec256 a = {.d.u8 = {
      0x81, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78,
      0x89, 0x9a, 0xab, 0xbc, 0xcd, 0xde, 0xef, 0x10,
      0x20, 0x31, 0x42, 0x53, 0x64, 0x75, 0x86, 0x97,
      0xa8, 0xb9, 0xca, 0xdb, 0xec, 0xfd, 0x0e, 0x1f}};
   vec256 zero = {.d.u64 = {0, 0, 0, 0}};
   vec256 mix = {.d.u64 = {0, 63, 64, 65}};
   vec256 mixb = {.d.u8 = {7,8,9,15,1,2,3,4,5,6,7,8,9,10,11,12,
                           7,8,9,15,1,2,3,4,5,6,7,8,9,10,11,12}};
   vec256 mix16 = {.d.u16 = {0,1,8,15,0,1,8,15,0,1,8,15,0,1,8,15}};
   vec256 mix32 = {.d.u32 = {0,16,31,8,0,16,31,8}};
   vec256 got, exp, b;

   DO_IMM256("xvrotri.b.0", 0, model_rotate_imm(exp.d.u8, a.d.u8, 8, 32, 0), __lasx_xvrotri_b(a.v, 0));
   DO_IMM256("xvrotri.b.7", 7, model_rotate_imm(exp.d.u8, a.d.u8, 8, 32, 7), __lasx_xvrotri_b(a.v, 7));
   DO_IMM256("xvrotri.h.0", 0, model_rotate_imm(exp.d.u8, a.d.u8, 16, 16, 0), __lasx_xvrotri_h(a.v, 0));
   DO_IMM256("xvrotri.h.15", 15, model_rotate_imm(exp.d.u8, a.d.u8, 16, 16, 15), __lasx_xvrotri_h(a.v, 15));
   DO_IMM256("xvrotri.w.0", 0, model_rotate_imm(exp.d.u8, a.d.u8, 32, 8, 0), __lasx_xvrotri_w(a.v, 0));
   DO_IMM256("xvrotri.w.31", 31, model_rotate_imm(exp.d.u8, a.d.u8, 32, 8, 31), __lasx_xvrotri_w(a.v, 31));
   DO_IMM256("xvrotri.d.0", 0, model_rotate_imm(exp.d.u8, a.d.u8, 64, 4, 0), __lasx_xvrotri_d(a.v, 0));
   DO_IMM256("xvrotri.d.63", 63, model_rotate_imm(exp.d.u8, a.d.u8, 64, 4, 63), __lasx_xvrotri_d(a.v, 63));

   b = mixb;
   DO_BIN256("xvrotr.b.var", model_rotate_var(exp.d.u8, a.d.u8, b.d.u8, 8, 32), __lasx_xvrotr_b(a.v, b.v));
   b = zero;
   DO_BIN256("xvrotr.b.zero", model_rotate_var(exp.d.u8, a.d.u8, b.d.u8, 8, 32), __lasx_xvrotr_b(a.v, b.v));
   b = mix16;
   DO_BIN256("xvrotr.h.var", model_rotate_var(exp.d.u8, a.d.u8, b.d.u8, 16, 16), __lasx_xvrotr_h(a.v, b.v));
   b = zero;
   DO_BIN256("xvrotr.h.zero", model_rotate_var(exp.d.u8, a.d.u8, b.d.u8, 16, 16), __lasx_xvrotr_h(a.v, b.v));
   b = mix32;
   DO_BIN256("xvrotr.w.var", model_rotate_var(exp.d.u8, a.d.u8, b.d.u8, 32, 8), __lasx_xvrotr_w(a.v, b.v));
   b = zero;
   DO_BIN256("xvrotr.w.zero", model_rotate_var(exp.d.u8, a.d.u8, b.d.u8, 32, 8), __lasx_xvrotr_w(a.v, b.v));
   b = mix;
   DO_BIN256("xvrotr.d.var", model_rotate_var(exp.d.u8, a.d.u8, b.d.u8, 64, 4), __lasx_xvrotr_d(a.v, b.v));
   b = zero;
   DO_BIN256("xvrotr.d.zero", model_rotate_var(exp.d.u8, a.d.u8, b.d.u8, 64, 4), __lasx_xvrotr_d(a.v, b.v));
}

static void test_logic_bit(test_state* tst)
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
      0x00, 0xff, 0x0f, 0xf0, 0x33, 0xcc, 0x55, 0xaa,
      0x01, 0x80, 0x7f, 0xfe, 0x10, 0x08, 0x04, 0x02,
      0xaa, 0x55, 0xcc, 0x33, 0xf0, 0x0f, 0xff, 0x00,
      0x11, 0x22, 0x44, 0x88, 0x7f, 0x3f, 0x1f, 0x0f}};
   vec256 got, exp;

   DO_BIN256("xvand.v", model_logic2(exp.d.u8, a.d.u8, b.d.u8, sizeof(exp.d.u8), 0), __lasx_xvand_v(a.v, b.v));
   DO_BIN256("xvor.v", model_logic2(exp.d.u8, a.d.u8, b.d.u8, sizeof(exp.d.u8), 1), __lasx_xvor_v(a.v, b.v));
   DO_BIN256("xvxor.v", model_logic2(exp.d.u8, a.d.u8, b.d.u8, sizeof(exp.d.u8), 2), __lasx_xvxor_v(a.v, b.v));
   DO_BIN256("xvnor.v", model_logic2(exp.d.u8, a.d.u8, b.d.u8, sizeof(exp.d.u8), 3), __lasx_xvnor_v(a.v, b.v));
   DO_BIN256("xvandn.v", model_logic2(exp.d.u8, a.d.u8, b.d.u8, sizeof(exp.d.u8), 4), __lasx_xvandn_v(a.v, b.v));
   DO_BIN256("xvorn.v", model_logic2(exp.d.u8, a.d.u8, b.d.u8, sizeof(exp.d.u8), 5), __lasx_xvorn_v(a.v, b.v));
   DO_IMM256("xvandi.b", 0x5a, model_logic_imm(exp.d.u8, a.d.u8, sizeof(exp.d.u8), 0x5a, 0), __lasx_xvandi_b(a.v, 0x5a));
   DO_IMM256("xvori.b", 0xa5, model_logic_imm(exp.d.u8, a.d.u8, sizeof(exp.d.u8), 0xa5, 1), __lasx_xvori_b(a.v, 0xa5));
   DO_IMM256("xvxori.b", 0xff, model_logic_imm(exp.d.u8, a.d.u8, sizeof(exp.d.u8), 0xff, 2), __lasx_xvxori_b(a.v, 0xff));
   DO_IMM256("xvnori.b", 0xa5, model_logic_imm(exp.d.u8, a.d.u8, sizeof(exp.d.u8), 0xa5, 3), __lasx_xvnori_b(a.v, 0xa5));
   DO_TRI256("xvbitsel.v", model_bitsel(exp.d.u8, a.d.u8, b.d.u8, c.d.u8, sizeof(exp.d.u8)), __lasx_xvbitsel_v(a.v, b.v, c.v));
   DO_IMM256("xvbitseli.b", 0x12, model_bitseli(exp.d.u8, a.d.u8, b.d.u8, 0x12, sizeof(exp.d.u8)), __lasx_xvbitseli_b(a.v, b.v, 0x12));
   DO_IMM256("xvbitseli.b.0xff", 0xff, model_bitseli(exp.d.u8, a.d.u8, b.d.u8, 0xff, sizeof(exp.d.u8)), __lasx_xvbitseli_b(a.v, b.v, 0xff));
   DO_IMM256("xvbitseli.b.0x00", 0x00, model_bitseli(exp.d.u8, a.d.u8, b.d.u8, 0x00, sizeof(exp.d.u8)), __lasx_xvbitseli_b(a.v, b.v, 0x00));

   b = (vec256){.d.u8 = {
      0, 7, 8, 15, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
      13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
   DO_BIN256("xvbitclr.b", model_bitop_var(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 1), __lasx_xvbitclr_b(a.v, b.v));
   b = (vec256){.d.u16 = {0, 15, 16, 31, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
   DO_BIN256("xvbitclr.h", model_bitop_var(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 1), __lasx_xvbitclr_h(a.v, b.v));
   b = (vec256){.d.u32 = {0, 31, 32, 5, 0, 31, 32, 5}};
   DO_BIN256("xvbitclr.w", model_bitop_var(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 1), __lasx_xvbitclr_w(a.v, b.v));
   b = (vec256){.d.u64 = {0, 63, 64, 65}};
   DO_BIN256("xvbitclr.d", model_bitop_var(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 1), __lasx_xvbitclr_d(a.v, b.v));
   b = (vec256){.d.u8 = {
      0, 7, 8, 15, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
      13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
   DO_BIN256("xvbitset.b", model_bitop_var(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 2), __lasx_xvbitset_b(a.v, b.v));
   b = (vec256){.d.u16 = {0, 15, 16, 31, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
   DO_BIN256("xvbitset.h", model_bitop_var(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 2), __lasx_xvbitset_h(a.v, b.v));
   b = (vec256){.d.u32 = {0, 31, 32, 5, 0, 31, 32, 5}};
   DO_BIN256("xvbitset.w", model_bitop_var(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 2), __lasx_xvbitset_w(a.v, b.v));
   b = (vec256){.d.u64 = {0, 63, 64, 65}};
   DO_BIN256("xvbitset.d", model_bitop_var(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 2), __lasx_xvbitset_d(a.v, b.v));
   b = (vec256){.d.u8 = {
      0, 7, 8, 15, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
      13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
   DO_BIN256("xvbitrev.b", model_bitop_var(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 0), __lasx_xvbitrev_b(a.v, b.v));
   b = (vec256){.d.u16 = {0, 15, 16, 31, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
   DO_BIN256("xvbitrev.h", model_bitop_var(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 0), __lasx_xvbitrev_h(a.v, b.v));
   b = (vec256){.d.u32 = {0, 31, 32, 5, 0, 31, 32, 5}};
   DO_BIN256("xvbitrev.w", model_bitop_var(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 0), __lasx_xvbitrev_w(a.v, b.v));
   b = (vec256){.d.u64 = {0, 63, 64, 65}};
   DO_BIN256("xvbitrev.d", model_bitop_var(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 0), __lasx_xvbitrev_d(a.v, b.v));
   DO_IMM256("xvbitclri.b.0", 0, model_bitop_imm(exp.d.u8, a.d.u8, 8, 32, 0, 0), __lasx_xvbitclri_b(a.v, 0));
   DO_IMM256("xvbitclri.b.7", 7, model_bitop_imm(exp.d.u8, a.d.u8, 8, 32, 7, 0), __lasx_xvbitclri_b(a.v, 7));
   DO_IMM256("xvbitclri.h.0", 0, model_bitop_imm(exp.d.u8, a.d.u8, 16, 16, 0, 0), __lasx_xvbitclri_h(a.v, 0));
   DO_IMM256("xvbitclri.h.15", 15, model_bitop_imm(exp.d.u8, a.d.u8, 16, 16, 15, 0), __lasx_xvbitclri_h(a.v, 15));
   DO_IMM256("xvbitclri.w.0", 0, model_bitop_imm(exp.d.u8, a.d.u8, 32, 8, 0, 0), __lasx_xvbitclri_w(a.v, 0));
   DO_IMM256("xvbitclri.w.31", 31, model_bitop_imm(exp.d.u8, a.d.u8, 32, 8, 31, 0), __lasx_xvbitclri_w(a.v, 31));
   DO_IMM256("xvbitclri.d.0", 0, model_bitop_imm(exp.d.u8, a.d.u8, 64, 4, 0, 0), __lasx_xvbitclri_d(a.v, 0));
   DO_IMM256("xvbitclri.d.63", 63, model_bitop_imm(exp.d.u8, a.d.u8, 64, 4, 63, 0), __lasx_xvbitclri_d(a.v, 63));
   DO_IMM256("xvbitseti.b.0", 0, model_bitop_imm(exp.d.u8, a.d.u8, 8, 32, 0, 1), __lasx_xvbitseti_b(a.v, 0));
   DO_IMM256("xvbitseti.b.7", 7, model_bitop_imm(exp.d.u8, a.d.u8, 8, 32, 7, 1), __lasx_xvbitseti_b(a.v, 7));
   DO_IMM256("xvbitseti.h.0", 0, model_bitop_imm(exp.d.u8, a.d.u8, 16, 16, 0, 1), __lasx_xvbitseti_h(a.v, 0));
   DO_IMM256("xvbitseti.h.15", 15, model_bitop_imm(exp.d.u8, a.d.u8, 16, 16, 15, 1), __lasx_xvbitseti_h(a.v, 15));
   DO_IMM256("xvbitseti.w.0", 0, model_bitop_imm(exp.d.u8, a.d.u8, 32, 8, 0, 1), __lasx_xvbitseti_w(a.v, 0));
   DO_IMM256("xvbitseti.w.31", 31, model_bitop_imm(exp.d.u8, a.d.u8, 32, 8, 31, 1), __lasx_xvbitseti_w(a.v, 31));
   DO_IMM256("xvbitseti.d.0", 0, model_bitop_imm(exp.d.u8, a.d.u8, 64, 4, 0, 1), __lasx_xvbitseti_d(a.v, 0));
   DO_IMM256("xvbitseti.d.63", 63, model_bitop_imm(exp.d.u8, a.d.u8, 64, 4, 63, 1), __lasx_xvbitseti_d(a.v, 63));
   DO_IMM256("xvbitrevi.h.0", 0, model_bitop_imm(exp.d.u8, a.d.u8, 16, 16, 0, 2), __lasx_xvbitrevi_h(a.v, 0));
   DO_IMM256("xvbitrevi.h.15", 15, model_bitop_imm(exp.d.u8, a.d.u8, 16, 16, 15, 2), __lasx_xvbitrevi_h(a.v, 15));
   DO_IMM256("xvbitrevi.w.0", 0, model_bitop_imm(exp.d.u8, a.d.u8, 32, 8, 0, 2), __lasx_xvbitrevi_w(a.v, 0));
   DO_IMM256("xvbitrevi.w.31", 31, model_bitop_imm(exp.d.u8, a.d.u8, 32, 8, 31, 2), __lasx_xvbitrevi_w(a.v, 31));
   DO_IMM256("xvbitrevi.d.0", 0, model_bitop_imm(exp.d.u8, a.d.u8, 64, 4, 0, 2), __lasx_xvbitrevi_d(a.v, 0));
   DO_IMM256("xvbitrevi.d.63", 63, model_bitop_imm(exp.d.u8, a.d.u8, 64, 4, 63, 2), __lasx_xvbitrevi_d(a.v, 63));
}

static void test_lane_move(test_state* tst)
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
   vec256 got, exp;

   DO_BIN256("xvilvl.b", model_lane_mix_128chunk(exp.d.u8, a.d.u8, b.d.u8, 8, sizeof(exp.d.u8), 0), __lasx_xvilvl_b(a.v, b.v));
   DO_BIN256("xvilvh.h", model_lane_mix_128chunk(exp.d.u8, a.d.u8, b.d.u8, 16, sizeof(exp.d.u8), 1), __lasx_xvilvh_h(a.v, b.v));
   DO_BIN256("xvpickev.w", model_lane_mix_128chunk(exp.d.u8, a.d.u8, b.d.u8, 32, sizeof(exp.d.u8), 2), __lasx_xvpickev_w(a.v, b.v));
   DO_BIN256("xvpickod.d", model_lane_mix_128chunk(exp.d.u8, a.d.u8, b.d.u8, 64, sizeof(exp.d.u8), 3), __lasx_xvpickod_d(a.v, b.v));
   DO_IMM256("xvshuf4i.b.0", 0x00, model_shuf4i_128chunk(exp.d.u8, a.d.u8, 8, sizeof(exp.d.u8), 0x00), __lasx_xvshuf4i_b(a.v, 0x00));
   DO_IMM256("xvshuf4i.b.e4", 0xe4, model_shuf4i_128chunk(exp.d.u8, a.d.u8, 8, sizeof(exp.d.u8), 0xe4), __lasx_xvshuf4i_b(a.v, 0xe4));
   DO_IMM256("xvshuf4i.w.1b", 0x1b, model_shuf4i_128chunk(exp.d.u8, a.d.u8, 32, sizeof(exp.d.u8), 0x1b), __lasx_xvshuf4i_w(a.v, 0x1b));
   DO_BIN256("xvpermi.w.1b", model_permi_w_128chunk(exp.d.u8, a.d.u8, b.d.u8, sizeof(exp.d.u8), 0x1b), __lasx_xvpermi_w(a.v, b.v, 0x1b));
   DO_IMM256("xvpermi.d.00", 0x00, model_xvpermi_d(exp.d.u8, a.d.u8, 0x00), __lasx_xvpermi_d(a.v, 0x00));
   DO_IMM256("xvpermi.d.1b", 0x1b, model_xvpermi_d(exp.d.u8, a.d.u8, 0x1b), __lasx_xvpermi_d(a.v, 0x1b));
   DO_IMM256("xvpermi.d.e4", 0xe4, model_xvpermi_d(exp.d.u8, a.d.u8, 0xe4), __lasx_xvpermi_d(a.v, 0xe4));
   DO_BIN256("xvpermi.q.00", model_xvpermi_q(exp.d.u8, a.d.u8, b.d.u8, 0x00), __lasx_xvpermi_q(a.v, b.v, 0x00));
   DO_BIN256("xvpermi.q.5a", model_xvpermi_q(exp.d.u8, a.d.u8, b.d.u8, 0x5a), __lasx_xvpermi_q(a.v, b.v, 0x5a));
   DO_BIN256("xvpermi.q.ff", model_xvpermi_q(exp.d.u8, a.d.u8, b.d.u8, 0xff), __lasx_xvpermi_q(a.v, b.v, 0xff));
   DO_BIN256("xvextrins.b", model_extrins_b_128chunk(exp.d.u8, a.d.u8, b.d.u8, sizeof(exp.d.u8), 0x4f), __lasx_xvextrins_b(a.v, b.v, 0x4f));
}

static void test_lane_scalar(test_state* tst)
{
   vec256 a = {.d.u8 = {
      0x80, 0x7f, 0x01, 0xfe, 0x10, 0x20, 0x30, 0x40,
      0x50, 0x60, 0x70, 0x81, 0x90, 0xa0, 0xb0, 0xc0,
      0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
      0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xf0, 0x0f}};
   vec256 b = {.d.u32 = {
      0x11111111, 0x22222222, 0x33333333, 0x44444444,
      0xaaaaaaaa, 0xbbbbbbbb, 0xcccccccc, 0xdddddddd}};
   vec256 got, exp;

   DO_INS256("xvinsgr2vr.w.5", 5, 0x13579bdfU,
             model_insgr2vr(exp.d.u8, a.d.u8, 32, 5, 0x13579bdfU, sizeof(exp.d.u8)),
             __lasx_xvinsgr2vr_w(a.v, 0x13579bdfU, 5));
   DO_INS256("xvinsgr2vr.d.3", 3, 0xfedcba9876543210ULL,
             model_insgr2vr(exp.d.u8, a.d.u8, 64, 3, 0xfedcba9876543210ULL, sizeof(exp.d.u8)),
             __lasx_xvinsgr2vr_d(a.v, 0xfedcba9876543210ULL, 3));

   DO_PICK256("xvpickve2gr.w.5", 5, model_pickve2gr(a.d.u8, 32, 5, 1),
              __lasx_xvpickve2gr_w(a.v, 5));
   DO_PICK256("xvpickve2gr.wu.5", 5, model_pickve2gr(a.d.u8, 32, 5, 0),
              __lasx_xvpickve2gr_wu(a.v, 5));
   DO_PICK256("xvpickve2gr.du.3", 3, model_pickve2gr(a.d.u8, 64, 3, 0),
              __lasx_xvpickve2gr_du(a.v, 3));

   DO_RV256("xvreplgr2vr.b", UINT64_MAX - 1,
            model_replgr2vr(exp.d.u8, 8, sizeof(exp.d.u8), UINT64_MAX - 1),
            __lasx_xvreplgr2vr_b(-2));
   DO_RV256("xvreplgr2vr.d", 0x0123456789abcdefULL,
            model_replgr2vr(exp.d.u8, 64, sizeof(exp.d.u8), 0x0123456789abcdefULL),
            __lasx_xvreplgr2vr_d(0x0123456789abcdefULL));

   model_xvreplve(exp.d.u8, a.d.u8, 32, 1);
   got.v = __lasx_xvreplve_w(a.v, 1);
   print_case2_imm("xvreplve.w.1", &a, 1, &got);
   check_bytes(tst, "xvreplve.w.1", got.d.u8, exp.d.u8, sizeof(exp.d.u8));

   model_xvreplve(exp.d.u8, a.d.u8, 16, 7);
   got.v = __lasx_xvrepl128vei_h(a.v, 7);
   print_case2_imm("xvrepl128vei.h.7", &a, 7, &got);
   check_bytes(tst, "xvrepl128vei.h.7", got.d.u8, exp.d.u8, sizeof(exp.d.u8));

   model_xvinsve0(exp.d.u8, b.d.u8, a.d.u8, 32, 6, sizeof(exp.d.u8));
   got.v = __lasx_xvinsve0_w(b.v, a.v, 6);
   print_case2v_imm256("xvinsve0.w.6", &b, &a, 6, &got);
   check_bytes(tst, "xvinsve0.w.6", got.d.u8, exp.d.u8, sizeof(exp.d.u8));

   model_xvpickve(exp.d.u8, a.d.u8, 32, 5, sizeof(exp.d.u8));
   got.v = __lasx_xvpickve_w(a.v, 5);
   print_case2_imm("xvpickve.w.5", &a, 5, &got);
   check_bytes(tst, "xvpickve.w.5", got.d.u8, exp.d.u8, sizeof(exp.d.u8));

   model_xvreplve0_q(exp.d.u8, a.d.u8);
   got.v = __lasx_xvreplve0_q(a.v);
   print_case1("xvreplve0.q", &a, &got);
   check_bytes(tst, "xvreplve0.q", got.d.u8, exp.d.u8, sizeof(exp.d.u8));
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

   DO_BIN256("xvhaddw.h.b", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 0, 0), __lasx_xvhaddw_h_b(a.v, b.v));
   DO_BIN256("xvhaddw.w.h", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 8, 0, 0), __lasx_xvhaddw_w_h(a.v, b.v));
   DO_BIN256("xvhaddw.d.w", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 4, 0, 0), __lasx_xvhaddw_d_w(a.v, b.v));
   DO_BIN256("xvhaddw.q.d", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 2, 0, 0), __lasx_xvhaddw_q_d(a.v, b.v));
   DO_BIN256("xvhaddw.hu.bu", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 1, 0), __lasx_xvhaddw_hu_bu(a.v, b.v));
   DO_BIN256("xvhaddw.wu.hu", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 8, 1, 0), __lasx_xvhaddw_wu_hu(a.v, b.v));
   DO_BIN256("xvhaddw.du.wu", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 4, 1, 0), __lasx_xvhaddw_du_wu(a.v, b.v));
   DO_BIN256("xvhaddw.qu.du", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 2, 1, 0), __lasx_xvhaddw_qu_du(a.v, b.v));
   DO_BIN256("xvhsubw.h.b", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 0, 1), __lasx_xvhsubw_h_b(a.v, b.v));
   DO_BIN256("xvhsubw.w.h", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 8, 0, 1), __lasx_xvhsubw_w_h(a.v, b.v));
   DO_BIN256("xvhsubw.d.w", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 4, 0, 1), __lasx_xvhsubw_d_w(a.v, b.v));
   DO_BIN256("xvhsubw.q.d", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 2, 0, 1), __lasx_xvhsubw_q_d(a.v, b.v));
   DO_BIN256("xvhsubw.hu.bu", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 1, 1), __lasx_xvhsubw_hu_bu(a.v, b.v));
   DO_BIN256("xvhsubw.wu.hu", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 8, 1, 1), __lasx_xvhsubw_wu_hu(a.v, b.v));
   DO_BIN256("xvhsubw.du.wu", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 4, 1, 1), __lasx_xvhsubw_du_wu(a.v, b.v));
   DO_BIN256("xvhsubw.qu.du", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 2, 1, 1), __lasx_xvhsubw_qu_du(a.v, b.v));
   DO_BIN256("xvaddwev.h.b", model_addw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 0, 1, 1), __lasx_xvaddwev_h_b(a.v, b.v));
   DO_BIN256("xvaddwev.w.h", model_addw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 8, 0, 1, 1), __lasx_xvaddwev_w_h(a.v, b.v));
   DO_BIN256("xvaddwev.d.w", model_addw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 4, 0, 1, 1), __lasx_xvaddwev_d_w(a.v, b.v));
   DO_BIN256("xvaddwev.q.d", model_addw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 2, 0, 1, 1), __lasx_xvaddwev_q_d(a.v, b.v));
   DO_BIN256("xvaddwod.h.b", model_addw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 1, 1, 1), __lasx_xvaddwod_h_b(a.v, b.v));
   DO_BIN256("xvaddwod.w.h", model_addw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 8, 1, 1, 1), __lasx_xvaddwod_w_h(a.v, b.v));
   DO_BIN256("xvaddwod.d.w", model_addw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 4, 1, 1, 1), __lasx_xvaddwod_d_w(a.v, b.v));
   DO_BIN256("xvaddwod.q.d", model_addw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 2, 1, 1, 1), __lasx_xvaddwod_q_d(a.v, b.v));
   DO_BIN256("xvaddwev.h.bu", model_addw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 0, 0, 0), __lasx_xvaddwev_h_bu(a.v, b.v));
   DO_BIN256("xvaddwev.w.hu", model_addw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 8, 0, 0, 0), __lasx_xvaddwev_w_hu(a.v, b.v));
   DO_BIN256("xvaddwev.d.wu", model_addw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 4, 0, 0, 0), __lasx_xvaddwev_d_wu(a.v, b.v));
   DO_BIN256("xvaddwev.q.du", model_addw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 2, 0, 0, 0), __lasx_xvaddwev_q_du(a.v, b.v));
   DO_BIN256("xvaddwod.h.bu", model_addw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 1, 0, 0), __lasx_xvaddwod_h_bu(a.v, b.v));
   DO_BIN256("xvaddwod.w.hu", model_addw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 8, 1, 0, 0), __lasx_xvaddwod_w_hu(a.v, b.v));
   DO_BIN256("xvaddwod.d.wu", model_addw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 4, 1, 0, 0), __lasx_xvaddwod_d_wu(a.v, b.v));
   DO_BIN256("xvaddwod.q.du", model_addw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 2, 1, 0, 0), __lasx_xvaddwod_q_du(a.v, b.v));
   DO_BIN256("xvaddwev.h.bu.b", model_addw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 0, 0, 1), __lasx_xvaddwev_h_bu_b(a.v, b.v));
   DO_BIN256("xvaddwev.w.hu.h", model_addw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 8, 0, 0, 1), __lasx_xvaddwev_w_hu_h(a.v, b.v));
   DO_BIN256("xvaddwev.d.wu.w", model_addw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 4, 0, 0, 1), __lasx_xvaddwev_d_wu_w(a.v, b.v));
   DO_BIN256("xvaddwev.q.du.d", model_addw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 2, 0, 0, 1), __lasx_xvaddwev_q_du_d(a.v, b.v));
   DO_BIN256("xvaddwod.h.bu.b", model_addw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 1, 0, 1), __lasx_xvaddwod_h_bu_b(a.v, b.v));
   DO_BIN256("xvaddwod.w.hu.h", model_addw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 8, 1, 0, 1), __lasx_xvaddwod_w_hu_h(a.v, b.v));
   DO_BIN256("xvaddwod.d.wu.w", model_addw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 4, 1, 0, 1), __lasx_xvaddwod_d_wu_w(a.v, b.v));
   DO_BIN256("xvaddwod.q.du.d", model_addw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 2, 1, 0, 1), __lasx_xvaddwod_q_du_d(a.v, b.v));
   DO_BIN256("xvsubwev.h.b", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 0, 1, 1, 1), __lasx_xvsubwev_h_b(a.v, b.v));
   DO_BIN256("xvsubwev.w.h", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 8, 0, 1, 1, 1), __lasx_xvsubwev_w_h(a.v, b.v));
   DO_BIN256("xvsubwev.d.w", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 4, 0, 1, 1, 1), __lasx_xvsubwev_d_w(a.v, b.v));
   DO_BIN256("xvsubwev.q.d", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 2, 0, 1, 1, 1), __lasx_xvsubwev_q_d(a.v, b.v));
   DO_BIN256("xvsubwod.h.b", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 1, 1, 1, 1), __lasx_xvsubwod_h_b(a.v, b.v));
   DO_BIN256("xvsubwod.w.h", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 8, 1, 1, 1, 1), __lasx_xvsubwod_w_h(a.v, b.v));
   DO_BIN256("xvsubwod.d.w", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 4, 1, 1, 1, 1), __lasx_xvsubwod_d_w(a.v, b.v));
   DO_BIN256("xvsubwod.q.d", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 2, 1, 1, 1, 1), __lasx_xvsubwod_q_d(a.v, b.v));
   DO_BIN256("xvsubwev.h.bu", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 0, 0, 0, 1), __lasx_xvsubwev_h_bu(a.v, b.v));
   DO_BIN256("xvsubwev.w.hu", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 8, 0, 0, 0, 1), __lasx_xvsubwev_w_hu(a.v, b.v));
   DO_BIN256("xvsubwev.d.wu", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 4, 0, 0, 0, 1), __lasx_xvsubwev_d_wu(a.v, b.v));
   DO_BIN256("xvsubwev.q.du", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 2, 0, 0, 0, 1), __lasx_xvsubwev_q_du(a.v, b.v));
   DO_BIN256("xvsubwod.h.bu", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 1, 0, 0, 1), __lasx_xvsubwod_h_bu(a.v, b.v));
   DO_BIN256("xvsubwod.w.hu", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 8, 1, 0, 0, 1), __lasx_xvsubwod_w_hu(a.v, b.v));
   DO_BIN256("xvsubwod.d.wu", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 4, 1, 0, 0, 1), __lasx_xvsubwod_d_wu(a.v, b.v));
   DO_BIN256("xvsubwod.q.du", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 2, 1, 0, 0, 1), __lasx_xvsubwod_q_du(a.v, b.v));
   DO_BIN256("xvmulwev.h.b", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 0, 1, 1), __lasx_xvmulwev_h_b(a.v, b.v));
   DO_BIN256("xvmulwev.w.h", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 8, 0, 1, 1), __lasx_xvmulwev_w_h(a.v, b.v));
   DO_BIN256("xvmulwev.d.w", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 4, 0, 1, 1), __lasx_xvmulwev_d_w(a.v, b.v));
   DO_BIN256("xvmulwev.q.d", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 2, 0, 1, 1), __lasx_xvmulwev_q_d(a.v, b.v));
   DO_BIN256("xvmulwod.h.b", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 1, 1, 1), __lasx_xvmulwod_h_b(a.v, b.v));
   DO_BIN256("xvmulwod.w.h", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 8, 1, 1, 1), __lasx_xvmulwod_w_h(a.v, b.v));
   DO_BIN256("xvmulwod.d.w", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 4, 1, 1, 1), __lasx_xvmulwod_d_w(a.v, b.v));
   DO_BIN256("xvmulwod.q.d", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 2, 1, 1, 1), __lasx_xvmulwod_q_d(a.v, b.v));
   DO_BIN256("xvmulwev.h.bu", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 0, 0, 0), __lasx_xvmulwev_h_bu(a.v, b.v));
   DO_BIN256("xvmulwev.w.hu", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 8, 0, 0, 0), __lasx_xvmulwev_w_hu(a.v, b.v));
   DO_BIN256("xvmulwev.d.wu", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 4, 0, 0, 0), __lasx_xvmulwev_d_wu(a.v, b.v));
   DO_BIN256("xvmulwev.q.du", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 2, 0, 0, 0), __lasx_xvmulwev_q_du(a.v, b.v));
   DO_BIN256("xvmulwod.h.bu", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 1, 0, 0), __lasx_xvmulwod_h_bu(a.v, b.v));
   DO_BIN256("xvmulwod.w.hu", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 8, 1, 0, 0), __lasx_xvmulwod_w_hu(a.v, b.v));
   DO_BIN256("xvmulwod.d.wu", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 4, 1, 0, 0), __lasx_xvmulwod_d_wu(a.v, b.v));
   DO_BIN256("xvmulwod.q.du", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 2, 1, 0, 0), __lasx_xvmulwod_q_du(a.v, b.v));
   DO_BIN256("xvmulwev.h.bu.b", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 0, 0, 1), __lasx_xvmulwev_h_bu_b(a.v, b.v));
   DO_BIN256("xvmulwev.w.hu.h", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 8, 0, 0, 1), __lasx_xvmulwev_w_hu_h(a.v, b.v));
   DO_BIN256("xvmulwev.d.wu.w", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 4, 0, 0, 1), __lasx_xvmulwev_d_wu_w(a.v, b.v));
   DO_BIN256("xvmulwev.q.du.d", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 2, 0, 0, 1), __lasx_xvmulwev_q_du_d(a.v, b.v));
   DO_BIN256("xvmulwod.h.bu.b", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 1, 0, 1), __lasx_xvmulwod_h_bu_b(a.v, b.v));
   DO_BIN256("xvmulwod.w.hu.h", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 8, 1, 0, 1), __lasx_xvmulwod_w_hu_h(a.v, b.v));
   DO_BIN256("xvmulwod.d.wu.w", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 4, 1, 0, 1), __lasx_xvmulwod_d_wu_w(a.v, b.v));
   DO_BIN256("xvmulwod.q.du.d", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 2, 1, 0, 1), __lasx_xvmulwod_q_du_d(a.v, b.v));
   DO_TRI256_ACC("xvmaddwev.h.b", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 0, 1, 1), __lasx_xvmaddwev_h_b(acc.v, a.v, b.v));
   DO_TRI256_ACC("xvmaddwev.w.h", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 16, 32, 8, 0, 1, 1), __lasx_xvmaddwev_w_h(acc.v, a.v, b.v));
   DO_TRI256_ACC("xvmaddwev.d.w", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 32, 64, 4, 0, 1, 1), __lasx_xvmaddwev_d_w(acc.v, a.v, b.v));
   DO_TRI256_ACC("xvmaddwev.q.d", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 64, 128, 2, 0, 1, 1), __lasx_xvmaddwev_q_d(acc.v, a.v, b.v));
   DO_TRI256_ACC("xvmaddwod.h.b", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 1, 1, 1), __lasx_xvmaddwod_h_b(acc.v, a.v, b.v));
   DO_TRI256_ACC("xvmaddwod.w.h", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 16, 32, 8, 1, 1, 1), __lasx_xvmaddwod_w_h(acc.v, a.v, b.v));
   DO_TRI256_ACC("xvmaddwod.d.w", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 32, 64, 4, 1, 1, 1), __lasx_xvmaddwod_d_w(acc.v, a.v, b.v));
   DO_TRI256_ACC("xvmaddwod.q.d", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 64, 128, 2, 1, 1, 1), __lasx_xvmaddwod_q_d(acc.v, a.v, b.v));
   DO_TRI256_ACC("xvmaddwev.h.bu", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 0, 0, 0), __lasx_xvmaddwev_h_bu(acc.v, a.v, b.v));
   DO_TRI256_ACC("xvmaddwev.w.hu", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 16, 32, 8, 0, 0, 0), __lasx_xvmaddwev_w_hu(acc.v, a.v, b.v));
   DO_TRI256_ACC("xvmaddwev.d.wu", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 32, 64, 4, 0, 0, 0), __lasx_xvmaddwev_d_wu(acc.v, a.v, b.v));
   DO_TRI256_ACC("xvmaddwev.q.du", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 64, 128, 2, 0, 0, 0), __lasx_xvmaddwev_q_du(acc.v, a.v, b.v));
   DO_TRI256_ACC("xvmaddwod.h.bu", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 1, 0, 0), __lasx_xvmaddwod_h_bu(acc.v, a.v, b.v));
   DO_TRI256_ACC("xvmaddwod.w.hu", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 16, 32, 8, 1, 0, 0), __lasx_xvmaddwod_w_hu(acc.v, a.v, b.v));
   DO_TRI256_ACC("xvmaddwod.d.wu", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 32, 64, 4, 1, 0, 0), __lasx_xvmaddwod_d_wu(acc.v, a.v, b.v));
   DO_TRI256_ACC("xvmaddwod.q.du", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 64, 128, 2, 1, 0, 0), __lasx_xvmaddwod_q_du(acc.v, a.v, b.v));
   DO_TRI256_ACC("xvmaddwev.h.bu.b", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 0, 0, 1), __lasx_xvmaddwev_h_bu_b(acc.v, a.v, b.v));
   DO_TRI256_ACC("xvmaddwev.w.hu.h", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 16, 32, 8, 0, 0, 1), __lasx_xvmaddwev_w_hu_h(acc.v, a.v, b.v));
   DO_TRI256_ACC("xvmaddwev.d.wu.w", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 32, 64, 4, 0, 0, 1), __lasx_xvmaddwev_d_wu_w(acc.v, a.v, b.v));
   DO_TRI256_ACC("xvmaddwev.q.du.d", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 64, 128, 2, 0, 0, 1), __lasx_xvmaddwev_q_du_d(acc.v, a.v, b.v));
   DO_TRI256_ACC("xvmaddwod.h.bu.b", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 8, 16, 16, 1, 0, 1), __lasx_xvmaddwod_h_bu_b(acc.v, a.v, b.v));
   DO_TRI256_ACC("xvmaddwod.w.hu.h", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 16, 32, 8, 1, 0, 1), __lasx_xvmaddwod_w_hu_h(acc.v, a.v, b.v));
   DO_TRI256_ACC("xvmaddwod.d.wu.w", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 32, 64, 4, 1, 0, 1), __lasx_xvmaddwod_d_wu_w(acc.v, a.v, b.v));
   DO_TRI256_ACC("xvmaddwod.q.du.d", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 64, 128, 2, 1, 0, 1), __lasx_xvmaddwod_q_du_d(acc.v, a.v, b.v));
}

static void test_bsll_bsrl(test_state* tst)
{
   vec256 a = {.d.u8 = {
       0x81, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78,
       0x89, 0x9a, 0xab, 0xbc, 0xcd, 0xde, 0xef, 0x10,
       0x20, 0x31, 0x42, 0x53, 0x64, 0x75, 0x86, 0x97,
       0xa8, 0xb9, 0xca, 0xdb, 0xec, 0xfd, 0x0e, 0x1f}};
   vec256 got, exp;

   DO_IMM256("xvbsll.v.0", 0, model_xvbsll(exp.d.u8, a.d.u8, 0), __lasx_xvbsll_v(a.v, 0));
   DO_IMM256("xvbsll.v.1", 1, model_xvbsll(exp.d.u8, a.d.u8, 1), __lasx_xvbsll_v(a.v, 1));
   DO_IMM256("xvbsll.v.4", 4, model_xvbsll(exp.d.u8, a.d.u8, 4), __lasx_xvbsll_v(a.v, 4));
   DO_IMM256("xvbsll.v.8", 8, model_xvbsll(exp.d.u8, a.d.u8, 8), __lasx_xvbsll_v(a.v, 8));
   DO_IMM256("xvbsll.v.15", 15, model_xvbsll(exp.d.u8, a.d.u8, 15), __lasx_xvbsll_v(a.v, 15));
   DO_IMM256("xvbsll.v.16", 16, model_xvbsll(exp.d.u8, a.d.u8, 16), __lasx_xvbsll_v(a.v, 16));
   DO_IMM256("xvbsrl.v.0", 0, model_xvbsrl(exp.d.u8, a.d.u8, 0), __lasx_xvbsrl_v(a.v, 0));
   DO_IMM256("xvbsrl.v.1", 1, model_xvbsrl(exp.d.u8, a.d.u8, 1), __lasx_xvbsrl_v(a.v, 1));
   DO_IMM256("xvbsrl.v.4", 4, model_xvbsrl(exp.d.u8, a.d.u8, 4), __lasx_xvbsrl_v(a.v, 4));
   DO_IMM256("xvbsrl.v.8", 8, model_xvbsrl(exp.d.u8, a.d.u8, 8), __lasx_xvbsrl_v(a.v, 8));
   DO_IMM256("xvbsrl.v.15", 15, model_xvbsrl(exp.d.u8, a.d.u8, 15), __lasx_xvbsrl_v(a.v, 15));
   DO_IMM256("xvbsrl.v.16", 16, model_xvbsrl(exp.d.u8, a.d.u8, 16), __lasx_xvbsrl_v(a.v, 16));
}

static void test_frstpi(test_state* tst)
{
   vec256 old_vd = {.d.u8 = {
       0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc,
       0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc,
       0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc,
       0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc}};
   vec256 vj;
   vec256 got, exp;

   // xvfrstpi.b, neg at low-byte 3 and high-byte 19 (=3 in hi lane), imm=7
   exp = old_vd;
   for (int i = 0; i < 32; i++) vj.d.u8[i] = 0x30;
   vj.d.u8[3] = 0x80;
   vj.d.u8[19] = 0x80;
   model_xvfrstpi(exp.d.u8, vj.d.u8, 8, 16, 7);
   got.v = __lasx_xvfrstpi_b(old_vd.v, vj.v, 7);
   printf("insn xvfrstpi.b.lane_indep:\n");
   printf("  v_old_vd = "); print_u64x4(old_vd.d.u64);
   printf("  v_vj     = "); print_u64x4(vj.d.u64);
   printf("  imm      = 7\n");
   printf("  v_result = "); print_u64x4(got.d.u64);
   check_bytes(tst, "xvfrstpi.b.lane_indep", got.d.u8, exp.d.u8, 32);

   // xvfrstpi.b, no negative, imm=3 -> write 16 to byte 3 and byte 19
   exp = old_vd;
   for (int i = 0; i < 32; i++) vj.d.u8[i] = 0x30;
   model_xvfrstpi(exp.d.u8, vj.d.u8, 8, 16, 3);
   got.v = __lasx_xvfrstpi_b(old_vd.v, vj.v, 3);
   printf("insn xvfrstpi.b.no_neg:\n");
   printf("  v_old_vd = "); print_u64x4(old_vd.d.u64);
   printf("  v_vj     = "); print_u64x4(vj.d.u64);
   printf("  imm      = 3\n");
   printf("  v_result = "); print_u64x4(got.d.u64);
   check_bytes(tst, "xvfrstpi.b.no_neg", got.d.u8, exp.d.u8, 32);

   // xvfrstpi.h, neg at hword 2 in low, hword 10 (=2) in hi, imm=4
   exp = old_vd;
   for (int i = 0; i < 32; i++) vj.d.u8[i] = 0x30;
   vj.d.u16[2] = 0x8000;
   vj.d.u16[10] = 0x8000;
   model_xvfrstpi(exp.d.u8, vj.d.u8, 16, 8, 4);
   got.v = __lasx_xvfrstpi_h(old_vd.v, vj.v, 4);
   printf("insn xvfrstpi.h.lane_indep:\n");
   printf("  v_old_vd = "); print_u64x4(old_vd.d.u64);
   printf("  v_vj     = "); print_u64x4(vj.d.u64);
   printf("  imm      = 4\n");
   printf("  v_result = "); print_u64x4(got.d.u64);
   check_bytes(tst, "xvfrstpi.h.lane_indep", got.d.u8, exp.d.u8, 32);

   // xvfrstpi.h, no negative, imm=5 -> write 8 to hword 5 and hword 13
   exp = old_vd;
   for (int i = 0; i < 32; i++) vj.d.u8[i] = 0x30;
   model_xvfrstpi(exp.d.u8, vj.d.u8, 16, 8, 5);
   got.v = __lasx_xvfrstpi_h(old_vd.v, vj.v, 5);
   printf("insn xvfrstpi.h.no_neg:\n");
   printf("  v_old_vd = "); print_u64x4(old_vd.d.u64);
   printf("  v_vj     = "); print_u64x4(vj.d.u64);
   printf("  imm      = 5\n");
   printf("  v_result = "); print_u64x4(got.d.u64);
   check_bytes(tst, "xvfrstpi.h.no_neg", got.d.u8, exp.d.u8, 32);
}

static void test_xvldi(test_state* tst)
{
   vec256 got, exp;

   exp.d.u64[0] = 0x0000000000000000ULL;
   exp.d.u64[1] = 0x0000000000000000ULL;
   exp.d.u64[2] = 0x0000000000000000ULL;
   exp.d.u64[3] = 0x0000000000000000ULL;
   got.v = __lasx_xvldi(0x000);
   printf("insn xvldi.repli.b.zero:\n");
   printf("  imm      = 0x000\n");
   printf("  v_result = "); print_u64x4(got.d.u64);
   check_bytes(tst, "xvldi.repli.b.zero", got.d.u8, exp.d.u8, 32);

   exp.d.u64[0] = 0x0101010101010101ULL;
   exp.d.u64[1] = 0x0101010101010101ULL;
   exp.d.u64[2] = 0x0101010101010101ULL;
   exp.d.u64[3] = 0x0101010101010101ULL;
   got.v = __lasx_xvldi(0x001);
   printf("insn xvldi.repli.b.0x01:\n");
   printf("  imm      = 0x001\n");
   printf("  v_result = "); print_u64x4(got.d.u64);
   check_bytes(tst, "xvldi.repli.b.0x01", got.d.u8, exp.d.u8, 32);

   exp.d.u64[0] = 0xffffffffffffffffULL;
   exp.d.u64[1] = 0xffffffffffffffffULL;
   exp.d.u64[2] = 0xffffffffffffffffULL;
   exp.d.u64[3] = 0xffffffffffffffffULL;
   got.v = __lasx_xvldi(0x0ff);
   printf("insn xvldi.repli.b.0xff:\n");
   printf("  imm      = 0x0ff\n");
   printf("  v_result = "); print_u64x4(got.d.u64);
   check_bytes(tst, "xvldi.repli.b.0xff", got.d.u8, exp.d.u8, 32);

   exp.d.u64[0] = 0x0001000100010001ULL;
   exp.d.u64[1] = 0x0001000100010001ULL;
   exp.d.u64[2] = 0x0001000100010001ULL;
   exp.d.u64[3] = 0x0001000100010001ULL;
   got.v = __lasx_xvldi(0x401);
   printf("insn xvldi.repli.h.0x0001:\n");
   printf("  imm      = 0x401\n");
   printf("  v_result = "); print_u64x4(got.d.u64);
   check_bytes(tst, "xvldi.repli.h.0x0001", got.d.u8, exp.d.u8, 32);

   exp.d.u64[0] = 0xfe00fe00fe00fe00ULL;
   exp.d.u64[1] = 0xfe00fe00fe00fe00ULL;
   exp.d.u64[2] = 0xfe00fe00fe00fe00ULL;
   exp.d.u64[3] = 0xfe00fe00fe00fe00ULL;
   got.v = __lasx_xvldi(0x600);
   printf("insn xvldi.repli.h.neg:\n");
   printf("  imm      = 0x600\n");
   printf("  v_result = "); print_u64x4(got.d.u64);
   check_bytes(tst, "xvldi.repli.h.neg", got.d.u8, exp.d.u8, 32);

   exp.d.u64[0] = 0x0000000100000001ULL;
   exp.d.u64[1] = 0x0000000100000001ULL;
   exp.d.u64[2] = 0x0000000100000001ULL;
   exp.d.u64[3] = 0x0000000100000001ULL;
   got.v = __lasx_xvldi(0x801);
   printf("insn xvldi.repli.w.0x0001:\n");
   printf("  imm      = 0x801\n");
   printf("  v_result = "); print_u64x4(got.d.u64);
   check_bytes(tst, "xvldi.repli.w.0x0001", got.d.u8, exp.d.u8, 32);

   exp.d.u64[0] = 0xfffffffefffffffeULL;
   exp.d.u64[1] = 0xfffffffefffffffeULL;
   exp.d.u64[2] = 0xfffffffefffffffeULL;
   exp.d.u64[3] = 0xfffffffefffffffeULL;
   got.v = __lasx_xvldi(0xbfe);
   printf("insn xvldi.repli.w.neg:\n");
   printf("  imm      = 0xbfe\n");
   printf("  v_result = "); print_u64x4(got.d.u64);
   check_bytes(tst, "xvldi.repli.w.neg", got.d.u8, exp.d.u8, 32);

   exp.d.u64[0] = 0x0000000000000001ULL;
   exp.d.u64[1] = 0x0000000000000001ULL;
   exp.d.u64[2] = 0x0000000000000001ULL;
   exp.d.u64[3] = 0x0000000000000001ULL;
   got.v = __lasx_xvldi(0xc01);
   printf("insn xvldi.repli.d.0x0001:\n");
   printf("  imm      = 0xc01\n");
   printf("  v_result = "); print_u64x4(got.d.u64);
   check_bytes(tst, "xvldi.repli.d.0x0001", got.d.u8, exp.d.u8, 32);

   exp.d.u64[0] = 0xfffffffffffffffeULL;
   exp.d.u64[1] = 0xfffffffffffffffeULL;
   exp.d.u64[2] = 0xfffffffffffffffeULL;
   exp.d.u64[3] = 0xfffffffffffffffeULL;
   got.v = __lasx_xvldi(0xffe);
   printf("insn xvldi.repli.d.neg:\n");
   printf("  imm      = 0xffe\n");
   printf("  v_result = "); print_u64x4(got.d.u64);
   check_bytes(tst, "xvldi.repli.d.neg", got.d.u8, exp.d.u8, 32);
}

static void test_divmod(test_state* tst)
{
   vec256 a = {.d.u8 = {
      0x81, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78,
      0x89, 0x9a, 0xab, 0xbc, 0xcd, 0xde, 0xef, 0x10,
      0x20, 0x31, 0x42, 0x53, 0x64, 0x75, 0x86, 0x97,
      0xa8, 0xb9, 0xca, 0xdb, 0xec, 0xfd, 0x0e, 0x1f}};
   vec256 b = {.d.u8 = {
      3, 5, 7, 9, 11, 13, 15, 17,
      19, 21, 23, 25, 27, 29, 31, 33,
      35, 37, 39, 41, 43, 45, 47, 49,
      51, 53, 55, 57, 59, 61, 63, 65}};
   vec256 z = {.d.u8 = {
      0, 5, 0, 9, 0, 13, 0, 17,
      0, 21, 0, 25, 0, 29, 0, 33,
      0, 37, 0, 41, 0, 45, 0, 49,
      0, 53, 0, 57, 0, 61, 0, 65}};
   vec256 neg_a = {.d.i64 = {-1000, 999, -7777777, 123456789}};
   vec256 neg_b = {.d.i64 = {7, -9, -13, 17}};
   vec256 got, exp;

   DO_BIN256("xvdiv.b", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 1, 0), __lasx_xvdiv_b(a.v, b.v));
   DO_BIN256("xvdiv.h", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 1, 0), __lasx_xvdiv_h(a.v, b.v));
   DO_BIN256("xvdiv.w", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 1, 0), __lasx_xvdiv_w(a.v, b.v));
   DO_BIN256("xvdiv.d", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 1, 0), __lasx_xvdiv_d(a.v, b.v));
   DO_BIN256("xvdiv.bu", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 0, 0), __lasx_xvdiv_bu(a.v, b.v));
   DO_BIN256("xvdiv.hu", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 0, 0), __lasx_xvdiv_hu(a.v, b.v));
   DO_BIN256("xvdiv.wu", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 0, 0), __lasx_xvdiv_wu(a.v, b.v));
   DO_BIN256("xvdiv.du", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 0, 0), __lasx_xvdiv_du(a.v, b.v));
   DO_BIN256("xvmod.b", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 1, 1), __lasx_xvmod_b(a.v, b.v));
   DO_BIN256("xvmod.h", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 1, 1), __lasx_xvmod_h(a.v, b.v));
   DO_BIN256("xvmod.w", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 1, 1), __lasx_xvmod_w(a.v, b.v));
   DO_BIN256("xvmod.d", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 1, 1), __lasx_xvmod_d(a.v, b.v));
   DO_BIN256("xvmod.bu", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 0, 1), __lasx_xvmod_bu(a.v, b.v));
   DO_BIN256("xvmod.hu", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 0, 1), __lasx_xvmod_hu(a.v, b.v));
   DO_BIN256("xvmod.wu", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 32, 8, 0, 1), __lasx_xvmod_wu(a.v, b.v));
   DO_BIN256("xvmod.du", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 0, 1), __lasx_xvmod_du(a.v, b.v));

   a = (vec256){.d.u8 = {
      0x81, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78,
      0x89, 0x9a, 0xab, 0xbc, 0xcd, 0xde, 0xef, 0x10,
      0x20, 0x31, 0x42, 0x53, 0x64, 0x75, 0x86, 0x97,
      0xa8, 0xb9, 0xca, 0xdb, 0xec, 0xfd, 0x0e, 0x1f}};
   b = z;
   DO_BIN256("xvdiv.h.zero", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 16, 16, 1, 0), __lasx_xvdiv_h(a.v, b.v));
   DO_BIN256("xvmod.bu.zero", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 8, 32, 0, 1), __lasx_xvmod_bu(a.v, b.v));

   a = neg_a;
   b = neg_b;
   DO_BIN256("xvdiv.d.neg", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 1, 0), __lasx_xvdiv_d(a.v, b.v));
   DO_BIN256("xvmod.d.neg", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 64, 4, 1, 1), __lasx_xvmod_d(a.v, b.v));
}

int main(void)
{
   test_state tst = {0, 0};

   test_basic(&tst);
   test_sat(&tst);
   test_cmpmask(&tst);
   test_vset(&tst);
   test_exth(&tst);
   test_unary_more(&tst);
   test_vshuf(&tst);
   test_shift(&tst);
   test_shift_round(&tst);
   test_shift_narrow(&tst);
   test_shift_narrow_round(&tst);
   test_rotate(&tst);
   test_logic_bit(&tst);
   test_lane_move(&tst);
   test_lane_scalar(&tst);
   test_widen(&tst);
   test_bsll_bsrl(&tst);
   test_frstpi(&tst);
   test_xvldi(&tst);
   test_divmod(&tst);

   return tst.fails != 0;
}
