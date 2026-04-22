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
   DO_BIN256("xvsigncov.w", model_signcov(exp.d.u8, a.d.u8, b.d.u8, 32, 8), __lasx_xvsigncov_w(a.v, b.v));
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
   test_widen(&tst);
   test_divmod(&tst);

   return tst.fails != 0;
}
