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

static void print_case_vs_ri(const char* name, const vec128* a, uint64_t r,
                             uint64_t imm)
{
   printf("insn %s:\n", name);
   printf("  v_arg1   = ");
   print_u64x2(a->d.u64);
   printf("  r_arg2   = %016llx\n", (unsigned long long)r);
   printf("  imm      = %016llx\n", (unsigned long long)imm);
}

static void print_case_pick128(const char* name, const vec128* a, uint64_t imm,
                               uint64_t got)
{
   printf("insn %s:\n", name);
   printf("  v_arg1   = ");
   print_u64x2(a->d.u64);
   printf("  imm      = %016llx\n", (unsigned long long)imm);
   printf("  r_result = %016llx\n", (unsigned long long)got);
}

static void print_case_rv128(const char* name, uint64_t r, const vec128* got)
{
   printf("insn %s:\n", name);
   printf("  r_arg1   = %016llx\n", (unsigned long long)r);
   printf("  v_result = ");
   print_u64x2(got->d.u64);
}

static void print_case_set128(const char* name, const vec128* a, uint64_t got)
{
   printf("insn %s:\n", name);
   printf("  v_arg1   = ");
   print_u64x2(a->d.u64);
   printf("  r_result = %016llx\n", (unsigned long long)got);
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

#define DO_INS128(name, immv, rval, model_stmt, expr) \
   do { \
      model_stmt; \
      got.v = (expr); \
      print_case_vs_ri((name), &a, (rval), (immv)); \
      printf("  v_result = "); \
      print_u64x2(got.d.u64); \
      check_bytes(tst, (name), got.d.u8, exp.d.u8, sizeof(exp.d.u8)); \
   } while (0)

#define DO_PICK128(name, immv, model_expr, expr) \
   do { \
      uint64_t got_r = (uint64_t)(expr); \
      uint64_t exp_r = (model_expr); \
      print_case_pick128((name), &a, (immv), got_r); \
      check_u64(tst, (name), got_r, exp_r); \
   } while (0)

#define DO_RV128(name, rval, model_stmt, expr) \
   do { \
      model_stmt; \
      got.v = (expr); \
      print_case_rv128((name), (rval), &got); \
      check_bytes(tst, (name), got.d.u8, exp.d.u8, sizeof(exp.d.u8)); \
   } while (0)

#define DO_SET128(name, vecv, model_expr, expr) \
   do { \
      uint64_t got_r = (uint64_t)(expr); \
      uint64_t exp_r = (model_expr); \
      print_case_set128((name), &(vecv), got_r); \
      check_u64(tst, (name), got_r, exp_r); \
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
   DO_BIN128("vavg.b", model_avg(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 1), __lsx_vavg_b(a.v, b.v));
   DO_BIN128("vavg.hu", model_avg(exp.d.u8, a.d.u8, b.d.u8, 16, 8, 0), __lsx_vavg_hu(a.v, b.v));
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

static void test_sat(test_state* tst)
{
   vec128 a, b, got, exp;

   a = (vec128){.d.i8 = {-120, -20, -5, -4, -3, -1, 0, 1, 2, 3, 4, 5, 20, 100, 110, 120}};
   b = (vec128){.d.i8 = {-120, -120, -120, -2, -1, 1, 1, 2, 120, 120, 120, 4, 20, 40, 60, 80}};
   DO_BIN128("vsadd.b", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 1, 0), __lsx_vsadd_b(a.v, b.v));
   DO_BIN128("vssub.b", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 1, 1), __lsx_vssub_b(a.v, b.v));
   DO_IMM128("vsat.b", 2, model_sat_imm(exp.d.u8, a.d.u8, 8, 16, 2, 1), __lsx_vsat_b(a.v, 2));

   a = (vec128){.d.u8 = {0, 1, 2, 3, 4, 5, 6, 7, 15, 31, 63, 95, 127, 191, 223, 255}};
   b = (vec128){.d.u8 = {0, 1, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8}};
   DO_BIN128("vsadd.bu", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 0, 0), __lsx_vsadd_bu(a.v, b.v));
   DO_BIN128("vssub.bu", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 0, 1), __lsx_vssub_bu(a.v, b.v));
   DO_IMM128("vsat.bu", 2, model_sat_imm(exp.d.u8, a.d.u8, 8, 16, 2, 0), __lsx_vsat_bu(a.v, 2));

   a = (vec128){.d.i16 = {-2000, -40, -33, -32, -31, 31, 32, 2000}};
   b = (vec128){.d.i16 = {-2000, -70, -1, 1, 31, 32, 2000, 2000}};
   DO_BIN128("vsadd.h", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 16, 8, 1, 0), __lsx_vsadd_h(a.v, b.v));
   DO_BIN128("vssub.h", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 16, 8, 1, 1), __lsx_vssub_h(a.v, b.v));
   DO_IMM128("vsat.h", 5, model_sat_imm(exp.d.u8, a.d.u8, 16, 8, 5, 1), __lsx_vsat_h(a.v, 5));

   a = (vec128){.d.u16 = {0, 1, 31, 32, 33, 255, 511, 60000}};
   b = (vec128){.d.u16 = {0, 1, 32, 33, 34, 1000, 1000, 60000}};
   DO_BIN128("vsadd.hu", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 16, 8, 0, 0), __lsx_vsadd_hu(a.v, b.v));
   DO_BIN128("vssub.hu", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 16, 8, 0, 1), __lsx_vssub_hu(a.v, b.v));
   DO_IMM128("vsat.hu", 5, model_sat_imm(exp.d.u8, a.d.u8, 16, 8, 5, 0), __lsx_vsat_hu(a.v, 5));

   a = (vec128){.d.i32 = {-200000, -600, -513, -512}};
   b = (vec128){.d.i32 = {-200000, -600, 1024, 200000}};
   DO_BIN128("vsadd.w", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 32, 4, 1, 0), __lsx_vsadd_w(a.v, b.v));
   DO_BIN128("vssub.w", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 32, 4, 1, 1), __lsx_vssub_w(a.v, b.v));
   DO_IMM128("vsat.w", 9, model_sat_imm(exp.d.u8, a.d.u8, 32, 4, 9, 1), __lsx_vsat_w(a.v, 9));

   a = (vec128){.d.u32 = {0, 1, 1023, 4000000000u}};
   b = (vec128){.d.u32 = {0, 1024, 2048, 4000000000u}};
   DO_BIN128("vsadd.wu", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 32, 4, 0, 0), __lsx_vsadd_wu(a.v, b.v));
   DO_BIN128("vssub.wu", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 32, 4, 0, 1), __lsx_vssub_wu(a.v, b.v));
   DO_IMM128("vsat.wu", 9, model_sat_imm(exp.d.u8, a.d.u8, 32, 4, 9, 0), __lsx_vsat_wu(a.v, 9));

   a = (vec128){.d.i64 = {-500000000000LL, 500000000000LL}};
   b = (vec128){.d.i64 = {-500000000000LL, 500000000000LL}};
   DO_BIN128("vsadd.d", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 64, 2, 1, 0), __lsx_vsadd_d(a.v, b.v));
   DO_BIN128("vssub.d", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 64, 2, 1, 1), __lsx_vssub_d(a.v, b.v));
   DO_IMM128("vsat.d", 17, model_sat_imm(exp.d.u8, a.d.u8, 64, 2, 17, 1), __lsx_vsat_d(a.v, 17));

   a = (vec128){.d.u64 = {0, UINT64_MAX}};
   b = (vec128){.d.u64 = {1, UINT64_MAX}};
   DO_BIN128("vsadd.du", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 64, 2, 0, 0), __lsx_vsadd_du(a.v, b.v));
   DO_BIN128("vssub.du", model_sat_addsub(exp.d.u8, a.d.u8, b.d.u8, 64, 2, 0, 1), __lsx_vssub_du(a.v, b.v));
   DO_IMM128("vsat.du", 17, model_sat_imm(exp.d.u8, a.d.u8, 64, 2, 17, 0), __lsx_vsat_du(a.v, 17));
}

static void test_cmpmask(test_state* tst)
{
   vec128 a = {.d.i8 = {-2, -1, 0, 1, 2, 3, 4, 5, -8, -7, -6, 6, 7, 8, 9, 10}};
   vec128 b = {.d.i8 = {-3, -1, 1, 1, 1, 4, 0, 5, -9, -7, -5, 5, 8, 7, 9, 11}};
   vec128 got, exp;

   DO_BIN128("vseq.b", model_cmp(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 1, 0), __lsx_vseq_b(a.v, b.v));
   DO_BIN128("vsle.b", model_cmp(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 1, 1), __lsx_vsle_b(a.v, b.v));
   DO_BIN128("vslt.bu", model_cmp(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 0, 2), __lsx_vslt_bu(a.v, b.v));
   DO_IMM128("vseqi.h", -1, model_cmp(exp.d.u8, a.d.u8, (vec128){.d.i16 = {-1, -1, -1, -1, -1, -1, -1, -1}}.d.u8, 16, 8, 1, 0), __lsx_vseqi_h(a.v, -1));
   DO_IMM128("vslei.h", 3, model_cmp(exp.d.u8, a.d.u8, (vec128){.d.i16 = {3, 3, 3, 3, 3, 3, 3, 3}}.d.u8, 16, 8, 1, 1), __lsx_vslei_h(a.v, 3));
   DO_IMM128("vslti.hu", 7, model_cmp(exp.d.u8, a.d.u8, (vec128){.d.u16 = {7, 7, 7, 7, 7, 7, 7, 7}}.d.u8, 16, 8, 0, 2), __lsx_vslti_hu(a.v, 7));
   DO_UN128("vmskltz.b", model_vmsk(exp.d.u8, a.d.u8, 8, sizeof(exp.d.u8), 0), __lsx_vmskltz_b(a.v));
   DO_UN128("vmskgez.b", model_vmsk(exp.d.u8, a.d.u8, 8, sizeof(exp.d.u8), 1), __lsx_vmskgez_b(a.v));
   DO_UN128("vmsknz.b", model_vmsk(exp.d.u8, a.d.u8, 8, sizeof(exp.d.u8), 2), __lsx_vmsknz_b(a.v));
}

static void test_vset(test_state* tst)
{
   vec128 z = {.d.u64 = {0, 0}};
   vec128 bh = {.d.u64 = {
      0x1122334400006677ULL, 0x8899aabbccddeeffULL}};
   vec128 w = {.d.u32 = {1, 0, 2, 3}};
   vec128 d = {.d.u64 = {0, 0x0102030405060708ULL}};
   vec128 nz = {.d.u64 = {
      0x0102030405060708ULL, 0x1112131415161718ULL}};

   DO_SET128("bz.v.z", z, model_vset(z.d.u8, 8, sizeof(z.d.u8), 0), __lsx_bz_v(z.v));
   DO_SET128("bz.v.nz", nz, model_vset(nz.d.u8, 8, sizeof(nz.d.u8), 0), __lsx_bz_v(nz.v));
   DO_SET128("bnz.v.z", z, model_vset(z.d.u8, 8, sizeof(z.d.u8), 1), __lsx_bnz_v(z.v));
   DO_SET128("bnz.v.nz", nz, model_vset(nz.d.u8, 8, sizeof(nz.d.u8), 1), __lsx_bnz_v(nz.v));

   DO_SET128("bz.b.haszero", bh, model_vset(bh.d.u8, 8, 16, 2), __lsx_bz_b(bh.v));
   DO_SET128("bz.b.nz", nz, model_vset(nz.d.u8, 8, 16, 2), __lsx_bz_b(nz.v));
   DO_SET128("bz.h.haszero", bh, model_vset(bh.d.u8, 16, 8, 2), __lsx_bz_h(bh.v));
   DO_SET128("bz.h.nz", nz, model_vset(nz.d.u8, 16, 8, 2), __lsx_bz_h(nz.v));
   DO_SET128("bz.w.haszero", w, model_vset(w.d.u8, 32, 4, 2), __lsx_bz_w(w.v));
   DO_SET128("bz.w.nz", nz, model_vset(nz.d.u8, 32, 4, 2), __lsx_bz_w(nz.v));
   DO_SET128("bz.d.haszero", d, model_vset(d.d.u8, 64, 2, 2), __lsx_bz_d(d.v));
   DO_SET128("bz.d.nz", nz, model_vset(nz.d.u8, 64, 2, 2), __lsx_bz_d(nz.v));

   DO_SET128("bnz.b.haszero", bh, model_vset(bh.d.u8, 8, 16, 3), __lsx_bnz_b(bh.v));
   DO_SET128("bnz.b.nz", nz, model_vset(nz.d.u8, 8, 16, 3), __lsx_bnz_b(nz.v));
   DO_SET128("bnz.h.haszero", bh, model_vset(bh.d.u8, 16, 8, 3), __lsx_bnz_h(bh.v));
   DO_SET128("bnz.h.nz", nz, model_vset(nz.d.u8, 16, 8, 3), __lsx_bnz_h(nz.v));
   DO_SET128("bnz.w.haszero", w, model_vset(w.d.u8, 32, 4, 3), __lsx_bnz_w(w.v));
   DO_SET128("bnz.w.nz", nz, model_vset(nz.d.u8, 32, 4, 3), __lsx_bnz_w(nz.v));
   DO_SET128("bnz.d.haszero", d, model_vset(d.d.u8, 64, 2, 3), __lsx_bnz_d(d.v));
   DO_SET128("bnz.d.nz", nz, model_vset(nz.d.u8, 64, 2, 3), __lsx_bnz_d(nz.v));
}

static void test_exth(test_state* tst)
{
   vec128 a = {.d.u8 = {
      0x81, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78,
      0x89, 0x9a, 0xab, 0xbc, 0xcd, 0xde, 0xef, 0x10}};
   vec128 got, exp;

   DO_UN128("vexth.h.b", model_exth(exp.d.u8, a.d.u8, 8, 16, 8, 0, sizeof(a.d.u8)), __lsx_vexth_h_b(a.v));
   DO_UN128("vexth.w.h", model_exth(exp.d.u8, a.d.u8, 16, 32, 4, 0, sizeof(a.d.u8)), __lsx_vexth_w_h(a.v));
   DO_UN128("vexth.d.w", model_exth(exp.d.u8, a.d.u8, 32, 64, 2, 0, sizeof(a.d.u8)), __lsx_vexth_d_w(a.v));
   DO_UN128("vexth.q.d", model_exth(exp.d.u8, a.d.u8, 64, 128, 1, 0, sizeof(a.d.u8)), __lsx_vexth_q_d(a.v));
   DO_UN128("vexth.hu.bu", model_exth(exp.d.u8, a.d.u8, 8, 16, 8, 1, sizeof(a.d.u8)), __lsx_vexth_hu_bu(a.v));
   DO_UN128("vexth.wu.hu", model_exth(exp.d.u8, a.d.u8, 16, 32, 4, 1, sizeof(a.d.u8)), __lsx_vexth_wu_hu(a.v));
   DO_UN128("vexth.du.wu", model_exth(exp.d.u8, a.d.u8, 32, 64, 2, 1, sizeof(a.d.u8)), __lsx_vexth_du_wu(a.v));
   DO_UN128("vexth.qu.du", model_exth(exp.d.u8, a.d.u8, 64, 128, 1, 1, sizeof(a.d.u8)), __lsx_vexth_qu_du(a.v));
}

static void test_unary_more(test_state* tst)
{
   vec128 a = {.d.u8 = {
      0x81, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78,
      0x89, 0x9a, 0xab, 0xbc, 0xcd, 0xde, 0xef, 0x10}};
   vec128 b = {.d.u8 = {
      0x11, 0xf2, 0xe3, 0xd4, 0xc5, 0xb6, 0xa7, 0x98,
      0x79, 0x6a, 0x5b, 0x4c, 0x3d, 0x2e, 0x1f, 0xf0}};
   vec128 got, exp;

   DO_UN128("vclo.w", model_clo(exp.d.u8, a.d.u8, 32, 4), __lsx_vclo_w(a.v));
   DO_UN128("vclz.d", model_clz(exp.d.u8, a.d.u8, 64, 2), __lsx_vclz_d(a.v));
   DO_UN128("vpcnt.b", model_pcnt(exp.d.u8, a.d.u8, 8, 16), __lsx_vpcnt_b(a.v));
   DO_UN128("vneg.h", model_neg(exp.d.u8, a.d.u8, 16, 8), __lsx_vneg_h(a.v));
   DO_BIN128("vsigncov.h", model_signcov(exp.d.u8, a.d.u8, b.d.u8, 16, 8), __lsx_vsigncov_h(a.v, b.v));
}

static void test_vshuf(test_state* tst)
{
   vec128 j = {.d.u8 = {
      0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
      0x20, 0x21, 0x22, 0x23, 0x30, 0x31, 0x32, 0x33}};
   vec128 k = {.d.u8 = {
      0x80, 0x81, 0x82, 0x83, 0x90, 0x91, 0x92, 0x93,
      0xa0, 0xa1, 0xa2, 0xa3, 0xb0, 0xb1, 0xb2, 0xb3}};
   vec128 selb = {.d.u8 = {
      0x00, 0x01, 0x08, 0x09, 0x10, 0x11, 0x18, 0x19,
      0x80, 0x81, 0x88, 0x89, 0xc0, 0xc1, 0xc8, 0xc9}};
   // FIXME: Manual says high selector bits for vshuf.h/w/d are disputed
   // across LoongArch variants. Keep tests to portable low-bit selectors.
   vec128 selh = {.d.u16 = {0, 1, 2, 3, 4, 5, 6, 7}};
   vec128 selw = {.d.u32 = {0, 1, 4, 7}};
   vec128 seld = {.d.u64 = {0, 3}};
   vec128 got, exp;

   model_vshuf_b(exp.d.u8, j.d.u8, k.d.u8, selb.d.u8, sizeof(exp.d.u8));
   got.v = __lsx_vshuf_b(j.v, k.v, selb.v);
   print_case3("vshuf.b", &j, &k, &selb, &got);
   check_bytes(tst, "vshuf.b", got.d.u8, exp.d.u8, sizeof(exp.d.u8));

   model_vshuf_hwd_128chunk(exp.d.u8, selh.d.u8, j.d.u8, k.d.u8, 16);
   got.v = __lsx_vshuf_h(selh.v, j.v, k.v);
   print_case3("vshuf.h", &selh, &j, &k, &got);
   check_bytes(tst, "vshuf.h", got.d.u8, exp.d.u8, sizeof(exp.d.u8));

   model_vshuf_hwd_128chunk(exp.d.u8, selw.d.u8, j.d.u8, k.d.u8, 32);
   got.v = __lsx_vshuf_w(selw.v, j.v, k.v);
   print_case3("vshuf.w", &selw, &j, &k, &got);
   check_bytes(tst, "vshuf.w", got.d.u8, exp.d.u8, sizeof(exp.d.u8));

   model_vshuf_hwd_128chunk(exp.d.u8, seld.d.u8, j.d.u8, k.d.u8, 64);
   got.v = __lsx_vshuf_d(seld.v, j.v, k.v);
   print_case3("vshuf.d", &seld, &j, &k, &got);
   check_bytes(tst, "vshuf.d", got.d.u8, exp.d.u8, sizeof(exp.d.u8));
}

static void test_shift(test_state* tst)
{
   vec128 a = {.d.u8 = {
      0x81, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78,
      0x89, 0x9a, 0xab, 0xbc, 0xcd, 0xde, 0xef, 0x10}};
   vec128 zero = {.d.u8 = {0}};
   vec128 mix = {.d.u8 = {
      7, 8, 9, 15, 1, 2, 3, 4,
      5, 6, 7, 8, 9, 10, 11, 12}};
   vec128 got, exp, b;

   DO_IMM128("vslli.b.0", 0, model_shift_imm(exp.d.u8, a.d.u8, 8, 16, 0, 0, 0), __lsx_vslli_b(a.v, 0));
   DO_IMM128("vslli.b.7", 7, model_shift_imm(exp.d.u8, a.d.u8, 8, 16, 7, 0, 0), __lsx_vslli_b(a.v, 7));
   DO_IMM128("vsrli.h.0", 0, model_shift_imm(exp.d.u8, a.d.u8, 16, 8, 0, 1, 0), __lsx_vsrli_h(a.v, 0));
   DO_IMM128("vsrli.h.15", 15, model_shift_imm(exp.d.u8, a.d.u8, 16, 8, 15, 1, 0), __lsx_vsrli_h(a.v, 15));
   DO_IMM128("vsrai.d.0", 0, model_shift_imm(exp.d.u8, a.d.u8, 64, 2, 0, 1, 1), __lsx_vsrai_d(a.v, 0));
   DO_IMM128("vsrai.d.63", 63, model_shift_imm(exp.d.u8, a.d.u8, 64, 2, 63, 1, 1), __lsx_vsrai_d(a.v, 63));

   b = mix;
   DO_BIN128("vsll.b.var", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 0, 0), __lsx_vsll_b(a.v, b.v));
   b = zero;
   DO_BIN128("vsll.b.zero", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 0, 0), __lsx_vsll_b(a.v, b.v));
   b = mix;
   DO_BIN128("vsrl.b.var", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 1, 0), __lsx_vsrl_b(a.v, b.v));
   b = zero;
   DO_BIN128("vsrl.b.zero", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 1, 0), __lsx_vsrl_b(a.v, b.v));
   b = mix;
   DO_BIN128("vsra.b.var", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 1, 1), __lsx_vsra_b(a.v, b.v));
   b = zero;
   DO_BIN128("vsra.b.zero", model_shift_var(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 1, 1), __lsx_vsra_b(a.v, b.v));
}

static void test_shift_round(test_state* tst)
{
   vec128 a = {.d.u8 = {
      0x81, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78,
      0x89, 0x9a, 0xab, 0xbc, 0xcd, 0xde, 0xef, 0x10}};
   vec128 zero = {.d.u8 = {0}};
   vec128 mix = {.d.u8 = {
      7, 8, 9, 15, 1, 2, 3, 4,
      5, 6, 7, 8, 9, 10, 11, 12}};
   vec128 got, exp, b;

   b = mix;
   DO_BIN128("vsrlr.b.var", model_shift_round_var(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 0), __lsx_vsrlr_b(a.v, b.v));
   b = zero;
   DO_BIN128("vsrlr.b.zero", model_shift_round_var(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 0), __lsx_vsrlr_b(a.v, b.v));
   b = mix;
   DO_BIN128("vsrar.b.var", model_shift_round_var(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 1), __lsx_vsrar_b(a.v, b.v));
   b = zero;
   DO_BIN128("vsrar.b.zero", model_shift_round_var(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 1), __lsx_vsrar_b(a.v, b.v));
}

static void test_shift_narrow(test_state* tst)
{
   vec128 a = {.d.u16 = {0x1281, 0x3423, 0x5645, 0x7867, 0x9a89, 0xbcab, 0xdecd, 0x10ef}};
   vec128 zero = {.d.u16 = {0, 0, 0, 0, 0, 0, 0, 0}};
   vec128 mix = {.d.u16 = {0, 1, 8, 15, 0, 1, 8, 15}};
   vec128 got, exp, b;

   b = mix;
   DO_BIN128("vsrln.b.h.var", model_narrow_shift(exp.d.u8, a.d.u8, b.d.u8, 16, 8, 8, 0, sizeof(exp.d.u8)), __lsx_vsrln_b_h(a.v, b.v));
   b = zero;
   DO_BIN128("vsrln.b.h.zero", model_narrow_shift(exp.d.u8, a.d.u8, b.d.u8, 16, 8, 8, 0, sizeof(exp.d.u8)), __lsx_vsrln_b_h(a.v, b.v));
   b = mix;
   DO_BIN128("vsran.b.h.var", model_narrow_shift(exp.d.u8, a.d.u8, b.d.u8, 16, 8, 8, 1, sizeof(exp.d.u8)), __lsx_vsran_b_h(a.v, b.v));
   b = zero;
   DO_BIN128("vsran.b.h.zero", model_narrow_shift(exp.d.u8, a.d.u8, b.d.u8, 16, 8, 8, 1, sizeof(exp.d.u8)), __lsx_vsran_b_h(a.v, b.v));
}

static void test_shift_narrow_round(test_state* tst)
{
   vec128 a = {.d.u16 = {0x1281, 0x3423, 0x5645, 0x7867, 0x9a89, 0xbcab, 0xdecd, 0x10ef}};
   vec128 zero = {.d.u16 = {0, 0, 0, 0, 0, 0, 0, 0}};
   vec128 mix = {.d.u16 = {0, 1, 8, 15, 0, 1, 8, 15}};
   vec128 got, exp, b;

   b = mix;
   DO_BIN128("vsrlrn.b.h.var", model_narrow_shift_round(exp.d.u8, a.d.u8, b.d.u8, 16, 8, 8, 0, sizeof(exp.d.u8)), __lsx_vsrlrn_b_h(a.v, b.v));
   b = zero;
   DO_BIN128("vsrlrn.b.h.zero", model_narrow_shift_round(exp.d.u8, a.d.u8, b.d.u8, 16, 8, 8, 0, sizeof(exp.d.u8)), __lsx_vsrlrn_b_h(a.v, b.v));
   b = mix;
   DO_BIN128("vsrarn.b.h.var", model_narrow_shift_round(exp.d.u8, a.d.u8, b.d.u8, 16, 8, 8, 1, sizeof(exp.d.u8)), __lsx_vsrarn_b_h(a.v, b.v));
   b = zero;
   DO_BIN128("vsrarn.b.h.zero", model_narrow_shift_round(exp.d.u8, a.d.u8, b.d.u8, 16, 8, 8, 1, sizeof(exp.d.u8)), __lsx_vsrarn_b_h(a.v, b.v));
}

static void test_rotate(test_state* tst)
{
   vec128 a = {.d.u8 = {
      0x81, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78,
      0x89, 0x9a, 0xab, 0xbc, 0xcd, 0xde, 0xef, 0x10}};
   vec128 zero = {.d.u8 = {0}};
   vec128 mix = {.d.u8 = {
      7, 8, 9, 15, 1, 2, 3, 4,
      5, 6, 7, 8, 9, 10, 11, 12}};
   vec128 got, exp, b;

   DO_IMM128("vrotri.b.0", 0, model_rotate_imm(exp.d.u8, a.d.u8, 8, 16, 0), __lsx_vrotri_b(a.v, 0));
   DO_IMM128("vrotri.b.7", 7, model_rotate_imm(exp.d.u8, a.d.u8, 8, 16, 7), __lsx_vrotri_b(a.v, 7));
   DO_IMM128("vrotri.d.0", 0, model_rotate_imm(exp.d.u8, a.d.u8, 64, 2, 0), __lsx_vrotri_d(a.v, 0));
   DO_IMM128("vrotri.d.63", 63, model_rotate_imm(exp.d.u8, a.d.u8, 64, 2, 63), __lsx_vrotri_d(a.v, 63));

   b = mix;
   DO_BIN128("vrotr.b.var", model_rotate_var(exp.d.u8, a.d.u8, b.d.u8, 8, 16), __lsx_vrotr_b(a.v, b.v));
   b = zero;
   DO_BIN128("vrotr.b.zero", model_rotate_var(exp.d.u8, a.d.u8, b.d.u8, 8, 16), __lsx_vrotr_b(a.v, b.v));
}

static void test_logic_bit(test_state* tst)
{
   vec128 a = {.d.u8 = {
      0x81, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78,
      0x89, 0x9a, 0xab, 0xbc, 0xcd, 0xde, 0xef, 0x10}};
   vec128 b = {.d.u8 = {
      0x11, 0xf2, 0xe3, 0xd4, 0xc5, 0xb6, 0xa7, 0x98,
      0x79, 0x6a, 0x5b, 0x4c, 0x3d, 0x2e, 0x1f, 0xf0}};
   vec128 c = {.d.u8 = {
      0x00, 0xff, 0x0f, 0xf0, 0x33, 0xcc, 0x55, 0xaa,
      0x01, 0x80, 0x7f, 0xfe, 0x10, 0x08, 0x04, 0x02}};
   vec128 got, exp;

   DO_BIN128("vand.v", model_logic2(exp.d.u8, a.d.u8, b.d.u8, sizeof(exp.d.u8), 0), __lsx_vand_v(a.v, b.v));
   DO_BIN128("vor.v", model_logic2(exp.d.u8, a.d.u8, b.d.u8, sizeof(exp.d.u8), 1), __lsx_vor_v(a.v, b.v));
   DO_BIN128("vxor.v", model_logic2(exp.d.u8, a.d.u8, b.d.u8, sizeof(exp.d.u8), 2), __lsx_vxor_v(a.v, b.v));
   DO_BIN128("vnor.v", model_logic2(exp.d.u8, a.d.u8, b.d.u8, sizeof(exp.d.u8), 3), __lsx_vnor_v(a.v, b.v));
   DO_BIN128("vandn.v", model_logic2(exp.d.u8, a.d.u8, b.d.u8, sizeof(exp.d.u8), 4), __lsx_vandn_v(a.v, b.v));
   DO_BIN128("vorn.v", model_logic2(exp.d.u8, a.d.u8, b.d.u8, sizeof(exp.d.u8), 5), __lsx_vorn_v(a.v, b.v));
   DO_IMM128("vandi.b", 0x5a, model_logic_imm(exp.d.u8, a.d.u8, sizeof(exp.d.u8), 0x5a, 0), __lsx_vandi_b(a.v, 0x5a));
   DO_IMM128("vori.b", 0xa5, model_logic_imm(exp.d.u8, a.d.u8, sizeof(exp.d.u8), 0xa5, 1), __lsx_vori_b(a.v, 0xa5));
   DO_IMM128("vxori.b", 0xff, model_logic_imm(exp.d.u8, a.d.u8, sizeof(exp.d.u8), 0xff, 2), __lsx_vxori_b(a.v, 0xff));
   DO_IMM128("vnori.b", 0xa5, model_logic_imm(exp.d.u8, a.d.u8, sizeof(exp.d.u8), 0xa5, 3), __lsx_vnori_b(a.v, 0xa5));
   DO_TRI128("vbitsel.v", model_bitsel(exp.d.u8, a.d.u8, b.d.u8, c.d.u8, sizeof(exp.d.u8)), __lsx_vbitsel_v(a.v, b.v, c.v));

   b = (vec128){.d.u8 = {
      0, 7, 8, 15, 1, 2, 3, 4,
      5, 6, 7, 8, 9, 10, 11, 12}};
   DO_BIN128("vbitclr.b", model_bitop_var(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 1), __lsx_vbitclr_b(a.v, b.v));
   b = (vec128){.d.u16 = {0, 15, 16, 31, 1, 2, 3, 4}};
   DO_BIN128("vbitset.h", model_bitop_var(exp.d.u8, a.d.u8, b.d.u8, 16, 8, 2), __lsx_vbitset_h(a.v, b.v));
   b = (vec128){.d.u32 = {0, 31, 32, 33}};
   DO_BIN128("vbitrev.w", model_bitop_var(exp.d.u8, a.d.u8, b.d.u8, 32, 4, 0), __lsx_vbitrev_w(a.v, b.v));
   DO_IMM128("vbitclri.b.0", 0, model_bitop_imm(exp.d.u8, a.d.u8, 8, 16, 0, 0), __lsx_vbitclri_b(a.v, 0));
   DO_IMM128("vbitclri.b.7", 7, model_bitop_imm(exp.d.u8, a.d.u8, 8, 16, 7, 0), __lsx_vbitclri_b(a.v, 7));
   DO_IMM128("vbitseti.h.0", 0, model_bitop_imm(exp.d.u8, a.d.u8, 16, 8, 0, 1), __lsx_vbitseti_h(a.v, 0));
   DO_IMM128("vbitseti.h.15", 15, model_bitop_imm(exp.d.u8, a.d.u8, 16, 8, 15, 1), __lsx_vbitseti_h(a.v, 15));
   DO_IMM128("vbitrevi.d.0", 0, model_bitop_imm(exp.d.u8, a.d.u8, 64, 2, 0, 2), __lsx_vbitrevi_d(a.v, 0));
   DO_IMM128("vbitrevi.d.63", 63, model_bitop_imm(exp.d.u8, a.d.u8, 64, 2, 63, 2), __lsx_vbitrevi_d(a.v, 63));
}

static void test_lane_move(test_state* tst)
{
   vec128 a = {.d.u8 = {
      0x81, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78,
      0x89, 0x9a, 0xab, 0xbc, 0xcd, 0xde, 0xef, 0x10}};
   vec128 b = {.d.u8 = {
      0x11, 0xf2, 0xe3, 0xd4, 0xc5, 0xb6, 0xa7, 0x98,
      0x79, 0x6a, 0x5b, 0x4c, 0x3d, 0x2e, 0x1f, 0xf0}};
   vec128 got, exp;

   DO_BIN128("vilvl.b", model_lane_mix_128chunk(exp.d.u8, a.d.u8, b.d.u8, 8, sizeof(exp.d.u8), 0), __lsx_vilvl_b(a.v, b.v));
   DO_BIN128("vilvh.h", model_lane_mix_128chunk(exp.d.u8, a.d.u8, b.d.u8, 16, sizeof(exp.d.u8), 1), __lsx_vilvh_h(a.v, b.v));
   DO_BIN128("vpickev.w", model_lane_mix_128chunk(exp.d.u8, a.d.u8, b.d.u8, 32, sizeof(exp.d.u8), 2), __lsx_vpickev_w(a.v, b.v));
   DO_BIN128("vpickod.d", model_lane_mix_128chunk(exp.d.u8, a.d.u8, b.d.u8, 64, sizeof(exp.d.u8), 3), __lsx_vpickod_d(a.v, b.v));
   DO_IMM128("vshuf4i.b.0", 0x00, model_shuf4i_128chunk(exp.d.u8, a.d.u8, 8, sizeof(exp.d.u8), 0x00), __lsx_vshuf4i_b(a.v, 0x00));
   DO_IMM128("vshuf4i.b.e4", 0xe4, model_shuf4i_128chunk(exp.d.u8, a.d.u8, 8, sizeof(exp.d.u8), 0xe4), __lsx_vshuf4i_b(a.v, 0xe4));
   DO_IMM128("vshuf4i.w.1b", 0x1b, model_shuf4i_128chunk(exp.d.u8, a.d.u8, 32, sizeof(exp.d.u8), 0x1b), __lsx_vshuf4i_w(a.v, 0x1b));
   DO_BIN128("vpermi.w.1b", model_permi_w_128chunk(exp.d.u8, a.d.u8, b.d.u8, sizeof(exp.d.u8), 0x1b), __lsx_vpermi_w(a.v, b.v, 0x1b));
   DO_BIN128("vextrins.b", model_extrins_b_128chunk(exp.d.u8, a.d.u8, b.d.u8, sizeof(exp.d.u8), 0x4f), __lsx_vextrins_b(a.v, b.v, 0x4f));
}

static void test_lane_scalar(test_state* tst)
{
   vec128 a = {.d.u8 = {
      0x80, 0x7f, 0x01, 0xfe, 0x10, 0x20, 0x30, 0x40,
      0x50, 0x60, 0x70, 0x81, 0x90, 0xa0, 0xb0, 0xc0}};
   vec128 got, exp;

   DO_INS128("vinsgr2vr.b.0", 0, UINT64_MAX,
             model_insgr2vr(exp.d.u8, a.d.u8, 8, 0, UINT64_MAX, sizeof(exp.d.u8)),
             __lsx_vinsgr2vr_b(a.v, -1, 0));
   DO_INS128("vinsgr2vr.w.3", 3, 0xdeadbeefU,
             model_insgr2vr(exp.d.u8, a.d.u8, 32, 3, 0xdeadbeefU, sizeof(exp.d.u8)),
             __lsx_vinsgr2vr_w(a.v, 0xdeadbeefU, 3));
   DO_INS128("vinsgr2vr.d.1", 1, 0x0123456789abcdefULL,
             model_insgr2vr(exp.d.u8, a.d.u8, 64, 1, 0x0123456789abcdefULL, sizeof(exp.d.u8)),
             __lsx_vinsgr2vr_d(a.v, 0x0123456789abcdefULL, 1));

   DO_PICK128("vpickve2gr.b.0", 0, model_pickve2gr(a.d.u8, 8, 0, 1),
              __lsx_vpickve2gr_b(a.v, 0));
   DO_PICK128("vpickve2gr.bu.0", 0, model_pickve2gr(a.d.u8, 8, 0, 0),
              __lsx_vpickve2gr_bu(a.v, 0));
   DO_PICK128("vpickve2gr.w.2", 2, model_pickve2gr(a.d.u8, 32, 2, 1),
              __lsx_vpickve2gr_w(a.v, 2));
   DO_PICK128("vpickve2gr.du.1", 1, model_pickve2gr(a.d.u8, 64, 1, 0),
              __lsx_vpickve2gr_du(a.v, 1));

   DO_RV128("vreplgr2vr.h", 0x1234,
            model_replgr2vr(exp.d.u8, 16, sizeof(exp.d.u8), 0x1234),
            __lsx_vreplgr2vr_h(0x1234));
   DO_RV128("vreplgr2vr.d", 0x0123456789abcdefULL,
            model_replgr2vr(exp.d.u8, 64, sizeof(exp.d.u8), 0x0123456789abcdefULL),
            __lsx_vreplgr2vr_d(0x0123456789abcdefULL));
   model_replve(exp.d.u8, a.d.u8, 32, sizeof(exp.d.u8), 3);
   got.v = __lsx_vreplve_w(a.v, 3);
   print_case2_imm("vreplve.w.3", &a, 3, &got);
   check_bytes(tst, "vreplve.w.3", got.d.u8, exp.d.u8, sizeof(exp.d.u8));
   model_replve(exp.d.u8, a.d.u8, 8, sizeof(exp.d.u8), 15);
   got.v = __lsx_vreplvei_b(a.v, 15);
   print_case2_imm("vreplvei.b.15", &a, 15, &got);
   check_bytes(tst, "vreplvei.b.15", got.d.u8, exp.d.u8, sizeof(exp.d.u8));
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

   DO_BIN128("vhaddw.h.b", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 0, 0), __lsx_vhaddw_h_b(a.v, b.v));
   DO_BIN128("vhaddw.w.h", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 4, 0, 0), __lsx_vhaddw_w_h(a.v, b.v));
   DO_BIN128("vhaddw.d.w", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 2, 0, 0), __lsx_vhaddw_d_w(a.v, b.v));
   DO_BIN128("vhaddw.q.d", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 1, 0, 0), __lsx_vhaddw_q_d(a.v, b.v));
   DO_BIN128("vhaddw.hu.bu", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 1, 0), __lsx_vhaddw_hu_bu(a.v, b.v));
   DO_BIN128("vhaddw.wu.hu", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 4, 1, 0), __lsx_vhaddw_wu_hu(a.v, b.v));
   DO_BIN128("vhaddw.du.wu", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 2, 1, 0), __lsx_vhaddw_du_wu(a.v, b.v));
   DO_BIN128("vhaddw.qu.du", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 1, 1, 0), __lsx_vhaddw_qu_du(a.v, b.v));
   DO_BIN128("vhsubw.h.b", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 0, 1), __lsx_vhsubw_h_b(a.v, b.v));
   DO_BIN128("vhsubw.w.h", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 4, 0, 1), __lsx_vhsubw_w_h(a.v, b.v));
   DO_BIN128("vhsubw.d.w", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 2, 0, 1), __lsx_vhsubw_d_w(a.v, b.v));
   DO_BIN128("vhsubw.q.d", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 1, 0, 1), __lsx_vhsubw_q_d(a.v, b.v));
   DO_BIN128("vhsubw.hu.bu", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 1, 1), __lsx_vhsubw_hu_bu(a.v, b.v));
   DO_BIN128("vhsubw.wu.hu", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 4, 1, 1), __lsx_vhsubw_wu_hu(a.v, b.v));
   DO_BIN128("vhsubw.du.wu", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 2, 1, 1), __lsx_vhsubw_du_wu(a.v, b.v));
   DO_BIN128("vhsubw.qu.du", model_half_addsub_widen(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 1, 1, 1), __lsx_vhsubw_qu_du(a.v, b.v));
   DO_BIN128("vaddwev.h.b", model_addw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 0, 1, 1), __lsx_vaddwev_h_b(a.v, b.v));
   DO_BIN128("vaddwev.w.h", model_addw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 4, 0, 1, 1), __lsx_vaddwev_w_h(a.v, b.v));
   DO_BIN128("vaddwev.d.w", model_addw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 2, 0, 1, 1), __lsx_vaddwev_d_w(a.v, b.v));
   DO_BIN128("vaddwev.q.d", model_addw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 1, 0, 1, 1), __lsx_vaddwev_q_d(a.v, b.v));
   DO_BIN128("vaddwod.h.b", model_addw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 1, 1, 1), __lsx_vaddwod_h_b(a.v, b.v));
   DO_BIN128("vaddwod.w.h", model_addw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 4, 1, 1, 1), __lsx_vaddwod_w_h(a.v, b.v));
   DO_BIN128("vaddwod.d.w", model_addw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 2, 1, 1, 1), __lsx_vaddwod_d_w(a.v, b.v));
   DO_BIN128("vaddwod.q.d", model_addw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 1, 1, 1, 1), __lsx_vaddwod_q_d(a.v, b.v));
   DO_BIN128("vaddwev.h.bu", model_addw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 0, 0, 0), __lsx_vaddwev_h_bu(a.v, b.v));
   DO_BIN128("vaddwev.w.hu", model_addw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 4, 0, 0, 0), __lsx_vaddwev_w_hu(a.v, b.v));
   DO_BIN128("vaddwev.d.wu", model_addw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 2, 0, 0, 0), __lsx_vaddwev_d_wu(a.v, b.v));
   DO_BIN128("vaddwev.q.du", model_addw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 1, 0, 0, 0), __lsx_vaddwev_q_du(a.v, b.v));
   DO_BIN128("vaddwod.h.bu", model_addw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 1, 0, 0), __lsx_vaddwod_h_bu(a.v, b.v));
   DO_BIN128("vaddwod.w.hu", model_addw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 4, 1, 0, 0), __lsx_vaddwod_w_hu(a.v, b.v));
   DO_BIN128("vaddwod.d.wu", model_addw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 2, 1, 0, 0), __lsx_vaddwod_d_wu(a.v, b.v));
   DO_BIN128("vaddwod.q.du", model_addw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 1, 1, 0, 0), __lsx_vaddwod_q_du(a.v, b.v));
   DO_BIN128("vaddwev.h.bu.b", model_addw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 0, 0, 1), __lsx_vaddwev_h_bu_b(a.v, b.v));
   DO_BIN128("vaddwev.w.hu.h", model_addw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 4, 0, 0, 1), __lsx_vaddwev_w_hu_h(a.v, b.v));
   DO_BIN128("vaddwev.d.wu.w", model_addw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 2, 0, 0, 1), __lsx_vaddwev_d_wu_w(a.v, b.v));
   DO_BIN128("vaddwev.q.du.d", model_addw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 1, 0, 0, 1), __lsx_vaddwev_q_du_d(a.v, b.v));
   DO_BIN128("vaddwod.h.bu.b", model_addw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 1, 0, 1), __lsx_vaddwod_h_bu_b(a.v, b.v));
   DO_BIN128("vaddwod.w.hu.h", model_addw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 4, 1, 0, 1), __lsx_vaddwod_w_hu_h(a.v, b.v));
   DO_BIN128("vaddwod.d.wu.w", model_addw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 2, 1, 0, 1), __lsx_vaddwod_d_wu_w(a.v, b.v));
   DO_BIN128("vaddwod.q.du.d", model_addw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 1, 1, 0, 1), __lsx_vaddwod_q_du_d(a.v, b.v));
   DO_BIN128("vsubwev.h.b", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 0, 1, 1, 1), __lsx_vsubwev_h_b(a.v, b.v));
   DO_BIN128("vsubwev.w.h", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 4, 0, 1, 1, 1), __lsx_vsubwev_w_h(a.v, b.v));
   DO_BIN128("vsubwev.d.w", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 2, 0, 1, 1, 1), __lsx_vsubwev_d_w(a.v, b.v));
   DO_BIN128("vsubwev.q.d", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 1, 0, 1, 1, 1), __lsx_vsubwev_q_d(a.v, b.v));
   DO_BIN128("vsubwod.h.b", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 1, 1, 1, 1), __lsx_vsubwod_h_b(a.v, b.v));
   DO_BIN128("vsubwod.w.h", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 4, 1, 1, 1, 1), __lsx_vsubwod_w_h(a.v, b.v));
   DO_BIN128("vsubwod.d.w", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 2, 1, 1, 1, 1), __lsx_vsubwod_d_w(a.v, b.v));
   DO_BIN128("vsubwod.q.d", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 1, 1, 1, 1, 1), __lsx_vsubwod_q_d(a.v, b.v));
   DO_BIN128("vsubwev.h.bu", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 0, 0, 0, 1), __lsx_vsubwev_h_bu(a.v, b.v));
   DO_BIN128("vsubwev.w.hu", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 4, 0, 0, 0, 1), __lsx_vsubwev_w_hu(a.v, b.v));
   DO_BIN128("vsubwev.d.wu", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 2, 0, 0, 0, 1), __lsx_vsubwev_d_wu(a.v, b.v));
   DO_BIN128("vsubwev.q.du", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 1, 0, 0, 0, 1), __lsx_vsubwev_q_du(a.v, b.v));
   DO_BIN128("vsubwod.h.bu", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 1, 0, 0, 1), __lsx_vsubwod_h_bu(a.v, b.v));
   DO_BIN128("vsubwod.w.hu", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 4, 1, 0, 0, 1), __lsx_vsubwod_w_hu(a.v, b.v));
   DO_BIN128("vsubwod.d.wu", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 2, 1, 0, 0, 1), __lsx_vsubwod_d_wu(a.v, b.v));
   DO_BIN128("vsubwod.q.du", model_addsubw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 1, 1, 0, 0, 1), __lsx_vsubwod_q_du(a.v, b.v));
   DO_BIN128("vmulwev.h.b", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 0, 1, 1), __lsx_vmulwev_h_b(a.v, b.v));
   DO_BIN128("vmulwev.w.h", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 4, 0, 1, 1), __lsx_vmulwev_w_h(a.v, b.v));
   DO_BIN128("vmulwev.d.w", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 2, 0, 1, 1), __lsx_vmulwev_d_w(a.v, b.v));
   DO_BIN128("vmulwev.q.d", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 1, 0, 1, 1), __lsx_vmulwev_q_d(a.v, b.v));
   DO_BIN128("vmulwod.h.b", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 1, 1, 1), __lsx_vmulwod_h_b(a.v, b.v));
   DO_BIN128("vmulwod.w.h", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 4, 1, 1, 1), __lsx_vmulwod_w_h(a.v, b.v));
   DO_BIN128("vmulwod.d.w", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 2, 1, 1, 1), __lsx_vmulwod_d_w(a.v, b.v));
   DO_BIN128("vmulwod.q.d", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 1, 1, 1, 1), __lsx_vmulwod_q_d(a.v, b.v));
   DO_BIN128("vmulwev.h.bu", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 0, 0, 0), __lsx_vmulwev_h_bu(a.v, b.v));
   DO_BIN128("vmulwev.w.hu", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 4, 0, 0, 0), __lsx_vmulwev_w_hu(a.v, b.v));
   DO_BIN128("vmulwev.d.wu", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 2, 0, 0, 0), __lsx_vmulwev_d_wu(a.v, b.v));
   DO_BIN128("vmulwev.q.du", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 1, 0, 0, 0), __lsx_vmulwev_q_du(a.v, b.v));
   DO_BIN128("vmulwod.h.bu", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 1, 0, 0), __lsx_vmulwod_h_bu(a.v, b.v));
   DO_BIN128("vmulwod.w.hu", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 4, 1, 0, 0), __lsx_vmulwod_w_hu(a.v, b.v));
   DO_BIN128("vmulwod.d.wu", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 2, 1, 0, 0), __lsx_vmulwod_d_wu(a.v, b.v));
   DO_BIN128("vmulwod.q.du", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 1, 1, 0, 0), __lsx_vmulwod_q_du(a.v, b.v));
   DO_BIN128("vmulwev.h.bu.b", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 0, 0, 1), __lsx_vmulwev_h_bu_b(a.v, b.v));
   DO_BIN128("vmulwev.w.hu.h", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 4, 0, 0, 1), __lsx_vmulwev_w_hu_h(a.v, b.v));
   DO_BIN128("vmulwev.d.wu.w", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 2, 0, 0, 1), __lsx_vmulwev_d_wu_w(a.v, b.v));
   DO_BIN128("vmulwev.q.du.d", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 1, 0, 0, 1), __lsx_vmulwev_q_du_d(a.v, b.v));
   DO_BIN128("vmulwod.h.bu.b", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 1, 0, 1), __lsx_vmulwod_h_bu_b(a.v, b.v));
   DO_BIN128("vmulwod.w.hu.h", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 16, 32, 4, 1, 0, 1), __lsx_vmulwod_w_hu_h(a.v, b.v));
   DO_BIN128("vmulwod.d.wu.w", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 32, 64, 2, 1, 0, 1), __lsx_vmulwod_d_wu_w(a.v, b.v));
   DO_BIN128("vmulwod.q.du.d", model_mulw(exp.d.u8, a.d.u8, b.d.u8, 64, 128, 1, 1, 0, 1), __lsx_vmulwod_q_du_d(a.v, b.v));
   DO_TRI128_ACC("vmaddwev.h.b", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 0, 1, 1), __lsx_vmaddwev_h_b(acc.v, a.v, b.v));
   DO_TRI128_ACC("vmaddwev.w.h", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 16, 32, 4, 0, 1, 1), __lsx_vmaddwev_w_h(acc.v, a.v, b.v));
   DO_TRI128_ACC("vmaddwev.d.w", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 32, 64, 2, 0, 1, 1), __lsx_vmaddwev_d_w(acc.v, a.v, b.v));
   DO_TRI128_ACC("vmaddwev.q.d", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 64, 128, 1, 0, 1, 1), __lsx_vmaddwev_q_d(acc.v, a.v, b.v));
   DO_TRI128_ACC("vmaddwod.h.b", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 1, 1, 1), __lsx_vmaddwod_h_b(acc.v, a.v, b.v));
   DO_TRI128_ACC("vmaddwod.w.h", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 16, 32, 4, 1, 1, 1), __lsx_vmaddwod_w_h(acc.v, a.v, b.v));
   DO_TRI128_ACC("vmaddwod.d.w", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 32, 64, 2, 1, 1, 1), __lsx_vmaddwod_d_w(acc.v, a.v, b.v));
   DO_TRI128_ACC("vmaddwod.q.d", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 64, 128, 1, 1, 1, 1), __lsx_vmaddwod_q_d(acc.v, a.v, b.v));
   DO_TRI128_ACC("vmaddwev.h.bu", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 0, 0, 0), __lsx_vmaddwev_h_bu(acc.v, a.v, b.v));
   DO_TRI128_ACC("vmaddwev.w.hu", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 16, 32, 4, 0, 0, 0), __lsx_vmaddwev_w_hu(acc.v, a.v, b.v));
   DO_TRI128_ACC("vmaddwev.d.wu", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 32, 64, 2, 0, 0, 0), __lsx_vmaddwev_d_wu(acc.v, a.v, b.v));
   DO_TRI128_ACC("vmaddwev.q.du", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 64, 128, 1, 0, 0, 0), __lsx_vmaddwev_q_du(acc.v, a.v, b.v));
   DO_TRI128_ACC("vmaddwod.h.bu", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 1, 0, 0), __lsx_vmaddwod_h_bu(acc.v, a.v, b.v));
   DO_TRI128_ACC("vmaddwod.w.hu", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 16, 32, 4, 1, 0, 0), __lsx_vmaddwod_w_hu(acc.v, a.v, b.v));
   DO_TRI128_ACC("vmaddwod.d.wu", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 32, 64, 2, 1, 0, 0), __lsx_vmaddwod_d_wu(acc.v, a.v, b.v));
   DO_TRI128_ACC("vmaddwod.q.du", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 64, 128, 1, 1, 0, 0), __lsx_vmaddwod_q_du(acc.v, a.v, b.v));
   DO_TRI128_ACC("vmaddwev.h.bu.b", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 0, 0, 1), __lsx_vmaddwev_h_bu_b(acc.v, a.v, b.v));
   DO_TRI128_ACC("vmaddwev.w.hu.h", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 16, 32, 4, 0, 0, 1), __lsx_vmaddwev_w_hu_h(acc.v, a.v, b.v));
   DO_TRI128_ACC("vmaddwev.d.wu.w", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 32, 64, 2, 0, 0, 1), __lsx_vmaddwev_d_wu_w(acc.v, a.v, b.v));
   DO_TRI128_ACC("vmaddwev.q.du.d", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 64, 128, 1, 0, 0, 1), __lsx_vmaddwev_q_du_d(acc.v, a.v, b.v));
   DO_TRI128_ACC("vmaddwod.h.bu.b", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 8, 16, 8, 1, 0, 1), __lsx_vmaddwod_h_bu_b(acc.v, a.v, b.v));
   DO_TRI128_ACC("vmaddwod.w.hu.h", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 16, 32, 4, 1, 0, 1), __lsx_vmaddwod_w_hu_h(acc.v, a.v, b.v));
   DO_TRI128_ACC("vmaddwod.d.wu.w", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 32, 64, 2, 1, 0, 1), __lsx_vmaddwod_d_wu_w(acc.v, a.v, b.v));
   DO_TRI128_ACC("vmaddwod.q.du.d", model_maddw(exp.d.u8, acc.d.u8, a.d.u8, b.d.u8, 64, 128, 1, 1, 0, 1), __lsx_vmaddwod_q_du_d(acc.v, a.v, b.v));
}

static void test_divmod(test_state* tst)
{
   vec128 a = {.d.u8 = {
      0x81, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78,
      0x89, 0x9a, 0xab, 0xbc, 0xcd, 0xde, 0xef, 0x10}};
   vec128 b = {.d.u8 = {
      3, 5, 7, 9, 11, 13, 15, 17,
      19, 21, 23, 25, 27, 29, 31, 33}};
   vec128 z = {.d.u8 = {
      0, 5, 0, 9, 0, 13, 0, 17,
      0, 21, 0, 25, 0, 29, 0, 33}};
   vec128 neg_a = {.d.i64 = {-1000, 999}};
   vec128 neg_b = {.d.i64 = {7, -9}};
   vec128 got, exp;

   DO_BIN128("vdiv.h", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 16, 8, 1, 0), __lsx_vdiv_h(a.v, b.v));
   DO_BIN128("vdiv.w", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 32, 4, 1, 0), __lsx_vdiv_w(a.v, b.v));
   DO_BIN128("vdiv.d", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 64, 2, 1, 0), __lsx_vdiv_d(a.v, b.v));
   DO_BIN128("vdiv.bu", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 0, 0), __lsx_vdiv_bu(a.v, b.v));
   DO_BIN128("vdiv.hu", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 16, 8, 0, 0), __lsx_vdiv_hu(a.v, b.v));
   DO_BIN128("vdiv.wu", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 32, 4, 0, 0), __lsx_vdiv_wu(a.v, b.v));
   DO_BIN128("vdiv.du", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 64, 2, 0, 0), __lsx_vdiv_du(a.v, b.v));
   DO_BIN128("vmod.b", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 1, 1), __lsx_vmod_b(a.v, b.v));
   DO_BIN128("vmod.h", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 16, 8, 1, 1), __lsx_vmod_h(a.v, b.v));
   DO_BIN128("vmod.w", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 32, 4, 1, 1), __lsx_vmod_w(a.v, b.v));
   DO_BIN128("vmod.d", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 64, 2, 1, 1), __lsx_vmod_d(a.v, b.v));
   DO_BIN128("vmod.bu", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 0, 1), __lsx_vmod_bu(a.v, b.v));
   DO_BIN128("vmod.hu", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 16, 8, 0, 1), __lsx_vmod_hu(a.v, b.v));
   DO_BIN128("vmod.wu", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 32, 4, 0, 1), __lsx_vmod_wu(a.v, b.v));
   DO_BIN128("vmod.du", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 64, 2, 0, 1), __lsx_vmod_du(a.v, b.v));

   a = (vec128){.d.u8 = {
      0x81, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78,
      0x89, 0x9a, 0xab, 0xbc, 0xcd, 0xde, 0xef, 0x10}};
   b = z;
   DO_BIN128("vdiv.b.zero", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 8, 16, 1, 0), __lsx_vdiv_b(a.v, b.v));
   DO_BIN128("vmod.h.zero", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 16, 8, 1, 1), __lsx_vmod_h(a.v, b.v));

   a = neg_a;
   b = neg_b;
   DO_BIN128("vdiv.d.neg", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 64, 2, 1, 0), __lsx_vdiv_d(a.v, b.v));
   DO_BIN128("vmod.d.neg", model_divmod(exp.d.u8, a.d.u8, b.d.u8, 64, 2, 1, 1), __lsx_vmod_d(a.v, b.v));
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
   test_divmod(&tst);

   return tst.fails != 0;
}
