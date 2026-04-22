#ifndef LOONGARCH64_VECTOR_COMMON_H
#define LOONGARCH64_VECTOR_COMMON_H

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef union {
   uint8_t u8[16];
   int8_t i8[16];
   uint16_t u16[8];
   int16_t i16[8];
   uint32_t u32[4];
   int32_t i32[4];
   uint64_t u64[2];
   int64_t i64[2];
   unsigned __int128 u128[1];
} vec128_data;

typedef union {
   uint8_t u8[32];
   int8_t i8[32];
   uint16_t u16[16];
   int16_t i16[16];
   uint32_t u32[8];
   int32_t i32[8];
   uint64_t u64[4];
   int64_t i64[4];
   unsigned __int128 u128[2];
} vec256_data;

typedef struct {
   int checks;
   int fails;
} test_state;

static __attribute__((unused)) void print_u64x2(const uint64_t* v)
{
   printf("%016llx | %016llx\n",
          (unsigned long long)v[1], (unsigned long long)v[0]);
}

static __attribute__((unused)) void print_u64x4(const uint64_t* v)
{
   printf("%016llx | %016llx | %016llx | %016llx\n",
          (unsigned long long)v[3], (unsigned long long)v[2],
          (unsigned long long)v[1], (unsigned long long)v[0]);
}

static uint64_t lane_mask_u64(unsigned bits)
{
   return bits == 64 ? UINT64_MAX : ((UINT64_C(1) << bits) - 1);
}

static uint64_t read_lane_u64(const uint8_t* src, unsigned bits, unsigned idx)
{
   uint64_t val = 0;
   unsigned bytes = bits / 8;
   unsigned i;
   for (i = 0; i < bytes; i++)
      val |= (uint64_t)src[idx * bytes + i] << (8 * i);
   return val & lane_mask_u64(bits);
}

static void write_lane_u64(uint8_t* dst, unsigned bits, unsigned idx, uint64_t val)
{
   unsigned bytes = bits / 8;
   unsigned i;
   val &= lane_mask_u64(bits);
   for (i = 0; i < bytes; i++)
      dst[idx * bytes + i] = (uint8_t)(val >> (8 * i));
}

static unsigned __int128 read_lane_u128(const uint8_t* src, unsigned idx)
{
   unsigned __int128 val = 0;
   unsigned i;
   for (i = 0; i < 16; i++)
      val |= (unsigned __int128)src[idx * 16 + i] << (8 * i);
   return val;
}

static void write_lane_u128(uint8_t* dst, unsigned idx, unsigned __int128 val)
{
   unsigned i;
   for (i = 0; i < 16; i++)
      dst[idx * 16 + i] = (uint8_t)(val >> (8 * i));
}

static void write_lane_wide(uint8_t* dst, unsigned bits, unsigned idx,
                            unsigned __int128 val)
{
   if (bits == 128)
      write_lane_u128(dst, idx, val);
   else
      write_lane_u64(dst, bits, idx, (uint64_t)val);
}

static unsigned __int128 read_lane_wide(const uint8_t* src, unsigned bits,
                                        unsigned idx)
{
   if (bits == 128)
      return read_lane_u128(src, idx);
   return read_lane_u64(src, bits, idx);
}

static int64_t sx64(uint64_t raw, unsigned bits)
{
   uint64_t sign = UINT64_C(1) << (bits - 1);
   uint64_t mask = lane_mask_u64(bits);
   raw &= mask;
   return (int64_t)((raw ^ sign) - sign);
}

static uint64_t ux_from_sx(int64_t val, unsigned bits)
{
   return ((uint64_t)val) & lane_mask_u64(bits);
}

static unsigned count_leading_zeros_u64(uint64_t raw, unsigned bits)
{
   unsigned i;
   for (i = 0; i < bits; i++) {
      unsigned shift = bits - 1 - i;
      if ((raw >> shift) & 1)
         return i;
   }
   return bits;
}

static unsigned count_leading_ones_u64(uint64_t raw, unsigned bits)
{
   unsigned i;
   raw &= lane_mask_u64(bits);
   for (i = 0; i < bits; i++) {
      unsigned shift = bits - 1 - i;
      if (((raw >> shift) & 1) == 0)
         return i;
   }
   return bits;
}

static unsigned popcount_u64(uint64_t raw)
{
   return (unsigned)__builtin_popcountll(raw);
}

static void model_addsub(uint8_t* dst, const uint8_t* a, const uint8_t* b,
                         unsigned bits, unsigned lanes, int is_sub)
{
   unsigned i;
   for (i = 0; i < lanes; i++) {
      uint64_t av = read_lane_u64(a, bits, i);
      uint64_t bv = read_lane_u64(b, bits, i);
      uint64_t rv = is_sub ? (av - bv) : (av + bv);
      write_lane_u64(dst, bits, i, rv);
   }
}

static void model_addi(uint8_t* dst, const uint8_t* a, unsigned bits,
                       unsigned lanes, uint64_t imm, int is_sub)
{
   unsigned i;
   for (i = 0; i < lanes; i++) {
      uint64_t av = read_lane_u64(a, bits, i);
      uint64_t rv = is_sub ? (av - imm) : (av + imm);
      write_lane_u64(dst, bits, i, rv);
   }
}

static void model_q_addsub(uint8_t* dst, const uint8_t* a, const uint8_t* b,
                           unsigned lanes, int is_sub)
{
   unsigned i;
   for (i = 0; i < lanes; i++) {
      unsigned __int128 av = read_lane_u128(a, i);
      unsigned __int128 bv = read_lane_u128(b, i);
      write_lane_u128(dst, i, is_sub ? (av - bv) : (av + bv));
   }
}

static void model_clo(uint8_t* dst, const uint8_t* a, unsigned bits, unsigned lanes)
{
   unsigned i;
   for (i = 0; i < lanes; i++)
      write_lane_u64(dst, bits, i,
                     count_leading_ones_u64(read_lane_u64(a, bits, i), bits));
}

static void model_clz(uint8_t* dst, const uint8_t* a, unsigned bits, unsigned lanes)
{
   unsigned i;
   for (i = 0; i < lanes; i++)
      write_lane_u64(dst, bits, i,
                     count_leading_zeros_u64(read_lane_u64(a, bits, i), bits));
}

static void model_pcnt(uint8_t* dst, const uint8_t* a, unsigned bits, unsigned lanes)
{
   unsigned i;
   for (i = 0; i < lanes; i++)
      write_lane_u64(dst, bits, i,
                     popcount_u64(read_lane_u64(a, bits, i)));
}

static void model_neg(uint8_t* dst, const uint8_t* a, unsigned bits, unsigned lanes)
{
   unsigned i;
   for (i = 0; i < lanes; i++)
      write_lane_u64(dst, bits, i, 0 - read_lane_u64(a, bits, i));
}

static int64_t sat_signed(int64_t val, unsigned bits)
{
   int64_t lo = -(INT64_C(1) << bits);
   int64_t hi = (INT64_C(1) << bits) - 1;
   if (val < lo)
      return lo;
   if (val > hi)
      return hi;
   return val;
}

static uint64_t sat_unsigned(uint64_t val, unsigned bits)
{
   uint64_t hi = bits == 64 ? UINT64_MAX : ((UINT64_C(1) << bits) - 1);
   return val > hi ? hi : val;
}

static void model_sat_addsub(uint8_t* dst, const uint8_t* a, const uint8_t* b,
                             unsigned bits, unsigned lanes, int is_signed,
                             int is_sub)
{
   unsigned i;
   for (i = 0; i < lanes; i++) {
      if (is_signed) {
         int64_t av = sx64(read_lane_u64(a, bits, i), bits);
         int64_t bv = sx64(read_lane_u64(b, bits, i), bits);
         int64_t rv = is_sub ? (av - bv) : (av + bv);
         write_lane_u64(dst, bits, i, ux_from_sx(sat_signed(rv, bits - 1), bits));
      } else {
         uint64_t av = read_lane_u64(a, bits, i);
         uint64_t bv = read_lane_u64(b, bits, i);
         uint64_t rv;
         if (is_sub)
            rv = av < bv ? 0 : (av - bv);
         else if (bits == 64)
            rv = UINT64_MAX - av < bv ? UINT64_MAX : (av + bv);
         else
            rv = sat_unsigned(av + bv, bits);
         write_lane_u64(dst, bits, i, rv);
      }
   }
}

static void model_sat_imm(uint8_t* dst, const uint8_t* a, unsigned bits,
                          unsigned lanes, unsigned imm, int is_signed)
{
   unsigned i;
   for (i = 0; i < lanes; i++) {
      if (is_signed) {
         int64_t av = sx64(read_lane_u64(a, bits, i), bits);
         int64_t rv = imm >= bits - 1 ? av : sat_signed(av, imm);
         write_lane_u64(dst, bits, i, ux_from_sx(rv, bits));
      } else {
         uint64_t av = read_lane_u64(a, bits, i);
         uint64_t rv = imm + 1 >= bits ? av : sat_unsigned(av, imm + 1);
         write_lane_u64(dst, bits, i, rv);
      }
   }
}

static void model_cmp(uint8_t* dst, const uint8_t* a, const uint8_t* b,
                      unsigned bits, unsigned lanes, int is_signed, int mode)
{
   unsigned i;
   uint64_t truev = bits == 64 ? UINT64_MAX : ((UINT64_C(1) << bits) - 1);
   for (i = 0; i < lanes; i++) {
      uint64_t rv = 0;
      if (is_signed) {
         int64_t av = sx64(read_lane_u64(a, bits, i), bits);
         int64_t bv = sx64(read_lane_u64(b, bits, i), bits);
         rv = mode == 0 ? (av == bv) : (mode == 1 ? (av <= bv) : (av < bv));
      } else {
         uint64_t av = read_lane_u64(a, bits, i);
         uint64_t bv = read_lane_u64(b, bits, i);
         rv = mode == 0 ? (av == bv) : (mode == 1 ? (av <= bv) : (av < bv));
      }
      write_lane_u64(dst, bits, i, rv ? truev : 0);
   }
}

static void model_vmsk(uint8_t* dst, const uint8_t* a, unsigned bits,
                       unsigned total_bytes, int mode)
{
   unsigned lanes_per_128 = 128 / bits;
   unsigned chunks = total_bytes / 16;
   memset(dst, 0, total_bytes);
   for (unsigned chunk = 0; chunk < chunks; chunk++) {
      uint64_t mask = 0;
      const uint8_t* base = a + chunk * 16;
      for (unsigned i = 0; i < lanes_per_128; i++) {
         uint64_t raw = read_lane_u64(base, bits, i);
         int set;
         if (mode == 0)
            set = sx64(raw, bits) < 0;
         else if (mode == 1)
            set = sx64(raw, 8) >= 0;
         else
            set = (raw & 0xff) != 0;
         if (set)
            mask |= UINT64_C(1) << i;
      }
      write_lane_u64(dst + chunk * 16, 64, 0, mask);
      write_lane_u64(dst + chunk * 16, 64, 1, 0);
   }
}

static void model_exth(uint8_t* dst, const uint8_t* a, unsigned src_bits,
                       unsigned dst_bits, unsigned dst_lanes, int is_unsigned,
                       unsigned bytes_total)
{
   unsigned src_lanes = (bytes_total * 8) / src_bits;
   unsigned start = src_lanes / 2;
   memset(dst, 0, bytes_total);
   for (unsigned i = 0; i < dst_lanes; i++) {
      uint64_t raw = read_lane_u64(a, src_bits, start + i);
      unsigned __int128 val;
      if (is_unsigned) {
         val = raw;
      } else if (dst_bits == 128) {
         val = (unsigned __int128)((__int128)sx64(raw, src_bits));
      } else {
         val = (unsigned __int128)ux_from_sx(sx64(raw, src_bits), dst_bits);
      }
      write_lane_wide(dst, dst_bits, i, val);
   }
}

static void model_xvexth(uint8_t* dst, const uint8_t* a, unsigned src_bits,
                         unsigned dst_bits, unsigned dst_lanes_per_128,
                         int is_unsigned)
{
   model_exth(dst, a, src_bits, dst_bits, dst_lanes_per_128, is_unsigned, 16);
   model_exth(dst + 16, a + 16, src_bits, dst_bits, dst_lanes_per_128, is_unsigned, 16);
}

static void model_vext2xv(uint8_t* dst, const uint8_t* a, unsigned src_bits,
                          unsigned dst_bits, unsigned dst_lanes, int is_unsigned)
{
   memset(dst, 0, 32);
   for (unsigned i = 0; i < dst_lanes; i++) {
      uint64_t raw = read_lane_u64(a, src_bits, i);
      unsigned __int128 val;
      if (is_unsigned) {
         val = raw;
      } else if (dst_bits == 128) {
         val = (unsigned __int128)((__int128)sx64(raw, src_bits));
      } else {
         val = (unsigned __int128)ux_from_sx(sx64(raw, src_bits), dst_bits);
      }
      write_lane_wide(dst, dst_bits, i, val);
   }
}

static void model_shift_imm(uint8_t* dst, const uint8_t* a, unsigned bits,
                            unsigned lanes, unsigned sh, int is_right,
                            int is_arith)
{
   unsigned i;
   for (i = 0; i < lanes; i++) {
      uint64_t raw = read_lane_u64(a, bits, i);
      uint64_t rv;
      if (!is_right) {
         rv = (raw << sh) & lane_mask_u64(bits);
      } else if (is_arith) {
         rv = ux_from_sx(sx64(raw, bits) >> sh, bits);
      } else {
         rv = raw >> sh;
      }
      write_lane_u64(dst, bits, i, rv);
   }
}

static void model_shift_var(uint8_t* dst, const uint8_t* a, const uint8_t* b,
                            unsigned bits, unsigned lanes, int is_right,
                            int is_arith)
{
   unsigned i;
   for (i = 0; i < lanes; i++) {
      unsigned sh = (unsigned)(read_lane_u64(b, bits, i) % bits);
      uint64_t raw = read_lane_u64(a, bits, i);
      uint64_t rv;
      if (!is_right) {
         rv = (raw << sh) & lane_mask_u64(bits);
      } else if (is_arith) {
         rv = ux_from_sx(sx64(raw, bits) >> sh, bits);
      } else {
         rv = raw >> sh;
      }
      write_lane_u64(dst, bits, i, rv);
   }
}

static void model_shift_round_var(uint8_t* dst, const uint8_t* a, const uint8_t* b,
                                  unsigned bits, unsigned lanes, int is_arith)
{
   unsigned i;
   for (i = 0; i < lanes; i++) {
      unsigned sh = (unsigned)(read_lane_u64(b, bits, i) % bits);
      uint64_t raw = read_lane_u64(a, bits, i);
      uint64_t rv;
      if (sh == 0) {
         rv = raw;
      } else if (is_arith) {
         int64_t base = sx64(raw, bits) >> sh;
         uint64_t round = (raw >> (sh - 1)) & 1;
         rv = ux_from_sx(base + (int64_t)round, bits);
      } else {
         rv = (raw >> sh) + ((raw >> (sh - 1)) & 1);
      }
      write_lane_u64(dst, bits, i, rv);
   }
}

static void model_narrow_shift(uint8_t* dst, const uint8_t* a, const uint8_t* b,
                               unsigned src_bits, unsigned dst_bits,
                               unsigned src_lanes, int is_arith,
                               unsigned total_bytes)
{
   unsigned chunk_bytes = src_lanes * src_bits / 8;
   unsigned chunks = total_bytes / chunk_bytes;
   memset(dst, 0, total_bytes);
   for (unsigned chunk = 0; chunk < chunks; chunk++) {
      const uint8_t* sa = a + chunk * chunk_bytes;
      const uint8_t* sb = b + chunk * chunk_bytes;
      uint8_t* dd = dst + chunk * chunk_bytes;
      for (unsigned i = 0; i < src_lanes; i++) {
         unsigned sh = (unsigned)(read_lane_u64(sb, src_bits, i) % src_bits);
         uint64_t raw = read_lane_u64(sa, src_bits, i);
         uint64_t rv = is_arith ? ux_from_sx(sx64(raw, src_bits) >> sh, src_bits)
                                : (raw >> sh);
         write_lane_u64(dd, dst_bits, i, rv);
      }
   }
}

static void model_narrow_shift_round(uint8_t* dst, const uint8_t* a, const uint8_t* b,
                                     unsigned src_bits, unsigned dst_bits,
                                     unsigned src_lanes, int is_arith,
                                     unsigned total_bytes)
{
   unsigned chunk_bytes = src_lanes * src_bits / 8;
   unsigned chunks = total_bytes / chunk_bytes;
   memset(dst, 0, total_bytes);
   for (unsigned chunk = 0; chunk < chunks; chunk++) {
      const uint8_t* sa = a + chunk * chunk_bytes;
      const uint8_t* sb = b + chunk * chunk_bytes;
      uint8_t* dd = dst + chunk * chunk_bytes;
      for (unsigned i = 0; i < src_lanes; i++) {
         unsigned sh = (unsigned)(read_lane_u64(sb, src_bits, i) % src_bits);
         uint64_t raw = read_lane_u64(sa, src_bits, i);
         uint64_t rv;
         if (sh == 0) {
            rv = raw;
         } else if (is_arith) {
            int64_t base = sx64(raw, src_bits) >> sh;
            uint64_t round = (raw >> (sh - 1)) & 1;
            rv = ux_from_sx(base + (int64_t)round, src_bits);
         } else {
            rv = (raw >> sh) + ((raw >> (sh - 1)) & 1);
         }
         write_lane_u64(dd, dst_bits, i, rv);
      }
   }
}

static void model_rotate_imm(uint8_t* dst, const uint8_t* a, unsigned bits,
                             unsigned lanes, unsigned sh)
{
   unsigned i;
   sh %= bits;
   for (i = 0; i < lanes; i++) {
      uint64_t raw = read_lane_u64(a, bits, i);
      uint64_t rv = sh == 0 ? raw
                            : ((raw >> sh) | (raw << (bits - sh))) & lane_mask_u64(bits);
      write_lane_u64(dst, bits, i, rv);
   }
}

static void model_rotate_var(uint8_t* dst, const uint8_t* a, const uint8_t* b,
                             unsigned bits, unsigned lanes)
{
   unsigned i;
   for (i = 0; i < lanes; i++) {
      unsigned sh = (unsigned)(read_lane_u64(b, bits, i) % bits);
      uint64_t raw = read_lane_u64(a, bits, i);
      uint64_t rv = sh == 0 ? raw
                            : ((raw >> sh) | (raw << (bits - sh))) & lane_mask_u64(bits);
      write_lane_u64(dst, bits, i, rv);
   }
}

static void model_avgr(uint8_t* dst, const uint8_t* a, const uint8_t* b,
                       unsigned bits, unsigned lanes, int is_signed)
{
   unsigned i;
   for (i = 0; i < lanes; i++) {
      uint64_t raw_a = read_lane_u64(a, bits, i);
      uint64_t raw_b = read_lane_u64(b, bits, i);
      if (is_signed) {
         int64_t av = sx64(raw_a, bits);
         int64_t bv = sx64(raw_b, bits);
         int64_t rv = (av >> 1) + (bv >> 1) + ((raw_a | raw_b) & 1);
         write_lane_u64(dst, bits, i, ux_from_sx(rv, bits));
      } else {
         uint64_t rv = (raw_a >> 1) + (raw_b >> 1) + ((raw_a | raw_b) & 1);
         write_lane_u64(dst, bits, i, rv);
      }
   }
}

static void model_avg(uint8_t* dst, const uint8_t* a, const uint8_t* b,
                      unsigned bits, unsigned lanes, int is_signed)
{
   unsigned i;
   for (i = 0; i < lanes; i++) {
      uint64_t raw_a = read_lane_u64(a, bits, i);
      uint64_t raw_b = read_lane_u64(b, bits, i);
      if (is_signed) {
         int64_t av = sx64(raw_a, bits);
         int64_t bv = sx64(raw_b, bits);
         int64_t rv = (av >> 1) + (bv >> 1) + ((raw_a & raw_b) & 1);
         write_lane_u64(dst, bits, i, ux_from_sx(rv, bits));
      } else {
         uint64_t rv = (raw_a >> 1) + (raw_b >> 1) + ((raw_a & raw_b) & 1);
         write_lane_u64(dst, bits, i, rv);
      }
   }
}

static uint64_t abs_signed_lane(uint64_t raw, unsigned bits)
{
   int64_t v = sx64(raw, bits);
   return ux_from_sx(v < 0 ? -v : v, bits);
}

static void model_adda(uint8_t* dst, const uint8_t* a, const uint8_t* b,
                       unsigned bits, unsigned lanes)
{
   unsigned i;
   for (i = 0; i < lanes; i++) {
      uint64_t rv = abs_signed_lane(read_lane_u64(a, bits, i), bits)
                  + abs_signed_lane(read_lane_u64(b, bits, i), bits);
      write_lane_u64(dst, bits, i, rv);
   }
}

static void model_absd(uint8_t* dst, const uint8_t* a, const uint8_t* b,
                       unsigned bits, unsigned lanes, int is_signed)
{
   unsigned i;
   for (i = 0; i < lanes; i++) {
      if (is_signed) {
         int64_t av = sx64(read_lane_u64(a, bits, i), bits);
         int64_t bv = sx64(read_lane_u64(b, bits, i), bits);
         int64_t diff = av > bv ? (av - bv) : (bv - av);
         write_lane_u64(dst, bits, i, ux_from_sx(diff, bits));
      } else {
         uint64_t av = read_lane_u64(a, bits, i);
         uint64_t bv = read_lane_u64(b, bits, i);
         uint64_t diff = av > bv ? (av - bv) : (bv - av);
         write_lane_u64(dst, bits, i, diff);
      }
   }
}

static void model_minmax(uint8_t* dst, const uint8_t* a, const uint8_t* b,
                         unsigned bits, unsigned lanes, int is_signed,
                         int is_min)
{
   unsigned i;
   for (i = 0; i < lanes; i++) {
      if (is_signed) {
         int64_t av = sx64(read_lane_u64(a, bits, i), bits);
         int64_t bv = sx64(read_lane_u64(b, bits, i), bits);
         int64_t rv = is_min ? (av < bv ? av : bv) : (av > bv ? av : bv);
         write_lane_u64(dst, bits, i, ux_from_sx(rv, bits));
      } else {
         uint64_t av = read_lane_u64(a, bits, i);
         uint64_t bv = read_lane_u64(b, bits, i);
         write_lane_u64(dst, bits, i, is_min ? (av < bv ? av : bv)
                                             : (av > bv ? av : bv));
      }
   }
}

static void model_minmaxi(uint8_t* dst, const uint8_t* a, unsigned bits,
                          unsigned lanes, int64_t imm, int is_signed, int is_min)
{
   unsigned i;
   for (i = 0; i < lanes; i++) {
      if (is_signed) {
         int64_t av = sx64(read_lane_u64(a, bits, i), bits);
         int64_t rv = is_min ? (av < imm ? av : imm) : (av > imm ? av : imm);
         write_lane_u64(dst, bits, i, ux_from_sx(rv, bits));
      } else {
         uint64_t av = read_lane_u64(a, bits, i);
         uint64_t uimm = (uint64_t)imm;
         write_lane_u64(dst, bits, i, is_min ? (av < uimm ? av : uimm)
                                             : (av > uimm ? av : uimm));
      }
   }
}

static void model_mullo(uint8_t* dst, const uint8_t* a, const uint8_t* b,
                        unsigned bits, unsigned lanes)
{
   unsigned i;
   for (i = 0; i < lanes; i++) {
      uint64_t rv = read_lane_u64(a, bits, i) * read_lane_u64(b, bits, i);
      write_lane_u64(dst, bits, i, rv);
   }
}

static void model_mulhi(uint8_t* dst, const uint8_t* a, const uint8_t* b,
                        unsigned bits, unsigned lanes, int is_signed)
{
   unsigned i;
   for (i = 0; i < lanes; i++) {
      if (is_signed) {
         __int128 prod = (__int128)sx64(read_lane_u64(a, bits, i), bits)
                       * (__int128)sx64(read_lane_u64(b, bits, i), bits);
         write_lane_u64(dst, bits, i, (uint64_t)((unsigned __int128)prod >> bits));
      } else {
         unsigned __int128 prod = (unsigned __int128)read_lane_u64(a, bits, i)
                                * (unsigned __int128)read_lane_u64(b, bits, i);
         write_lane_u64(dst, bits, i, (uint64_t)(prod >> bits));
      }
   }
}

static void model_maddmsub(uint8_t* dst, const uint8_t* acc, const uint8_t* a,
                           const uint8_t* b, unsigned bits, unsigned lanes,
                           int is_sub)
{
   unsigned i;
   for (i = 0; i < lanes; i++) {
      uint64_t rv = read_lane_u64(acc, bits, i);
      uint64_t prod = read_lane_u64(a, bits, i) * read_lane_u64(b, bits, i);
      write_lane_u64(dst, bits, i, is_sub ? (rv - prod) : (rv + prod));
   }
}

static void model_divmod(uint8_t* dst, const uint8_t* a, const uint8_t* b,
                         unsigned bits, unsigned lanes, int is_signed,
                         int is_mod)
{
   unsigned i;
   for (i = 0; i < lanes; i++) {
      if (is_signed) {
         int64_t av = sx64(read_lane_u64(a, bits, i), bits);
         int64_t bv = sx64(read_lane_u64(b, bits, i), bits);
         int64_t rv = 0;
         if (bv != 0)
            rv = is_mod ? (av % bv) : (av / bv);
         write_lane_u64(dst, bits, i, ux_from_sx(rv, bits));
      } else {
         uint64_t av = read_lane_u64(a, bits, i);
         uint64_t bv = read_lane_u64(b, bits, i);
         uint64_t rv = 0;
         if (bv != 0)
            rv = is_mod ? (av % bv) : (av / bv);
         write_lane_u64(dst, bits, i, rv);
      }
   }
}

static void model_signcov(uint8_t* dst, const uint8_t* a, const uint8_t* b,
                          unsigned bits, unsigned lanes)
{
   unsigned i;
   for (i = 0; i < lanes; i++) {
      int64_t av = sx64(read_lane_u64(a, bits, i), bits);
      int64_t bv = sx64(read_lane_u64(b, bits, i), bits);
      int64_t rv = av == 0 ? 0 : (av > 0 ? bv : -bv);
      write_lane_u64(dst, bits, i, ux_from_sx(rv, bits));
   }
}

static void model_mulw(uint8_t* dst, const uint8_t* a, const uint8_t* b,
                       unsigned src_bits, unsigned dst_bits, unsigned lanes,
                       int odd, int sign_a, int sign_b)
{
   unsigned i;
   for (i = 0; i < lanes; i++) {
      unsigned src_idx = 2 * i + (unsigned)odd;
      __int128 av = sign_a ? (__int128)sx64(read_lane_u64(a, src_bits, src_idx), src_bits)
                           : (__int128)read_lane_u64(a, src_bits, src_idx);
      __int128 bv = sign_b ? (__int128)sx64(read_lane_u64(b, src_bits, src_idx), src_bits)
                           : (__int128)read_lane_u64(b, src_bits, src_idx);
      unsigned __int128 raw = (unsigned __int128)(av * bv);
      write_lane_wide(dst, dst_bits, i, raw);
   }
}

static void model_half_addsub_widen(uint8_t* dst, const uint8_t* a,
                                    const uint8_t* b, unsigned src_bits,
                                    unsigned dst_bits, unsigned lanes,
                                    int is_unsigned, int is_sub)
{
   unsigned i;
   for (i = 0; i < lanes; i++) {
      unsigned idx_a = 2 * i + 1;
      unsigned idx_b = 2 * i;
      if (is_unsigned) {
         uint64_t av = read_lane_u64(a, src_bits, idx_a);
         uint64_t bv = read_lane_u64(b, src_bits, idx_b);
         uint64_t rv = is_sub ? (av - bv) : (av + bv);
         write_lane_wide(dst, dst_bits, i, rv);
      } else {
         __int128 av = sx64(read_lane_u64(a, src_bits, idx_a), src_bits);
         __int128 bv = sx64(read_lane_u64(b, src_bits, idx_b), src_bits);
         __int128 rv = is_sub ? (av - bv) : (av + bv);
         write_lane_wide(dst, dst_bits, i, (unsigned __int128)rv);
      }
   }
}

static void model_addw(uint8_t* dst, const uint8_t* a, const uint8_t* b,
                       unsigned src_bits, unsigned dst_bits, unsigned lanes,
                       int odd, int sign_a, int sign_b)
{
   unsigned i;
   for (i = 0; i < lanes; i++) {
      unsigned idx = 2 * i + (unsigned)odd;
      __int128 av = sign_a ? (__int128)sx64(read_lane_u64(a, src_bits, idx), src_bits)
                           : (__int128)read_lane_u64(a, src_bits, idx);
      __int128 bv = sign_b ? (__int128)sx64(read_lane_u64(b, src_bits, idx), src_bits)
                           : (__int128)read_lane_u64(b, src_bits, idx);
      __int128 rv = av + bv;
      write_lane_wide(dst, dst_bits, i, (unsigned __int128)rv);
   }
}

static void model_addsubw(uint8_t* dst, const uint8_t* a, const uint8_t* b,
                          unsigned src_bits, unsigned dst_bits,
                          unsigned lanes, int odd, int sign_a, int sign_b,
                          int is_sub)
{
   unsigned i;
   for (i = 0; i < lanes; i++) {
      unsigned idx = 2 * i + (unsigned)odd;
      __int128 av = sign_a ? (__int128)sx64(read_lane_u64(a, src_bits, idx), src_bits)
                           : (__int128)read_lane_u64(a, src_bits, idx);
      __int128 bv = sign_b ? (__int128)sx64(read_lane_u64(b, src_bits, idx), src_bits)
                           : (__int128)read_lane_u64(b, src_bits, idx);
      __int128 rv = is_sub ? (av - bv) : (av + bv);
      write_lane_wide(dst, dst_bits, i, (unsigned __int128)rv);
   }
}

static void model_maddw(uint8_t* dst, const uint8_t* acc, const uint8_t* a,
                        const uint8_t* b, unsigned src_bits, unsigned dst_bits,
                        unsigned lanes, int odd, int sign_a, int sign_b)
{
   unsigned i;
   for (i = 0; i < lanes; i++) {
      unsigned src_idx = 2 * i + (unsigned)odd;
      __int128 av = sign_a ? (__int128)sx64(read_lane_u64(a, src_bits, src_idx), src_bits)
                           : (__int128)read_lane_u64(a, src_bits, src_idx);
      __int128 bv = sign_b ? (__int128)sx64(read_lane_u64(b, src_bits, src_idx), src_bits)
                           : (__int128)read_lane_u64(b, src_bits, src_idx);
      unsigned __int128 raw = (unsigned __int128)(av * bv)
                            + read_lane_wide(acc, dst_bits, i);
      write_lane_wide(dst, dst_bits, i, raw);
   }
}

static void check_bytes(test_state* tst, const char* name,
                        const void* got, const void* exp, size_t nbytes)
{
   tst->checks++;
   if (memcmp(got, exp, nbytes) == 0)
      return;

   tst->fails++;
   printf("FAIL %s\n", name);
}

#endif
