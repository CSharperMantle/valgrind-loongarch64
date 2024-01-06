#include <stdio.h>

#define TESTINST_MOV(v1, v2, v3, v4, v5, v6, val)                              \
   {                                                                           \
      union {                                                                  \
         unsigned long u;                                                      \
         double        d;                                                      \
      } res1 = {.u = (unsigned long)v1};                                       \
      union {                                                                  \
         unsigned long u;                                                      \
         double        d;                                                      \
      } res2 = {.u = (unsigned long)v2};                                       \
      union {                                                                  \
         unsigned long u;                                                      \
         double        d;                                                      \
      } res3             = {.u = (unsigned long)v3};                           \
      unsigned long res4 = (unsigned long)v4;                                  \
      unsigned long res5 = (unsigned long)v5;                                  \
      unsigned long res6 = (unsigned long)v6;                                  \
      union {                                                                  \
         unsigned long u;                                                      \
         double        d;                                                      \
      } fval = {.u = (unsigned long)val};                                      \
      __asm__ __volatile__("movgr2fr.w %0, %6  \n\t"                           \
                           "movgr2fr.d %1, %6  \n\t"                           \
                           "movgr2frh.w %2, %6 \n\t"                           \
                           "movfr2gr.s %3, %7  \n\t"                           \
                           "movfrh2gr.s %4, %7  \n\t"                          \
                           "movfr2gr.d %5, %7  \n\t"                           \
                           : "+f"(res1.d), "+f"(res2.d), "+f"(res3.d),         \
                             "+r"(res4), "+r"(res5), "+r"(res6)                \
                           : "r"(val), "f"(fval.d)                             \
                           : "memory");                                        \
      printf("movgr2fr.w ::\n");                                               \
      printf("input: %#018lx %#018lx\n", v1, val);                             \
      printf("output: %#018lx\n", res1.u);                                     \
      printf("movgr2fr.d ::\n");                                               \
      printf("input: %#018lx %#018lx\n", v2, val);                             \
      printf("output: %#018lx\n", res2.u);                                     \
      printf("movgr2frh.w ::\n");                                              \
      printf("input: %#018lx %#018lx\n", v3, val);                             \
      printf("output: %#018lx\n", res3.u);                                     \
      printf("movfr2gr.s ::\n");                                               \
      printf("input: %#018lx %#018lx\n", v4, val);                             \
      printf("output: %#018lx\n", res4);                                       \
      printf("movfrh2gr.s ::\n");                                              \
      printf("input: %#018lx %#018lx\n", v5, val);                             \
      printf("output: %#018lx\n", res5);                                       \
      printf("movfr2gr.d ::\n");                                               \
      printf("input: %#018lx %#018lx\n", v6, val);                             \
      printf("output: %#018lx\n", res6);                                       \
   }

#define TESTINST_FSCR(fcsr, val)                                               \
   {                                                                           \
      unsigned long res;                                                       \
      __asm__ __volatile__("movgr2fcsr " fcsr ", %1 \n\t"                      \
                           "movfcsr2gr %0, " fcsr " \n\t"                      \
                           : "=r"(res)                                         \
                           : "r"(val)                                          \
                           : "memory");                                        \
      printf("movgr2fcsr movfcsr2gr ::\n");                                    \
      printf("input: %#018lx\n", val);                                         \
      printf("output: %#018lx\n", res);                                        \
   }

#define TESTINST_CF(fcc, v1, v2, val)                                          \
   {                                                                           \
      union {                                                                  \
         unsigned long u;                                                      \
         double        d;                                                      \
      } res1             = {.u = (unsigned long)v1};                           \
      unsigned long res2 = (unsigned long)v2;                                  \
      union {                                                                  \
         unsigned long u;                                                      \
         double        d;                                                      \
      } fval = {.u = (unsigned long)val};                                      \
      __asm__ __volatile__("movfr2cf " fcc ", %2 \n\t"                         \
                           "movcf2fr %0, " fcc " \n\t"                         \
                           "movgr2cf " fcc ", %3 \n\t"                         \
                           "movcf2gr %1, " fcc " \n\t"                         \
                           : "+f"(res1.d), "+r"(res2)                          \
                           : "f"(fval.d), "r"(val)                             \
                           : "memory");                                        \
      printf("movfr2cf movcf2fr ::\n");                                        \
      printf("input: %#018lx %#018lx\n", v1, val);                             \
      printf("output: %lx\n", res1.u);                                         \
      printf("movgr2cf movcf2gr ::\n");                                        \
      printf("input: %#018lx %#018lx\n", v2, val);                             \
      printf("output: %lx\n", res2);                                           \
   }

void test(void)
{
   TESTINST_MOV(0x1234123412341234UL, 0x5678567856785678UL, 0x9abc9abc9abc9abcUL, 0xdef0def0def0def0UL, 0x2468246824682468UL, 0x3579357935793579UL, 0x0123456789abcdefUL);
   TESTINST_MOV(0x1234123412341234UL, 0x5678567856785678UL, 0x9abc9abc9abc9abcUL, 0xdef0def0def0def0UL, 0x2468246824682468UL, 0x3579357935793579UL, 0xfedcba9876543210UL);

   TESTINST_FSCR("$fcsr0", 0x0123456789abcdefUL);
   TESTINST_FSCR("$fcsr0", 0xfedcba9876543210UL);
   TESTINST_FSCR("$fcsr1", 0x0123456789abcdefUL);
   TESTINST_FSCR("$fcsr1", 0xfedcba9876543210UL);
   TESTINST_FSCR("$fcsr2", 0x0123456789abcdefUL);
   TESTINST_FSCR("$fcsr2", 0xfedcba9876543210UL);
   TESTINST_FSCR("$fcsr3", 0x0123456789abcdefUL);
   TESTINST_FSCR("$fcsr3", 0xfedcba9876543210UL);

   TESTINST_CF("$fcc0", 0x1234123412341234UL, 0x5678567856785678UL, 0xffffffffffffffffUL);
   TESTINST_CF("$fcc0", 0x1234123412341234UL, 0x5678567856785678UL, 0xdef0def0def0def0UL);
   TESTINST_CF("$fcc1", 0x1234123412341234UL, 0x5678567856785678UL, 0xffffffffffffffffUL);
   TESTINST_CF("$fcc1", 0x1234123412341234UL, 0x5678567856785678UL, 0xdef0def0def0def0UL);
   TESTINST_CF("$fcc2", 0x1234123412341234UL, 0x5678567856785678UL, 0xffffffffffffffffUL);
   TESTINST_CF("$fcc2", 0x1234123412341234UL, 0x5678567856785678UL, 0xdef0def0def0def0UL);
   TESTINST_CF("$fcc3", 0x1234123412341234UL, 0x5678567856785678UL, 0xffffffffffffffffUL);
   TESTINST_CF("$fcc3", 0x1234123412341234UL, 0x5678567856785678UL, 0xdef0def0def0def0UL);
   TESTINST_CF("$fcc4", 0x1234123412341234UL, 0x5678567856785678UL, 0xffffffffffffffffUL);
   TESTINST_CF("$fcc4", 0x1234123412341234UL, 0x5678567856785678UL, 0xdef0def0def0def0UL);
   TESTINST_CF("$fcc5", 0x1234123412341234UL, 0x5678567856785678UL, 0xffffffffffffffffUL);
   TESTINST_CF("$fcc5", 0x1234123412341234UL, 0x5678567856785678UL, 0xdef0def0def0def0UL);
   TESTINST_CF("$fcc6", 0x1234123412341234UL, 0x5678567856785678UL, 0xffffffffffffffffUL);
   TESTINST_CF("$fcc6", 0x1234123412341234UL, 0x5678567856785678UL, 0xdef0def0def0def0UL);
   TESTINST_CF("$fcc7", 0x1234123412341234UL, 0x5678567856785678UL, 0xffffffffffffffffUL);
   TESTINST_CF("$fcc7", 0x1234123412341234UL, 0x5678567856785678UL, 0xdef0def0def0def0UL);
}

int main(void)
{
   test();
   return 0;
}
