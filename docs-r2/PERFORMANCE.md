# PERFORMANCE R2 (ub=4096, b=8192, ctx 262144, MTP on, 128 generated tokens)
Prefill mean tok/s (last block in brackets). A = flag off. B-R1 = kernel of R1 (240 VGPR). B-P2 = waves_per_eu(8), 128 VGPR, 68 B private scratch.

| ctx | A | B-R1 | B-P2 |
|---|---|---|---|
| 40K (6/6/3 runs) | 773-802 (585-602) | 860-891 (831-855) | 909-912 (923-931) |
| 65K (3/3/3) | 656-661 (441-453) | 875-877 (836-846) | 918-920 (904-914) |
| 127K (3/3/3) | 453-456 (279-280) | 848-853 (785-803) | 910-911 (864-873) |
| 204K (1/1/1) | 326 (187) | 828 (757) | 888 (811) |
| 250K (1/1/1) | 279 (156) | 807 (735) | 863 (794) |

Decode tok/s unchanged: A 47-57, B 47-56. Peak VRAM ~16017-16025 MiB, min free ~343-351 MiB in every run. ub=8192 (R1 only): B fits (compute 2364 MiB, 40K mean 894, 127K mean 869); A fails to allocate.
Per-block from the start of the request: ~-20% from the first block to 127K, ~-10% from the first sparse block (~32K). Earlier "-8%" statement is withdrawn.

## P2 profile (rocprofv3)
- Bench kernel only (n_t=4096, n_sel=2304): 131K ctx 130 ms -> 91 ms (-30%); 262K 144 -> 101 ms. Cause: VGPR 240 -> 128 lets two 8-wave CTAs co-reside per CU. LDS 18.9 KiB/CTA unchanged; scratch 0 -> 68 B/thread (small spill, private scratch, no global scratch buffer).
- Real server at 65K prefill: QSA kernel 15% of GPU time (108 calls, 95 ms avg) vs dense flash_attn_tile 14% (84 calls: the sub-32K ubatches); mul_mat_q is now the largest consumer.
- Rollback of P2: define QSA_R2_WAVES_PER_EU larger/remove the attribute or revert commit after b111c95.

## MoE counters (log lines "prefill ring ... read in place" / "rewarm: ... evicted")
Evicted entries before rewarm, A -> B-R1: 40K 386 -> 98; 65K 386 -> 125; 127K 720 -> 435-503; 204K 1620 -> 1009; 250K 2232 -> 1299. Experts read in place: 40K 14.0% -> 15.3%; 127K 13.8% -> 14.7%; 250K 11.8% -> 13.3%. This is lower cache pressure, not proof of better decode hit rate; hits/misses and PCIe after prefill were not measured.
