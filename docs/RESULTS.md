# 실험 결과 종합 (2026-07 선행 연구)

연구 여정: **failure-prototype 공유 (federated RL)** → Gate 1에서 기각 →
**stochastic replay perturbation** → mechanism 분해 → **priority staleness와
periodic re-anchoring** 으로 수렴.

모든 branch 실험은 seed별 동일 pre-change checkpoint(150k step, local_per,
4 client Hopper/Walker + action-scale wrapper)에서 분기하는 paired 설계.
C3 actuator 손상(1.0→0.65) 후 50k step, post-change AUC 기준, 5 seeds.

## 1. Gate 1: prototype 공유는 경험 전달이 아니었다 (STOP)

| 방법 | 첫 20k 실패 | 50k AUC | AUROC(우선순위→실패 예측) |
|---|---|---|---|
| local_per | 114.0 | 5695 | — |
| noisy_per (내용 없는 매칭 noise) | 113.4 | 6336 | 0.50 |
| proto_random (무작위 centroid) | 112.8 | 6315 | 0.44 |
| proto_per (진짜 prototype) | 112.4 | 6455 | 0.54 |

진짜 ≈ 무작위 ≈ noise. 이득의 ~85%는 priority perturbation 자체.
초기 실패 감소는 모든 방법에서 0. → federation 기각, replay 연구로 pivot.

## 2. Pivot: noise인가 de-greedification인가 → 둘 다 아닌 제3의 답

vs local_per (paired):

| 방법 | AUC | 승률 |
|---|---|---|
| local (uniform) | +2.8% | 4/5 |
| per_soft (α=0.3) | +4.3% | 3/5 |
| per_eps (50% uniform floor) | +7.0% | 5/5 |
| **per_prng (PER × 주기적 곱셈 noise)** | **+11.7%** | **5/5** |
| noisy_per (federated) | +11.2% | 4/5 |

순수 local per_prng가 federated noisy_per와 일치 → 통신 기여 최종 0 확정.
softer-only는 절반의 이득 → 평탄화만으로는 부족.

## 3. Mechanism 분해 (stage 1–3)

**재추첨 주기** (per_prng): static **−2.7%** / 1k +5.9% / 5k +10.3% / **10k +11.7%**
→ 고정 noise는 해로움. 주기적 재추첨 + epoch 유지가 필수.

**강도 λ**: 0.5 +5.5% / **2.0 +11.7%** / 8.0 +5.4% → inverted-U.

**Freshness vs perturbation** (stage 2, 결정 실험):

| 방법 | AUC | 승률 | worst seed |
|---|---|---|---|
| **per_reset_periodic (전체 priority를 현재 TD로 재계산, noise 없음)** | **+15.7%** | 5/5 | +48 |
| per_prng | +11.7% | 5/5 | +15 |
| reset + prng | +11.9% | 5/5 | — |
| per_batch_prng (batch마다 재추첨) | +6.2% | 4/5 | — |

→ **핵심 메커니즘은 priority freshness.** noise는 freshness의 signal-free
근사물(둘을 합쳐도 reset 단독보다 못함). per-batch는 marginalize되어 무효
(epoch 유지 필요 확인). 진단: reset은 pri_reliability를 0.77→0.87로 올리고
(신호 수리), prng는 0.68로 내림(낡은 신호 파괴) — 두 경로 모두 stale 신호보다
낫지만 수리가 우월.

**ESS-matched 대조군** (stage 3): per_prng의 ESS(0.661)는 plain PER(0.715)보다
낮음 → 이득은 평탄화가 아님. ESS를 맞춘 정적 per_soft(α=0.65)는 +7.4%(4/5,
min −1)로 재현 실패. α=0.75는 평균 +11.9%이지만 2/5 seed 음수(min −271)의
복권 — **주기적 재순위화만이 전 seed 균일 양의 개선**(sign test 5/5, p≈0.03).

**Stationary 대조군**: 변화 없음에도 prng +5.0%, reset +5.2% → 무해,
detector 없이 상시 적용 가능. drift-특이 증분은 reset 기준 ~+10.5%p.

**reset 주기 곡선**: 5k +6.2% / 10k +15.7% / 20k +4.8% — 전 주기 양수
(seed×주기 14/15승)이나 10k 피크는 seed 4 의존, 정밀 곡선엔 seed 추가 필요.

## 4. Walker2d 일반화: 실패 — 그러나 이론과 일치하는 실패

| | AUC | 승률 |
|---|---|---|
| per_prng | −14.9% | 1/5 |
| per_reset_periodic | +1.0% | 2/5 |

진단: Walker의 local_per pri_reliability = **0.85** (Hopper 0.77) —
**Walker에서는 priority가 애초에 stale하지 않음** → 고칠 것이 없고 noise는
분산만 추가. 벤치마크 전이 실패가 아니라 전제 조건(staleness) 부재.

주의: Hopper C3의 reliability도 무변화 client와 같음(0.77) → 이득의 예측
변수는 "drift 자체"가 아니라 **환경/학습단계의 staleness 수준**
(Hopper stationary +5%와 일치).

## 5. Stage 5: staleness 가설 반증 — 방향 재검토 필요

### 5a. 버퍼 조작: re-anchoring보다 "버퍼를 비우는 것"이 2배 낫다

branch 시작 시 replay를 최근 30k만 남기고 절단. **동일 checkpoint·seed paired.**

| 조건 | local_per AUC | per_prng | per_reset_periodic |
|---|---|---|---|
| FULL buffer (150k) | 5695 | +11.7% (5/5) | **+15.7% (5/5)** |
| FRESH buffer (30k) | **7656 (+34%, 5/5)** | +2.7% (3/5) | −3.1% (4/5) |

두 가지가 동시에 드러남:

1. **버퍼 절단만으로 +34%** — 우리가 찾은 어떤 priority 기법(+15.7%)보다
   2배 이상 크고 5/5 seed 일관.
2. **fresh buffer에서 re-anchoring 이득은 소멸** (+15.7% → −3.1%).

**그러나 메커니즘 지표는 움직이지 않음**: local_per의 pri_reliability는
FULL 0.776 vs FRESH 0.767로 **사실상 동일**. 즉 "절단이 staleness를 없애서
이득이 사라졌다"는 설명은 **데이터가 지지하지 않음**. reliability 지표는
실제 구동 변수를 측정하지 못하고 있음.

**실제로 바뀐 것은 버퍼 구성비**(capacity 300k 고정, 절단은 size만 변경):

| | branch 종료 시 size | post-change 데이터 비율 |
|---|---|---|
| FULL | 150k → 200k | 25% |
| FRESH | 30k → 80k | **62%** |

→ 적응 속도를 지배하는 것은 우선순위 재조정이 아니라 **replay 내 post-change
데이터 비율**. priority re-anchoring은 이에 대한 약한 간접 대리물로 보임.
(단, 이 해석은 사후적이며 별도 검증 필요. 또한 buffer size/data age 효과는
plasticity·replay ratio 문헌에 선행 연구가 있을 가능성이 높음.)

### 5b. per_adaptive (reliability < τ=0.8일 때만 re-anchoring): 양쪽 다 실패

| | AUC vs local_per | 승률 | gate 발화율 | 평균 reliability |
|---|---|---|---|---|
| Hopper (발화 기대) | +7.3% | 5/5 | 71% | 0.730 |
| Walker (침묵 기대) | **−8.6%** | 1/5 | **52%** | 0.789 |

- Hopper: 전 seed 양수로 안전하지만 reset_periodic(+15.7%)의 **절반**.
  gating이 이득만 깎음.
- Walker: gate가 침묵하지 않고 **52% 발화** → 해로운 개입을 절반이나 수행,
  −8.6%. Walker의 reliability(0.789)가 τ=0.8 바로 아래에 걸려 있음.

→ reliability 통계는 두 환경을 분리하지 못하며 **gate 신호로 부적합**.
5a·5b가 독립적으로 같은 결론을 가리킴: **staleness/reliability는 이 현상의
올바른 메커니즘 설명이 아니다.**

## 6. 현재 증거 상태 요약 (정직한 정리)

**살아있는 것:**
- Hopper full-buffer에서 per_reset_periodic +15.7% (5/5 seed, worst +48).
  주기적 재추첨이 정적 방법보다 우수하다는 stage 1–3 분해는 그대로 유효.
- Federation은 기여 0 (Gate 1, 확정).

**무너진 것:**
- "priority staleness가 원인"이라는 메커니즘 설명 (5a 지표 불변, 5b gate 실패)
- Walker2d 일반화 (−14.9% / +1.0%), 그리고 그 실패를 설명하려던 staleness 가설
- staleness-gated adaptive 방법

**새로 드러난 것 (검증 필요):**
- 버퍼 recency/구성비가 priority 조작보다 훨씬 큰 효과(+34%). 단 선행연구
  가능성 높음 — prior-art 조사가 다음 실험보다 우선.

**결론:** 현재 상태로는 "periodic re-anchoring" 메커니즘 논문을 지지하기
어려움. 남은 선택지는 (a) buffer-composition 방향으로 재정향(선행연구 조사
선행), (b) Hopper 한정 경험적 결과로 범위 축소, (c) 중단.

## 8. Stage 6A: composition 인과성 확인 — 강한 GO

**설계:** 버퍼를 자르지 않고(전체 유지) **배치의 post-change 비율 ρ만 강제**.
파티션 내부는 동일하게 PER. 동일 checkpoint·seed paired, 5 seeds.

| 조건 | post AUC vs local_per | final return | 첫 20k 실패 |
|---|---|---|---|
| local_per (자연) | — | 610 | 114.0 |
| ρ=0.25 | +11.0% (5/5) | +11.9% | −6.8 (5/5) |
| ρ=0.50 | +15.2% (5/5) | +15.0% | −11.6 (5/5) |
| ρ=0.75 | **+35.3%** (5/5) | +46.4% | −14.8 (5/5) |
| ρ=1.00 | **+93.7%** (5/5) | **+195.5%** (5/5) | −7.2 |
| *(참고)* reset_periodic | +15.7% | +37.5% | −0.2 |
| *(참고)* fresh-30k 버퍼 | +34.4% | — | — |

**Q1 판정: GO.** ρ에 대해 **완전 단조**, 전 구간 5/5 seed 일관.
ρ=1.00은 우리가 찾은 어떤 priority 기법(+15.7%)보다 **6배**, fresh-buffer보다
**3배** 큼. **초기 실패도 처음으로 유의하게 감소**(ρ=0.75에서 −14.8, 5/5) —
Gate 1 이래 어떤 방법도 못 하던 것.

**핵심: 삭제는 불필요.** ρ=0.75(+35.3%) ≈ fresh-30k(+34.4%). fresh buffer의
이득은 **데이터를 지워서**가 아니라 **sampling 구성비** 때문이며, full buffer에서
비율만 강제해도 완전히 재현됨. (fresh-30k의 종료 시 post 비율 62%가
ρ 0.50–0.75 구간에 놓이는 것과 정합.)

### Q2: reset은 composition을 통해 작동하지 **않음**

| seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Δ AUC (reset − lp) | +624 | +269 | +30 | −428 | +247 |
| Δ 실측 post 비율 | 0.000 | +0.002 | 0.000 | 0.000 | −0.001 |

corr = **+0.058** (≈0). reset은 post sampling mass를 전혀 올리지 않음.
→ reset의 +15.7%는 composition 경로가 아니며, reliability로도 설명 안 됨
(stage 5). **메커니즘 미해명 상태로 남되, composition(+93.7%)에 비해
부차적 효과**로 격하.

### 메커니즘: PER은 변화 후 새 데이터를 *굶긴다* (seed별 검증)

| seed | td_pre | td_post | ratio | 버퍼 비중 | 샘플링 비율 | starvation |
|---|---|---|---|---|---|---|
| 0 | 2.004 | 2.171 | 1.083 | 0.149 | 0.147 | −0.002 |
| 1 | 2.266 | 2.010 | 0.887 | 0.149 | 0.138 | −0.012 |
| 2 | 1.958 | 1.622 | 0.828 | 0.149 | 0.132 | −0.017 |
| 3 | 1.884 | 1.721 | 0.913 | 0.149 | 0.131 | −0.019 |
| 4 | 1.635 | 1.685 | 1.031 | 0.149 | 0.145 | −0.004 |

**정정:** 평균 ratio 0.945는 TD 역전이 보편적이라는 뜻이 아님.
TD 역전(post < pre)은 **3/5 seed**에서만 성립(seed 0·4는 역전 없음).

**그러나 starvation은 5/5 seed 일관** — PER이 새 regime을 버퍼 비중보다 적게
샘플링하는 현상 자체는 예외 없음. 그리고 **두 값이 seed 간 단조 대응**:
TD ratio가 낮을수록 starvation이 큼(1.083→−0.002, 1.031→−0.004,
0.913→−0.019, 0.887→−0.012, 0.828→−0.017).

→ 정확한 진술: *"PER은 변화 후 새 regime을 일관되게 과소 샘플링하며(5/5),
그 정도는 두 regime의 상대적 TD error 크기로 설명된다(seed 간 단조)."*
"변화 후 새 데이터의 TD가 더 낮다"는 일반 법칙으로 주장할 수 없음.

인과 사슬:
```
분포 변화 → 구 regime transition이 off-distribution화 → 높은 TD error 유지
        → PER이 구 데이터에 sampling mass 집중 → 새 regime 학습 신호 희석
        → 적응 지연.  ρ 강제로 교정 시 단조 회복(최대 +93.7%)
```

**단, ρ=1.00 우세는 단발 변화 설정의 한계를 그대로 드러냄** — "새 데이터만
쓰기"가 압도적이라는 것은 buffer-clear류 baseline이 강력하다는 뜻이며,
사용자가 지적한 대로 저널 신규성은 **반복 regime(A→B→A)의 plasticity–retention
균형**에서 나와야 함. ρ=1.00은 A 복귀 시 파국적 망각을 보일 것으로 예상되며,
그 지점이 다음 실험(6B)의 핵심.

## 9. Stage 6B: 반복 regime — lifetime 승자 없음 (STOP)

**설계:** A(base 150k) → B 50k(scale 0.65) → A 50k(scale 1.0).
슬라이딩 50k recency window 위 ρ 강제. 5 seeds.

| 조건 | phaseB AUC | returnA AUC | **lifetime** | 복귀 시점 첫 평가 |
|---|---|---|---|---|
| local_per | 6003 | 21384 | **27387** | 2395 |
| ρ=1.00 (recent-only) | **8125** | 18109 | 26234 (−4.2%, 2/5) | 1581 (−34%, 4/5 악화) |
| ρ=0.75 | 6553 | 20836 | 27388 (+0.0%, 3/5) | 2315 |
| ρ=0.50 | 6917 | 17098 | 24015 (−12.3%, 1/5) | 1694 |

**Trade-off 자체는 실재하고 곡선에서 뚜렷함:**
phase B 구간(190k–200k)에서 ρ=1.00은 1109/929/1232 vs local_per 627/711/607
(약 +75%) — 6A의 적응 이득 재현. 복귀 순간에는 반대로 1581 vs 2395로
**−34% retention 비용**을 지불(4/5 seed 악화, seed 0·4는 446·260으로 파국적).

**그러나 사용자 정의 Go 기준("lifetime AUC에서 둘 다 이김")을 만족하는 방법 없음.**
ρ=0.75가 정확히 동률(27388 vs 27387), 나머지는 손해. 승률 1/5–3/5는 잡음 수준.
→ **STOP.**

### 판정을 신뢰하기 전에 반드시 밝혀야 할 설계 결함 3가지

1. **Lifetime AUC가 A phase에 지배됨** (21384 vs 6003, 3.5배). A가 고보상
   regime이므로 lifetime 합산은 사실상 "A를 얼마나 잘하나"를 재며,
   B 적응 이득을 공정하게 반영할 수 없는 지표였음.
2. **파국적 망각이 애초에 발생하지 않음.** 전 방법의 recovery ratio가 1.27–1.71로
   **변화 전 A 성능을 오히려 초과**. base checkpoint(150k, return 1604)가
   Hopper 상한(~3000+)에서 멀어 "A 복귀"가 실질적으로 "A 학습 계속"이었음.
   retention 압력이 설계상 생성되지 않았음.
3. **Seed 분산이 효과를 압도** (ρ=0.75 lifetime 19447~32752).

→ 이 결과는 "혼합이 무용하다"의 증거라기보다 **이 스케줄이 retention을
시험하지 못했다**는 증거에 가까움. 재시도한다면 수렴된 base(300k+),
더 긴 B, 짧은 복귀 window, phase별 정규화 지표가 필요.

## 10. Stage 7: starvation 예측 gate — 기각 (STOP)

`G = buffer_share_post − sampled_post_mass` (local_per에서만, 변화 후 초기
10k/20k)가 ρ=0.75 개입의 이득을 예측하는지 검정. 4 조건 × 5 seed.

| 조건 | \|1−c\| | G@20k | D_early | 승률 | D_full | 승률 |
|---|---|---|---|---|---|---|
| Hopper 0.85 | 0.15 | +0.0117 | **+50.2%** | 5/5 | +34.4% | 5/5 |
| Hopper 0.65 | 0.35 | +0.0079 | +32.7% | 4/5 | +29.5% | 5/5 |
| Hopper 0.45 | 0.55 | +0.0094 | **−0.8%** | 2/5 | +6.6% | 3/5 |
| Walker 0.65 | 0.35 | +0.0155 | +17.4% | 4/5 | **+43.2%** | 4/5 |

### 판정: STOP 기준 5개 중 4개 충족

1. **순위 불일치** — G 순위(Walker > H0.85 > H0.45 > H0.65) ≠ 이득 순위
   (H0.85 > H0.65 > Walker > H0.45).
2. **Hopper 단조성 역전** — 예측은 0.45 > 0.65 > 0.85였으나 실제는
   **+50.2% > +32.7% > −0.8%**로 정반대. **약한 drift가 가장 큰 이득**,
   강한 drift는 이득 없음. (스모크에서 본 "drift↑ ⇒ G↑"도 본실험에서
   무너짐: G는 0.0117 / 0.0079 / 0.0094로 비단조.)
3. **Confound test 실패** — 동일 \|1−c\|=0.35인 Hopper vs Walker에서
   Walker가 G는 크고(+0.0155 vs +0.0079) early 이득은 작음(+17.4% vs +32.7%).
   **부호 반대.**
4. **Seed 수준 전 조건 음의 상관** — −0.541 / −0.699 / −0.605 / −0.220.
   starvation이 큰 seed가 오히려 이득이 작음.
5. **메커니즘 방향 반전** — 예측은 `corr(G,D_early) > corr(G,D_full)`인데
   실제는 −0.332 vs +0.299.

탐색적 R²에서 `drift+G`(0.665)가 `drift`(0.391)보다 높지만, G의 부호가
가설과 반대이므로 설명력 증가는 메커니즘 근거가 되지 못함.

→ **present-regime starvation은 중심 메커니즘으로 사용할 수 없음.**

### 대신 드러난 더 단순한 규칙성: 붕괴 정도

| 조건 | 변화 후 유지율 | ρ=0.75 절대 이득 |
|---|---|---|
| Hopper 0.85 | 60.6% | 1308 → 1753 (+445) |
| Walker 0.65 | 60.0% | 983 → 1460 (+477) |
| Hopper 0.45 | **20.7%** | 355 → 379 (**+24**) |

정책이 절반쯤 살아남는 조건에서는 큰 이득, **거의 붕괴하면(20%) 아무 효과 없음.**
즉 개입은 *"적응할 정책이 남아 있을 때"* 작동하며, sampling 결손 크기와는
무관. 단 유지율도 크기 **순서**까지 설명하지는 못함(H0.85와 Walker가 같은
60%인데 early 이득은 +50% vs +17%).

### 중요한 양성 결과: Walker에서 composition은 성공

Stage 4에서 priority 개입은 Walker에서 실패했음 (per_prng −14.9%,
per_reset_periodic +1.0%). 그러나 **ρ=0.75는 Walker에서 +43.2%(4/5),
절대 +477** — priority 조작이 실패한 환경에서 composition 조작은 작동.
"Walker는 replay 개입에 면역"이라는 이전 서술은 **틀렸음**; priority 축과
composition 축이 서로 다른 것이었음.

**요약:** 진단변수(G)는 죽었지만 개입(ρ quota)은 4조건 중 3조건에서 견고.
남은 문제는 이 개입이 ERE·Truncated Geometric replay와 구별되는가임.

## 11. Stage 8: 최종 신규성 Gate — STOP (Case A)

배포 가능한 `per_recent`(ρ=0.75, W=50k, 변화 시점 미사용)를 기존 recency
방법과 대조. TGR의 α는 return을 보지 않고 expected recency 정합으로 보정
(λ=1.875e−5, TGR 논문 표기로 α≈4.06 at N_max=150k / 8.12 at capacity 300k).

### Hopper 1.0→0.85

| 방법 | early | full | recency | entropy |
|---|---|---|---|---|
| PER | — | — | — | — |
| *Oracle quota* | *+43.9% (5/5)* | *+34.0% (5/5)* | — | — |
| Recent-window quota | +19.3% (4/5) | +25.3% (5/5) | 0.731 | 0.950 |
| **ERE-PER (cyc)** | **+56.4% (5/5)** | **+31.5% (4/5)** | 0.457 | 0.986 |
| TGR (matched) | −4.1% (2/5) | +1.8% (3/5) | 0.736 | 0.970 |

### Walker 1.0→0.65

| 방법 | early | full | recency | entropy |
|---|---|---|---|---|
| *Oracle quota* | *+18.3% (4/5)* | *+48.5% (4/5)* | — | — |
| Recent-window quota | +25.8% (5/5) | +26.0% (4/5) | 0.730 | 0.948 |
| **ERE-PER (cyc)** | +16.5% (3/5) | **+49.0% (5/5)** | 0.469 | 0.986 |
| TGR (matched) | +35.4% (4/5) | +35.1% (4/5) | 0.736 | 0.970 |

### Head-to-head (seed paired)

| | vs ERE early | vs ERE full | vs TGR early | vs TGR full |
|---|---|---|---|---|
| Hopper 0.85 | −23.7% (2/5) | −4.7% (2/5) | +24.4% (4/5) | +23.2% (4/5) |
| Walker 0.65 | +8.0% (3/5) | −15.4% (1/5) | −7.1% (2/5) | −6.8% (2/5) |

### 판정: **STOP**

1. **ERE-PER가 full AUC에서 두 환경 모두 quota를 앞섬**
   (+31.5% vs +25.3%, +49.0% vs +26.0%). 게다가 **oracle quota와 동등**
   (Hopper +34.0%, Walker +48.5%) — 즉 2019년 방법이 변화 시점을 아는 우리
   상한과 같은 성능을 detector 없이 달성.
2. **TGR도 Walker에서 quota를 이김**(−7.1%/−6.8%, 2/5). 따라서 "ERE만 졌으니
   canonical block ERE를 확인해야 한다"는 면제 조항이 적용되지 않음.
3. **환경별 승자가 갈림**(Hopper: quota > TGR, Walker: TGR > quota) — Case D도
   동시 성립. 단일 method paper는 지지되지 않음.
4. **Recency 정합은 유효**했음: quota 0.711→0.747 vs TGR 0.714→0.757로 양끝
   모두 근접. 따라서 TGR 비교 결과는 해석 가능하며, 남은 차이는 분포 모양
   (quota entropy 0.950 < TGR 0.970)뿐.

**ERE는 recency 정합 대상이 아님**(expected recency 0.46 vs 0.73)에도 이김 —
"현재 regime을 많이 뽑을수록 좋다"는 우리의 작업가설조차 정확하지 않음.
ERE는 **낮은 평균 recency + 높은 엔트로피 + cycle 내 범위 축소**라는 다른
구조로 더 나은 결과를 냄.

→ **replay-composition 방법 연구 종료.** 남은 효과는 알려진 recency 효과이며,
기존 방법이 이를 더 잘 구현함.

## 7. Prior-art 조사 (2026-07-28)

### 7a. 버퍼 구성비 방향 → 선행연구로 이미 점유됨

| 연구 | 내용 | 우리 발견과의 관계 |
|---|---|---|
| **When Does Non-Uniform Replay Matter in RL?** (arXiv 2605.10236, 2026-05) | 비균등 replay의 효과를 지배하는 3요소를 **replay volume · expected recency · sampling entropy**로 규정. "non-uniform replay는 replay volume이 낮을 때 가장 유용." **Truncated Geometric replay** 제안(최근 경험 편향 + 고엔트로피 유지, 오버헤드 무시 가능). 3 알고리즘 × 5 벤치마크. | **치명적.** 우리의 "버퍼 구성비가 prioritization을 압도한다"를 2개월 전에 더 강한 방법과 넓은 검증으로 발표. 우리 +34%는 이들의 recency 축 재발견. |
| **Revisiting Fundamentals of Experience Replay** (Fedus et al., ICML 2020) | replay capacity와 **age of oldest policy**를 분리 연구. oldest policy age를 줄이면 성능 향상. | 우리 절단 실험이 정확히 age of oldest policy를 낮춘 것. 기본 프레임이 이미 존재. |
| **Replay Buffer with Local Forgetting** (arXiv 2303.08690) | FIFO 대신 새 관측의 국소 이웃 샘플을 제거 → **local environment change 적응**. Dyna·PlaNet·DreamerV2. | 환경 변화 적응을 위한 buffer forgetting이라는 문제 설정 자체가 선행. |

→ **재정향 불가.** 특히 2605.10236은 우리가 stage 5에서 도달한 결론을 이미
담고 있고 해결책까지 제시함.

### 7b. Periodic re-anchoring 방향 → 혼잡하고 검증도 불리

Priority staleness는 **명명된 기지 문제**이며 해법이 다수 존재:

- **LaBER** (arXiv 2110.01528): 큰 uniform batch에서 priority를 즉석 재계산해
  staleness를 우회.
- **Freshness-Aware PER** (arXiv 2604.16918, 2026): 저장 priority에 **곱셈형
  지수 age decay**를 적용, ESS 분석으로 근거화. (우리 per_prng의 곱셈 수정,
  per_reset_periodic의 freshness 동기를 동시에 선점하는 형태)
- **Imp-PER** (MDPI Appl. Sci. 10(19):6925): 전체 데이터의 실제 TD error 합을
  예측해 update deviation 보정.
- **Investigating the Interplay of Prioritized Replay and Generalization**
  (arXiv 2407.09702): PER 변형들의 완화책이 "**일반적으로 uniform replay를
  능가하지 못한다**"고 보고 — PER 변형 논문 전반에 불리한 선행 결과.

우리 per_reset_periodic(전체 priority 주기적 재계산)의 정확한 동일 방법은
확인 범위에서 특정 논문으로 찾지 못했으나, (1) 문제·동기·해법 공간이 이미
조밀하고, (2) 우리 결과는 Hopper 단일 환경·메커니즘 설명 실패·Walker 일반화
실패 상태이므로 **신규성 주장 근거가 약함.**

### 7c. 종합 판단

buffer-composition 재정향은 **닫혔고**, periodic re-anchoring은 **혼잡한
공간에서 근거가 부족한 상태**. 확장 대신 다음이 현실적:

**살아남은 발표 가치 있는 자산 — Gate 1 음성 결과.**
"연합 학습에서 실패 prototype 공유는 매칭된 noise 대비 이득이 0"이라는 결과는
- 통제군 설계가 엄격함(random centroid / shuffled label / noise-matched,
  동일 checkpoint paired, 5 seed)
- AUROC 0.54로 공유 신호의 무용성을 직접 측정
- federated RL에서 "무엇이 작동하지 않는가"를 다루는 짧은 논문/워크숍
  트랙에 적합

이 방향이 남은 유일한 실질 자산으로 판단됨.

## 재현

```bash
python scripts/gate1.py --seeds 0 1 2 3 4        # base + gate-1 branches
python scripts/pivot.py                          # pivot branches
python scripts/ablation.py                       # stage 1
python scripts/ablation2.py                      # stage 2 (+ diag)
python scripts/ablation3.py                      # stage 2.5
python scripts/ablation4.py                      # stage 3 (ESS-matched)
python scripts/walker.py                         # stage 4 (Walker2d)
python scripts/ablation5.py                      # stage 5 (staleness/adaptive)
python scripts/gate1_analyze.py                  # 분석들
python scripts/pivot_analyze.py
python scripts/ablation_analyze.py
python scripts/ablation2_analyze.py
```
