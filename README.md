# Fluid-Mesh-HPC: Decentralized Autonomous Homeostatic Fluidic Control Infrastructure

A production-grade, deterministically bounded jitter, fault-tolerant distributed intelligence topology designed for real-time macro-flux wave prediction and autonomous fluidic bypass rerouting in high-availability industrial liquid distribution networks and smart fluid infrastructures.

---

## 📌 Core Architectural Breakthroughs

### 1. Deterministic Execution via Hardware MUX Emulation (Layer 1)
Traditional control loops suffer from code branching overheads that trigger CPU pipeline stalls and unpredictable microstructural latency. FluidMesh-HPC introduces a **Branchless Bare-Metal C Kernel** that eliminates logical `if-else` macros in favor of ternary register-level switching. This enforces compilation into hardware conditional move instructions (`CMOV` on x86, `VSEL/FCSEL` on ARM v8), minimizing branch-induced execution time variance (jitter) to a deterministic baseline. Heavy transcendental functions (`exp()`) are bypassed via non-linear high-speed **Padé [1/1] Rational Damping** mapping executed in unified floating-point unit (FPU) clocks.

### 2. Distributed Layer 2 AI Cores with Gradient Isolation Barriers via `jax.lax.stop_gradient`
In massive plant array networks, traditional centralized intelligence models suffer from whole-system failure when a single segment collapses. FluidMesh-HPC deploys **Multiple Independent Layer 2 AI Cores (Sector Sovereigns)** distributed across the network infrastructure, where each core instance is mapped 1:1 to a specific **32-channel edge sensor array**. Upon catastrophic peripheral node damage—where a hardware breach triggers an absolute failure token (`-99.0f`)—the mathematical **Fault-Isolation Accelerator Engine** within that specific sector's core engages instantly. By activating `jax.lax.stop_gradient`, it freezes the backpropagation chain on JAX/XLA graph runtimes locally. This absolute single-direction computational insulation shield ensures that the pretrained sovereign weights of the damaged sector are frozen and protected mathematically, while the remaining operational Layer 2 cores continue active homeostasis across other sectors without a single nanosecond of global latency or cross-contamination.


### 3. Asynchronous Event-Driven "Cold Standby" Routing Topology (Layer 3)
To completely neutralize monolithic CPU bottlenecks and Python's Global Interpreter Lock (GIL), the architecture decouples computational time-scales between edge physics and global governance. The supreme **Layer 3 Global Event Orchestrator** operates asynchronously on a `100ms` epoch loop, acting as an ultra-lightweight **Asynchronous Event Router**. Rather than executing heavy computations over raw 32-channel space-time wave tensors, Layer 3 exclusively listens to single-bit event markers refined and passed upward by the **multiple Layer 2 AI Cores**, suppressing active runtime computational overhead to a strict zero baseline during parity. Upon intercepting an isolated `-99.0f` catastrophic failure token from any damaged Layer 2 sovereign, the orchestrator triggers an immediate hardware interrupt, instantly awakens dormant backup infrastructures from a zero-overhead sleep mode (`Cold Standby`), and injects 32-channel static scalar parameter tensors directly via high-speed DMA register streaming into the target sector to execute instantaneous **Cross-Axis Curl Inversion** rerouting.


---

## 📐 Unified System Topology Map

```text
 [👑 Layer 3: Global Event Orchestrator]  ➔ (Python/C++ Asynchronous Event Router)
                ▲                          - Ultra-lightweight asynchronous governance on a 100ms global epoch.
                │ [Asynchronous Interrupt] - Zero active runtime computational load during operational parity.
                │                          - Exclusively evaluates single-bit event markers refined by Layer 2.
                │
    ┌───────────┴───────────┐ (Gathers telemetry alerts from multiple decentralized zones)
    ▼                       ▼
 [🏰 Sector 01: Layer 2] [🏰 Sector N...: Layer 2] ➔ (JAX / XLA Static Fused Kernel Compiler Backend)
                ▲                       ▲           - Multiple Independent AI Cores acting as Sector Sovereigns.
                │ [1:1 Physical Link]   │           - 32-channel space-time wave predicting PINN intelligence.
                │                       │           - stop_gradient firewall freezes backprop to isolate failures.
                │                       │
 [⛓️ 32-Channel Sensor]   [⛓️ 32-Channel Sensor] ➔ (Pure C99 FPU Register Inlined Processing Matrix)
                                                    - 4-Neighbor 2D Mesh-Grid topology (East/West/North/South).
                                                    - AC-coupled piezoelectric high-frequency edge physics.
                                                    - Zero-Branching deterministic execution with CMOV/FCSEL.
                                                    - Emits instantaneous 3-bit hardware markers: [0.0]/[1.0]/[-99.0f].
```


---

### 🔄 Data Pipeline Sequence

1. **Edge Detection (Layer 1)**: The C99 kernel continuously samples high-frequency vortex oscillations via AC-coupled piezoelectric sensors across 32-channel edge arrays. Upon catastrophic physical failure, it stamps a `-99.0f` absolute fracture token into hardware registers instantly without branch-induced jitter.
2. **Decentralized AI Isolation (Layer 2)**: The specific **Layer 2 AI Core** assigned to that sector ingests the local 32-channel telemetry. Upon detecting the `-99.0f` marker, it immediately flags the malfunctioning segment, activates the `stop_gradient` firewall to mathematically freeze and protect its sovereign weights from gradient explosion, and passes a single-bit refined alert token upward.
3. **Global Mobilization (Layer 3)**: Gaining asynchronous **telemetry alerts** from the compromised **Layer 2 Core**, the supreme **Layer 3 Global Orchestrator** maintains its zero-runtime computational load advantage. It triggers a hardware interrupt, awakens the sleeping `Cold Standby` actuators, and injects parametric constants via high-speed DMA register streaming to initiate instantaneous **Cross-Axis Curl Inversion** bypass rerouting.


```text
  [👑 Layer 3: Global Event Orchestrator] ➔ (FinalEventOrchestrator: Python)
        ▲                                  - Zero runtime active computational load during parity.
        │ [Asynchronous Alert Ingestion]   - Intercepts refined single-bit "Sector Fracture" tokens.
        │ (100ms Event Bus Interrupt)      - Triggers DMA register streaming to mobilize Cold Standby.
        │
  ┌─────┴──────────────────────────────────┐ (Asynchronously gathers alerts from decentralized sectors)
  ▼                                        ▼
  [🏰 Sector 01: Layer 2 AI Core]   ...    [🏰 Sector N: Layer 2 AI Core] ➔ (MasterControlAI: JAX / XLA Fused)
        ▲                                        ▲                         - Multiple independent cores.
        │ [32-Ch Ingestion]                      │ [32-Ch Ingestion]       - Deploys stop_gradient locally.
        │ (1:1 High-Speed Telemetry Links)       │ (1:1 High-Speed Links)  - Insulates sovereign weights.
        │                                        │
  [⛓️ Layer 1: 32-Ch Sensor Rail]          [⛓️ Layer 1: 32-Ch Sensor Rail] ➔ (fluid_mesh_cell32: Pure C99)
                                                                           - High-frequency AC-piezoelectric.
                                                                           - Branchless register switching.
                                                                           - Emits markers: [0.0]/[1.0]/[-99.0f].
```

---

## ⚡ Mathematical & Structural Proof of Concept

### Cross-Axis Curl Inversion & Auto-Rerouting (Matrix-Free Solvers)
Rather than executing compute-intensive partial differential equation (PDE) Navier-Stokes linear algebraic matrices over the entire global footprint, FluidMesh evaluates localized tensor fields dynamically. This matrix-free evaluation is executed independently across **multiple decentralized sectors supervised by their respective Layer 2 AI Cores**, evaluating high-frequency 4-directional physical lines on each 32-channel edge array:

$$\text{Spatial Gradient } U = \text{East} - \text{West}$$
$$\text{Spatial Gradient } V = \text{North} - \text{South}$$

Upon the physical fracture or catastrophic anomaly of a single quadrant (e.g., $\text{North} \to -99.0\text{f}$), the internal C-Kernel edge logic instantly intercepts the fault boundary, mapping $\text{North} \to 0.0$. The localized cross-axis compiler mask inverses the operator dynamically within that specific Layer 2 sector runtime:

$$V_{\text{fault}} = -\text{Spatial Gradient } V \implies + \text{South}$$

This mathematical cross-inversion generates an autonomous rotational Curl vector without any telemetry command intervention or centralized scaling bottlenecks from the Layer 3 server. By harnessing the fluid's own residual momentum and kinetic energy as the computational driver, it diverts fluid velocity vectorially towards predefined physical bypass pipelines within strict microsecond bounds.


---

## 🔍 Implementation Notes & Known Constraints

* **Decentralized JAX Static Tracing Warm-up**: Due to the nature of JAX/XLA's ahead-of-time (AOT) compilation, the first computational epoch experiences a non-deterministic compilation overhead. In high-availability production environments, the Layer 3 orchestrator must trigger a concurrent, dummy warm-up trace across **all decentralized Layer 2 AI Core instances simultaneously** before live fluid ingestion. This guarantees that every sector's sovereign weights are pre-compiled into static machine code tracks matching the strict `num_segments` parameter, completely locking zero-latency response pipelines prior to system engagement.
* **Compiler Flags Enforcement**: To ensure that branchless registers are not optimized out by the compiler, the Bare-Metal C Kernel requires specific optimization flags (`-O3 -fno-fast-math`). Relying on `-Ofast` may violate IEEE 754 precision constraints required for absolute `-99.0f` catastrophic failure token evaluation.
* **AC-Coupled Sensory Ingestion (Vortex Tracking)**: To prevent static signaling lock (where continuous flow forces a raw signal to remain at `1.0f`), the infrastructure must be deployed using **AC-coupled piezoelectric sensors** or **piezoelectric vortex shedding flowmeters** arranged in the 4-neighbor grid. The oscillating frequency emitted by these dynamic physical elements perfectly resonates with the kernel's `phi0 / phi1` state estimation loop (where `raw_flow_signal - phi0_pred` naturally attenuates static baselines to `0.0` while capturing high-frequency micro-vortex wavefronts without a single logical `if` branch).


---

## 🔒 Defensive Patent Notice (GNU GPLv3 Enforced)

This software repository is published under the **GNU General Public License v3**. This codebase serves as an official, un-expungable **Defensive Prior Art Registration**. Any attempt by centralized enterprise entities or patent monetization agents to enclose, clone, privatize, or restrict these specific structural topographies, JAX isolation gates, or branchless C-MUX emulation mechanisms is legally prohibited under international copyleft patent grant clauses.


---

# Fluid-Mesh-HPC: 분산형 자율 항상성 유체 제어 인프라

고가용성 산업용 액체 분배 네트워크 및 스마트 유체 인프라에서 실시간 거시 유동 파동 예측과 자율 유체 바이패스 우회 라우팅을 수행하기 위해 설계된 프로덕션 등급의 고정 지터(Deterministic Jitter) 및 결함 허용(Fault-Tolerant) 분산 지능 토폴로지입니다.

---

### 1. 하드웨어 MUX 에뮬레이션을 통한 결정론적 실행 (Layer 1)
전통적인 제어 루프는 CPU 파이프라인 스톨과 예측 불가능한 미세 지연을 유발하는 코드 분기(Branching) 오버헤드로 인해 취약성을 갖습니다. FluidMesh-HPC는 논리적 `if-else` 매크로를 배제하고 레지스터 수준의 삼항 연산 스위칭을 사용하는 **분기 없는 베어메탈 C 커널(Branchless Bare-Metal C Kernel)**을 도입합니다. 이는 하드웨어 조건부 이동 명령어(x86의 `CMOV`, ARM v8의 `VSEL/FCSEL`)로의 컴파일을 강제하여, 분기로 인한 실행 시간 편차(지터)를 결정론적 베이스라인 수준으로 압축합니다. 연산 부하가 큰 초월함수(`exp()`)는 통합 부동소수점 연산 장치(FPU) 클럭 내에서 실행되는 고속 비선형 **Padé [1/1] 유리함수 감쇠(Rational Damping)** 매핑을 통해 우회 처리됩니다.

### 2. `jax.lax.stop_gradient` 기반 그라디언트 고립 방화벽을 탑재한 다중 분산 레이어 2 AI 코어 (Layer 2)
거대 플랜트 어레이 네트워크 환경에서는 단일 세그먼트가 붕괴할 때 중앙 집중형 지능 모델 전체가 마비되는 치명적인 취약점을 가집니다. FluidMesh-HPC는 그리드 전역에 독립적으로 분산된 **다수의 레이어 2 AI 코어(구역 영주)**를 배포하며, 각 코어 인스턴스는 특정 **32채널 말단 센서 어레이와 1:1로 매핑**됩니다. 물리적 대파열 및 배관 파손을 뜻하는 절대적 결함 토큰(`-99.0f`)이 말단에서 감지되는 즉시, 해당 구역을 전담하는 레이어 2 코어 내부의 수학적 **결함 고립 가속 엔진(Mathematical Fault-Isolation Accelerator Engine)**이 광속으로 가동됩니다. `jax.lax.stop_gradient`를 발동시켜 해당 국소 그래프 상의 역전파 체인을 즉각 동결함으로써, 카타스트로픽 하드웨어 손상이 발생하더라도 파손된 구역의 고유 가중치(Sovereign weights) 자산을 수학적으로 완벽히 격리 보호합니다. 이와 동시에, 파괴되지 않은 나머지 다수의 레이어 2 코어들은 다른 구역의 자율 항상성 제어를 단 1나노초의 전역 레이턴시 병목이나 상호 오염 없이 독립적으로 지속합니다.

### 3. 이벤트 기반 "콜드 스탠바이" 라우팅 토폴로지 (Layer 3)
단일 CPU 보틀넥과 파이썬의 글로벌 인터프리터 락(GIL) 한계를 무력화하기 위해, 본 아키텍처는 말단 물리 현상과 전역 제어의 계산 시간 스케일을 완전히 분리했습니다. 최상단 **레이어 3 전역 오케스트레이터 사령탑**은 `100ms` 에포크 루프로 비동기 작동하며 가벼운 **비동기 이벤트 라우터(Asynchronous Event Router)** 역할만 수행합니다. 사령탑은 로우 데이터인 32채널 시공간 파동 텐서 전체를 직접 연산하지 않고, 하단의 **다수 레이어 2 AI 코어들이 최종 정제하여 쏘아 올린 단일 비트 이벤트 마커만 수신**함으로써 정상 가동 시 전역 계산 부하를 strict 제로(0) 베이스라인으로 무력화합니다. 특정 레이어 2 영주로부터 파손 속보(`-99.0f`)를 포착하는 즉시 하드웨어 인터럽트를 발생시켜 잠들어 있던 예비 관로(**Cold Standby**) 자율 우회 인스턴스를 깨우고, 고속 DMA 레지스터 스트리밍을 통해 32채널 정적 스칼라 파라미터 텐서를 직분사하여 **교차축 컬 반전(Cross-Axis Curl Inversion)** 우회 라우팅을 오차 없이 기동합니다.


---

## 📐 통합 시스템 토폴로지 맵

```text
 [👑 계층 3: 전역 오케스트레이터 사령탑] ➔ (Python/C++ 비동기 이벤트 라우터)
                ▲                          - 100ms 글로벌 에포크 밸런싱 및 모니터링 대시보드
                │ [비동기 얼럿 인터럽트]   - 정상 가동 시 런타임 계산 부하 제로(0) 유지
                │                          - 레이어 2가 최종 정제해 올린 단일 비트 마커만 평가
                │
    ┌───────────┴───────────┐ (전국 분산 구역들로부터 비동기식 고장 속보 취합)
    ▼                       ▼
 [🏰 섹터 01: 레이어 2]  [🏰 섹터 N...: 레이어 2] ➔ (JAX / XLA 정적 퓨즈드 커널 컴파일러 백엔드)
                ▲                       ▲           - 다수의 독립된 AI 코어가 구역 영주로 분산 배치
                │ [1:1 하드웨어 링크]   │           - 32채널 시공간 파동 예측 PINN(물리 기반 신경망) 지능
                │                       │           - stop_gradient 방화벽 가동으로 국소 결함 즉각 격리
                │                       │
 [⛓️ 32채널 센서 레일]    [⛓️ 32채널 센서 레일]   ➔ (순수 C99 FPU 레지스터 인라인 프로세싱 매트릭스)
                                                    - 4방향 2D 메쉬 그리드 토폴로지 (East/West/North/South)
                                                    - AC 결합형 압전 와류 센서 고주파 말단 물리 감지
                                                    - 삼항 연산 기반 CMOV/FCSEL 분기 없는 결정론적 실행
                                                    - 즉각적인 3비트 하드웨어 마커 스트리밍: [0.0]/[1.0]/[-99.0f]
```


---

### 🔄 데이터 흐름 및 판단 체인 (Data Pipeline Sequence)

1. **말단 감지 (Layer 1)**: AC 결합 압전 센서가 32채널 말단 어레이 전역에서 고주파 와류 진동을 초고속 샘플링합니다. 물리적 대파열 및 관로 파손을 감지하는 즉시, 분기문 없이 하드웨어 레지스터에 `-99.0f` 절대 결함 마커를 실시간 0ns 지터 수준으로 남깁니다.
2. **분산형 AI 정제 및 격리 (Layer 2)**: 해당 구역을 전담하는 특정 **레이어 2 AI 코어(구역 영주)**가 국소적인 32채널 텔레메트리 데이터를 흡수합니다. `-99.0f` 마커 포착 즉시 정확히 몇 번 세그먼트가 터졌는지 판정하고, `stop_gradient` 방화벽을 쳐서 자기 두뇌(고유 가중치)를 그라디언트 폭발로부터 수학적으로 완벽히 격리한 뒤, 최종 정제된 단일 비트 알럿 토큰만 최상단으로 토스합니다.
3. **사령탑 최종 개입 (Layer 3)**: 결함이 발생한 **레이어 2 코어**로부터 비동기 고장 속보를 받은 최상단 **레이어 3 전역 오케스트레이터 사령탑**은 평상시 연산 부하 제로(0)의 이점을 그대로 유지합니다. 즉시 하드웨어 인터럽트를 발생시켜 잠들어 있던 예비 관로(**Cold Standby**) 액추에이터를 즉각 기상시키고, 고속 DMA 레지스터 스트리밍으로 제어 상수를 직분사하여 마이크로초 단위의 **교차축 컬 반전(Cross-Axis Curl Inversion)** 자율 우회 라우팅을 마감합니다.

```
```text
  [👑 계층 3: 전역 오케스트레이터 사령탑] ➔ (FinalEventOrchestrator: Python)
        ▲                                  - 평상시 계산 부하 제로(0), 극단적 자원 절약 대기
        │ [비상 속보 비동기 접수]          - AI 코어로부터 '구역 파손' 확정 신호 비동기 수신
        │ (100ms Event Bus Interrupt)      - 💤 잠들어 있는 예비 관로(Cold Standby)를 즉시 기상
        │
  ┌─────┴──────────────────────────────────┐ (다수의 분산 구역들로부터 비동기식 고장 속보 수렴)
  ▼                                        ▼
  [🏰 섹터 01: 레이어 2 AI 코어]    ...    [🏰 섹터 N: 레이어 2 AI 코어] ➔ (MasterControlAI: JAX Fused)
        ▲                                        ▲                         - 다수의 독립 영주 분산 배치
        │ [32채널 데이터 흡수]                   │ [32채널 데이터 흡수]    - stop_gradient 방화벽 가동
        │ (1:1 High-Speed Telemetry)             │ (1:1 High-Speed Links)  - 고유 가중치 수학적 완벽 격리
        │                                        │
  [⛓️ 계층 1: 32채널 센서 레일]            [⛓️ 계층 1: 32채널 센서 레일]  ➔ (fluid_mesh_cell32: Pure C99)
                                                                           - AC 결합 압전 고주파 샘플링
                                                                           - 삼항 연산 기반 분기 없는 실행
                                                                           - 마커 생성: [0.0]/[1.0]/[-99.0f]
```


---

## ⚡ Mathematical & Structural Proof of Concept (수학적 & 구조적 개념 검증)

### 교차축 컬 반전 및 자율 우회 라우팅 (행렬 프리 솔버)
전체 시스템 영역에 걸쳐 계산 집약적인 수치해석용 나비에-스토크스(Navier-Stokes) 선형 대수 행렬을 푸는 대신, FluidMesh는 전역 인프라에 물리적 레이턴시 병목을 주지 않도록 국소 텐서 필드를 동적으로 평가합니다. 이 행렬 프리(Matrix-Free) 평가는 **다수의 레이어 2 AI 코어(구역 영주)들이 독립 통치하는 각각의 분산 세그먼트 단에서 독자적으로 집행**되며, 각 32채널 격자점의 사방(4방향) 물리 감지선 데이터를 기반으로 작동합니다.

$$\text{공간 그라디언트 } U = \text{East} - \text{West}$$
$$\text{공간 그라디언트 } V = \text{North} - \text{South}$$

특정 쿼드런트 배선에 배관 대파열이나 물리적 단선 사고가 발생하여 (예: $\text{North} \to -99.0\text{f}$), 내부 C 커널의 말단 엣지 로직이 결함 경계를 포착하면 즉시 $\text{North} \to 0.0$으로 매핑합니다. 동시에 해당 레이어 2 런타임 내의 국소적 교차축 컴파일러 마스크가 연산자를 동적으로 반전시킵니다.

$$V_{\text{fault}} = -\text{공간 그라디언트 } V \implies + \text{South}$$

이 수학적 교차 반전은 최상단 레이어 3 서버의 원격 제어 명령이나 대규모 행렬 연산 대기 시간 없이, **밀려오고 있던 유체 고유의 관성과 유동 역학 에너지 자체를 연산의 동력(Driver)으로 삼아** 자율적인 회전 컬(Curl) 벡터를 생성합니다. 이를 통해 유체 속도 벡터를 사전에 정의된 물리적 바이패스 파이프라인으로 마이크로초(µs) 임베디드 결정성 바운드 이내에 광속 자율 우회시킵니다.


---

## 🔍 구현 참고 사항 및 제약 조건

* **분산형 JAX 정적 트레이싱 웜업(Warm-up)**: JAX/XLA의 사전(AOT) 컴파일 특성상, 최초 계산 에포크 실행 시 비결정론적인 컴파일 오버헤드가 발생합니다. 고가용성이 요구되는 프로덕션 환경에서는 라이브 데이터를 투입하기 전에 최상단 레이어 3 오케스트레이터의 사령 하에 **전국에 분산 배포된 다수의 레이어 2 AI 코어 인스턴스 전체를 대상으로 동시에 병렬 더미 트레이싱(Concurrent Dummy Trace)을 수행**하여 시스템을 완전 예열해야 합니다. 이 예열 시퀀스를 통해 각 구역의 고유 가중치(Sovereign weights)가 설정된 채널 크기(`num_segments`) 파라미터에 맞춰 기기 정적 기계어 코드로 완전히 박제되며, 실전 유체 흡수 전 전역 컴파일 렉 병목을 원천 완봉합니다.
* **컴파일러 플래그 강제**: 분기 없는 레지스터 최적화가 컴파일러에 의해 임의로 해제되거나 변경되는 것을 막기 위해, 베어메탈 C 커널 빌드 시 특정 최적화 플래그(`-O3 -fno-fast-math`)를 반드시 명시해야 합니다. `-Ofast`에 의존할 경우, `-99.0f` 결함 토큰 평가에 필요한 IEEE 754 정밀도 제약 조건이 깨질 수 있습니다.
* **AC 결합형 센서 동적 흡수(와류 추적)**: 연속적인 유동으로 인해 로우(Raw) 신호가 1.0f로 고정되는 신호 잠금(Lock) 현상을 방지하기 위해, 본 인프라는 4-이웃 그리드 배열에 **AC 결합형 압전 센서(Piezoelectric)** 또는 **압전 와류(Vortex) 유량계 소자**를 배치해야 합니다. 하드웨어 소자가 뿜어내는 출렁이는 주파수는 C 커널의 `phi0 / phi1` 상태 추정 루프와 완벽하게 공진합니다. (`raw_flow_signal - phi0_pred` 수식을 통해) 단 하나의 if 분기문 없이도, 정적 베이스라인은 0.0으로 자동 소멸시키고 고주파 미세 와류의 앞부분(Wavefront) 펄스만 광속으로 낚아챕니다.

---

## 🔒 방어적 특허 공지 (GNU GPLv3 적용)

본 소프트웨어 리포지토리는 **GNU General Public License v3**에 따라 배포됩니다. 본 소스코드와 아키텍처는 공식적이고 말소 불가능한 **방어적 선행기술 등록(Defensive Prior Art Registration)** 자격을 갖춥니다. 거대 기업이나 특허 괴물(Patent Troll)이 본 기술 고유의 구조적 토폴로지, JAX 격리 게이트, 또는 분기 없는 C-MUX 에뮬레이션 메커니즘을 독점, 복제, 사유화하거나 권리를 제한하려는 모든 시도는 국제 카피레프트 특허 허가 조항에 따라 법적으로 금지됩니다.
