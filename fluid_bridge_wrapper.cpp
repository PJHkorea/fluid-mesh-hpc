/**
 * @file fluid_bridge_wrapper.cpp
 * @brief Zero-Copy High-Speed pybind11 Bridge Connecting Layer 1 C-Kernel Memory to Layer 2 JAX Backend
 * @license GNU GPLv3 Enforced (Defensive Prior Art Registration)
 * @author PJHkorea (The Sovereign Architect)
 *
 * [🔒 CROSS-DOMAIN ARCHITECTURAL SYNERGY]
 * - 본 0ns 제로카피 메모리 바인딩 브릿지는 자매 저장소인 Quantum-Mesh-QEC v2 (Apache 2.0)의
 *   quantum_bridge_wrapper.cpp 아키텍처와 상호 교차 리엔지니어링된 핵심 자산입니다.
 * - 본 코드를 타 도메인에 독점 상용(Proprietary) 및 폐쇄형 아키텍처로 무단 유용하거나
 *   특허를 사유화하려는 모든 시도는 GNU GPLv3 조항에 의거하여 소스코드 전면 공개 의무에 귀속됩니다.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "fluid_mesh_baremetal_core_v2.h"

namespace py = pybind11;

/**
 * @brief [KR] L1 캐시라인에 정렬된 하드웨어 레지스터 주소를 복사 없이 파이썬 JAX/NumPy로 직송
 *        [EN] Intercepts and exposes the 32-Byte aligned raw fluid registry to Python without memory copy.
 */
py::array_t<float> extract_peripheral_fluid_buffer(uintptr_t struct_raw_ptr) {
    // 1. 호스트 메모리나 FPGA DMA 레지스터에서 전달된 포인터 주소를 유체 세포 구조체 타입으로 직접 형변환 (0ns)
    FluidCell32* self = reinterpret_cast<FluidCell32*>(struct_raw_ptr);
    
    // 2. 국소 밀도(phi0)와 속도 그래디언트(phi1)가 시작되는 메모리 헤드 포인터 주소를 돋보기로 가리킴
    float* fluid_data_head_ptr = &(self->phi0);

    // 3. 📌 THE PYTHON GC BYPASS: 파이썬 가비지 컬렉터의 메모리 임의 해제 가드를 무력화하기 위한 빈 델리터 캡슐 설계
    py::capsule buffer_handle(fluid_data_head_ptr, [](void* p) {
        // [KR] 하드웨어 레지스터 생명 주기는 C 베어메탈 단에서 수동 관리하므로 파이썬이 해제하지 못하도록 밀봉
        // [EN] Memory lifecycle is manually governed by the C bare-metal layer; blocks Python GC inference.
    });

    // 4. 단 1바이트의 메모리 할당이나 데이터 복사 오버헤드 없이 0ns 만에 파이썬 배열 뷰 생성
    return py::array_t<float>(
        {2},              // Shape: [phi0 (Density), phi1 (Velocity Gradient)]
        {sizeof(float)},  // Strides: 32비트 float 단위 보폭 설정
        fluid_data_head_ptr,
        buffer_handle
    );
}

PYBIND11_MODULE(fluid_bridge_wrapper, m) {
    m.doc() = "Zero-Copy High-Speed Hardware Register Memory Binding Wrapper for FluidMesh-HPC V2";
    m.def("extract_peripheral_fluid_buffer", &extract_peripheral_fluid_buffer, 
          "Extracts raw hardware fluid cells with strict 0ns pointer bypass allocation");
}
