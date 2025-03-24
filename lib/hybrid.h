// Copyright 2019 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef HYBRID_H_
#define HYBRID_H_

#include <algorithm>
#include <array>
#include <complex>
#include <vector>

#include "gate.h"
#include "gate_appl.h"

namespace qsim {

/**
 * Hybrid Feynman-Schrodinger simulator.
 */
template <typename IO, typename GateT,
          template <typename, typename> class FuserT, typename For>
struct HybridSimulator final {
 public:
  using Gate = GateT;
  using GateKind = typename Gate::GateKind;
  using fp_type = typename Gate::fp_type;

 private:
  // Note that one can use "struct GateHybrid : public Gate {" in C++17.
  struct GateHybrid {
    using GateKind = HybridSimulator::GateKind;
    using fp_type = HybridSimulator::fp_type;

    GateKind kind;
    unsigned time;
    std::vector<unsigned> qubits;
    std::vector<unsigned> controlled_by;
    uint64_t cmask;
    std::vector<fp_type> params;
    Matrix<fp_type> matrix;
    bool unfusible;
    bool swapped;

    const Gate* parent;
    unsigned id;

    /**
     * Which parts the GateHybrid is in.
     * -1 on second index indicates there is only one part.
     * Order of cuts is 0t1, 1t2, 2t0.
     * Observe. Two qubits in part p gives {p, -1}
     */
    std::array<signed,2> parts;
  };

  struct GateX {
    // The part of decomposed1 is the part of decomposed0 + 1 (mod3)
    GateHybrid* decomposed0;
    GateHybrid* decomposed1;
    schmidt_decomp_type<fp_type> schmidt_decomp;
    unsigned schmidt_bits;
    unsigned part_q0;
    unsigned part_q1;
  };

 public:
  using Fuser = FuserT<IO, GateHybrid>;
  using GateFused = typename Fuser::GateFused;

  /**
   * Contextual data for hybrid simulation.
   */
  struct HybridData {
    /**
     * List of gates on part "0".
     */
    std::vector<GateHybrid> gates0;
    /**
     * List of gates on part "1".
     */
    std::vector<GateHybrid> gates1;
    /**
     * List of gates on part "2".
     */
    std::vector<GateHybrid> gates2;
    /**
     * List of gates on cut 0t1.
     */
    std::vector<GateX> gatexs0t1;
    /**
     * List of gates on cut 1t2.
     */
    std::vector<GateX> gatexs1t2;
    /**
     * List of gates on cut 2t0.
     */
    std::vector<GateX> gatexs2t0;
    /**
     * Global qubit index to local qubit index map.
     */
    std::vector<unsigned> qubit_map;
    /**
     * Number of qubits on part "0".
     */
    unsigned num_qubits0;
    /**
     * Number of qubits on part "1".
     */
    unsigned num_qubits1;
    /**
     * Number of qubits on part "2".
     */
    unsigned num_qubits2;
    /**
     * Number of gates on the cut 0t1.
     */
    unsigned num_gatexs0t1;
    /**
     * Number of gates on the cut 1t2.
     */
    unsigned num_gatexs1t2;
    /**
     * Number of gates on the cut 2t0.
     */
    unsigned num_gatexs2t0;
  };

  /**
   * User-specified parameters for gate fusion and hybrid simulation.
   */
  struct Parameter : public Fuser::Parameter {
    /**
     * Fixed bitstring indicating values to assign to Schmidt decomposition
     * indices of prefix gates.
     */
    uint64_t prefix0t1;
    uint64_t prefix1t2;
    uint64_t prefix2t0;
    /**
     * Number of gates on all cuts that are part of the prefix. Indices of these
     * gates are assigned the value indicated by `prefix` for the specific cut.
     */
    unsigned num_prefix_gatexs;
    /**
     * Number of gates on all cuts that are part of the root. All gates that are
     * not part of the prefix or root are part of the suffix.
     */
    unsigned num_root_gatexs;
    unsigned num_threads;
  };

  template <typename... Args>
  explicit HybridSimulator(Args&&... args) : for_(args...) {}

  /**
   * Splits the lattice into three parts, using Schmidt decomposition for gates
   * on the cut.
   * @param parts Lattice sections to be simulated.
   * @param gates List of all gates in the circuit.
   * @param hd Output data with split parts.
   * @return True if the splitting done successfully; false otherwise.
   */
  static bool SplitLattice(const std::vector<unsigned>& parts,
                           const std::vector<Gate>& gates, HybridData& hd) {
    hd.num_gatexs0t1 = 0;
    hd.num_gatexs1t2 = 0;
    hd.num_gatexs2t0 = 0;
    hd.num_qubits0 = 0;
    hd.num_qubits1 = 0;
    hd.num_qubits2 = 0;

    hd.gates0.reserve(gates.size());
    hd.gates1.reserve(gates.size());
    hd.gates2.reserve(gates.size());
    hd.qubit_map.reserve(parts.size());

    unsigned count0 = 0;
    unsigned count1 = 0;
    unsigned count2 = 0;

    // Global qubit index to local qubit index map.
    for (std::size_t i = 0; i < parts.size(); ++i) {
      switch (parts[i]) {
        case 0:
          ++hd.num_qubits0;
          hd.qubit_map.push_back(count0++);
          break;
        case 1:
          ++hd.num_qubits1;
          hd.qubit_map.push_back(count1++);
          break;
        case 2:
          ++hd.num_qubits2;
          hd.qubit_map.push_back(count2++);
      }
    }

    // Split the lattice.
    for (const auto& gate : gates) {
      if (gate.kind == gate::kMeasurement) {
        IO::errorf("measurement gates are not suported by qsimh.\n");
        return false;
      }

      if (gate.controlled_by.size() > 0) {
        IO::errorf("controlled gates are not suported by qsimh.\n");
        return false;
      }

      switch (gate.qubits.size()) {
      case 1:  // Single qubit gates.
        switch (parts[gate.qubits[0]]) {
        case 0:
          hd.gates0.emplace_back(GateHybrid{gate.kind, gate.time,
            {hd.qubit_map[gate.qubits[0]]}, {}, 0, gate.params, gate.matrix,
            false, false, nullptr, 0, {0,-1}});
          break;
        case 1:
          hd.gates1.emplace_back(GateHybrid{gate.kind, gate.time,
            {hd.qubit_map[gate.qubits[0]]}, {}, 0, gate.params, gate.matrix,
            false, false, nullptr, 0, {1,-1}});
          break;
        case 2:
          hd.gates2.emplace_back(GateHybrid{gate.kind, gate.time,
            {hd.qubit_map[gate.qubits[0]]}, {}, 0, gate.params, gate.matrix,
            false, false, nullptr, 0, {2,-1}});
          break;
        }
        break;
      case 2:  // Two qubit gates.
        {
          if (parts[gate.qubits[0]] == parts[gate.qubits[1]]) {
            // Both qubits in the same part.
            switch (parts[gate.qubits[0]]) {
            case 0:
              hd.gates0.emplace_back(GateHybrid{gate.kind, gate.time,
                {hd.qubit_map[gate.qubits[0]], hd.qubit_map[gate.qubits[1]]},
                {}, 0, gate.params, gate.matrix, false, gate.swapped,
                nullptr, 0, {0,-1}});
              break;
            case 1:
              hd.gates1.emplace_back(GateHybrid{gate.kind, gate.time,
                {hd.qubit_map[gate.qubits[0]], hd.qubit_map[gate.qubits[1]]},
                {}, 0, gate.params, gate.matrix, false, gate.swapped,
                nullptr, 0, {1,-1}});
              break;
            case 2:
              hd.gates2.emplace_back(GateHybrid{gate.kind, gate.time,
                {hd.qubit_map[gate.qubits[0]], hd.qubit_map[gate.qubits[1]]},
                {}, 0, gate.params, gate.matrix, false, gate.swapped,
                nullptr, 0, {2,-1}});
            }
          }
          else {
            // The qubits are in different parts.
            switch (parts[gate.qubits[0]]) {
            case 0:
              switch (parts[gate.qubits[1]]) {
              case 1:
                hd.gates0.emplace_back(GateHybrid{GateKind::kDecomp, gate.time,
                  {hd.qubit_map[gate.qubits[0]]}, {}, 0, gate.params, {},
                  true, gate.swapped, &gate, hd.num_gatexs0t1, {0,1}});
                hd.gates1.emplace_back(GateHybrid{GateKind::kDecomp, gate.time,
                  {hd.qubit_map[gate.qubits[1]]}, {}, 0, gate.params, {},
                  true, gate.swapped, &gate, hd.num_gatexs0t1, {0,1}});
                ++hd.num_gatexs0t1;
                break;
              case 2:
                hd.gates0.emplace_back(GateHybrid{GateKind::kDecomp, gate.time,
                  {hd.qubit_map[gate.qubits[0]]}, {}, 0, gate.params, {},
                  true, gate.swapped, &gate, hd.num_gatexs2t0, {2,0}});
                hd.gates2.emplace_back(GateHybrid{GateKind::kDecomp, gate.time,
                  {hd.qubit_map[gate.qubits[1]]}, {}, 0, gate.params, {},
                  true, gate.swapped, &gate, hd.num_gatexs2t0, {2,0}});
                ++hd.num_gatexs2t0;
              }
              break;
            case 1:
              switch (parts[gate.qubits[1]]) {
              case 0:
                hd.gates1.emplace_back(GateHybrid{GateKind::kDecomp, gate.time,
                  {hd.qubit_map[gate.qubits[0]]}, {}, 0, gate.params, {},
                  true, gate.swapped, &gate, hd.num_gatexs0t1, {0,1}});
                hd.gates0.emplace_back(GateHybrid{GateKind::kDecomp, gate.time,
                  {hd.qubit_map[gate.qubits[1]]}, {}, 0, gate.params, {},
                  true, gate.swapped, &gate, hd.num_gatexs0t1, {0,1}});
                ++hd.num_gatexs0t1;
                break;
              case 2:
                hd.gates1.emplace_back(GateHybrid{GateKind::kDecomp, gate.time,
                  {hd.qubit_map[gate.qubits[0]]}, {}, 0, gate.params, {},
                  true, gate.swapped, &gate, hd.num_gatexs1t2, {1,2}});
                hd.gates2.emplace_back(GateHybrid{GateKind::kDecomp, gate.time,
                  {hd.qubit_map[gate.qubits[1]]}, {}, 0, gate.params, {},
                  true, gate.swapped, &gate, hd.num_gatexs1t2, {1,2}});
                ++hd.num_gatexs1t2;
              }
              break;
            case 2:
              switch (parts[gate.qubits[1]]) {
              case 0:
                hd.gates2.emplace_back(GateHybrid{GateKind::kDecomp, gate.time,
                  {hd.qubit_map[gate.qubits[0]]}, {}, 0, gate.params, {},
                  true, gate.swapped, &gate, hd.num_gatexs2t0, {2,0}});
                hd.gates0.emplace_back(GateHybrid{GateKind::kDecomp, gate.time,
                  {hd.qubit_map[gate.qubits[1]]}, {}, 0, gate.params, {},
                  true, gate.swapped, &gate, hd.num_gatexs2t0, {2,0}});
                ++hd.num_gatexs2t0;
                break;
              case 1:
                hd.gates2.emplace_back(GateHybrid{GateKind::kDecomp, gate.time,
                  {hd.qubit_map[gate.qubits[0]]}, {}, 0, gate.params, {},
                  true, gate.swapped, &gate, hd.num_gatexs1t2, {1,2}});
                hd.gates1.emplace_back(GateHybrid{GateKind::kDecomp, gate.time,
                  {hd.qubit_map[gate.qubits[1]]}, {}, 0, gate.params, {},
                  true, gate.swapped, &gate, hd.num_gatexs1t2, {1,2}});
                ++hd.num_gatexs1t2;
              }
            }
          }
        }
        break;
      default:
        IO::errorf("multi-qubit gates are not suported by qsimh.\n");
        return false;
      }
    }

    auto compare = [](const GateHybrid& l, const GateHybrid& r) -> bool {
      return l.time < r.time || (l.time == r.time &&
          (l.parent < r.parent || (l.parent == r.parent && l.id < r.id)));
    };

    // Sort gates.
    std::sort(hd.gates0.begin(), hd.gates0.end(), compare);
    std::sort(hd.gates1.begin(), hd.gates1.end(), compare);
    std::sort(hd.gates2.begin(), hd.gates2.end(), compare);

    hd.gatexs0t1.reserve(hd.num_gatexs0t1);
    hd.gatexs1t2.reserve(hd.num_gatexs1t2);
    hd.gatexs2t0.reserve(hd.num_gatexs2t0);

    // Get Schmidt matrices 0t1 and 2t0
    // Reasoning: We create every XGate we can. Otherwise we need to loop part 0 again.
    for (auto& gate0 : hd.gates0) {
      if (gate0.parent != nullptr) {
        auto d = GetSchmidtDecomp(gate0.parent->kind, gate0.parent->params);
        if (d.size() == 0) {
          IO::errorf("no Schmidt decomposition for gate kind %u.\n",
                     gate0.parent->kind);
          return false;
        }

        unsigned schmidt_bits = SchmidtBits(d.size());
        if (schmidt_bits > 2) {
          IO::errorf("Schmidt rank is too large for gate kind %u.\n",
                     gate0.parent->kind);
          return false;
        }

        // TODO: Test the swapping.
        unsigned part_q0 = parts[gate0.parent->qubits[0]];
        unsigned part_q1 = parts[gate0.parent->qubits[1]];

        // Other part than 0
        unsigned other_part;
        if (part_q0 == 0) {
          other_part = part_q1;
        }
        else {
          other_part = part_q0;
        }

        // If gate is swapped we swap the parts.
        if (gate0.parent->swapped) {
          std::swap(part_q0, part_q1);
        }

        if (other_part == 1) {
          // Suggestion of generalization: if other part > this part (mod)
          hd.gatexs0t1.emplace_back(GateX{&gate0, nullptr, std::move(d),
                                     schmidt_bits, part_q0, part_q1});
          auto x = hd.gatexs0t1;
          auto y = hd.gatexs0t1;
        }
        else {
          hd.gatexs2t0.emplace_back(GateX{nullptr, &gate0, std::move(d),
                                     schmidt_bits, part_q0, part_q1});
        }
      }
    }

    // Get Schmidt matrices 1t2
    // and adding gate from part 1 to previously created 0t1 GateX.
    unsigned count0t1 = 0;
    for (auto& gate1 : hd.gates1) {
      if (gate1.parent != nullptr) {

        unsigned part_q0 = parts[gate1.parent->qubits[0]];
        unsigned part_q1 = parts[gate1.parent->qubits[1]];

        //The part that is not 1;
        unsigned other_part;
        if (part_q0 == 1) {
          other_part = part_q1;
        }
        else {
          other_part = part_q0;
        }

        if (other_part == 0) {
          hd.gatexs0t1.at(count0t1++).decomposed1 = &gate1;
          continue;
        }
        // other_part == 2
        auto d = GetSchmidtDecomp(gate1.parent->kind, gate1.parent->params);
        if (d.size() == 0) {
          IO::errorf("no Schmidt decomposition for gate kind %u.\n",
                     gate1.parent->kind);
          return false;
        }

        unsigned schmidt_bits = SchmidtBits(d.size());
        if (schmidt_bits > 2) {
          IO::errorf("Schmidt rank is too large for gate kind %u.\n",
                     gate1.parent->kind);
          return false;
        }

        // If gate is swapped we switch the parts.
        if (gate1.parent->swapped) {
            std::swap(part_q0, part_q1);
        }

        hd.gatexs1t2.emplace_back(GateX{&gate1, nullptr, std::move(d),
                                     schmidt_bits, part_q0, part_q1});
      }
    }

    unsigned count1t2 = 0;
    unsigned count2t0 = 0;
    // Adding gate from part 2 to previously created 1t2 and 2t0 GateX.
    for (auto& gate2 : hd.gates2) {
      // nullptr check not really necessary
      if (gate2.parent != nullptr) {
        if (gate2.parts[0] == 1 && gate2.parts[1] == 2) {
          hd.gatexs1t2[count1t2++].decomposed1 = &gate2;
        }
        else if (gate2.parts[0] == 2 && gate2.parts[1] == 0) {
          hd.gatexs2t0[count2t0++].decomposed0 = &gate2;
        }
        else {
          IO::errorf("GateHybrid in part 2 with wrong connections\n");
        }
      }
    }

    for (auto& gatex : hd.gatexs0t1) {
      if (gatex.schmidt_decomp.size() == 1) {
        FillSchmidtMatrices(0, gatex);
      }
    }

    for (auto& gatex : hd.gatexs1t2) {
      if (gatex.schmidt_decomp.size() == 1) {
        FillSchmidtMatrices(0, gatex);
      }
    }

    for (auto& gatex : hd.gatexs2t0) {
      if (gatex.schmidt_decomp.size() == 1) {
        FillSchmidtMatrices(0, gatex);
      }
    }

    return true;
  }

  /**
   * Runs the hybrid simulator on a sectioned lattice.
   * @param param Options for parallelism and logging. Also specifies the size
   *   of the 'prefix' and 'root' sections of the lattice.
   * @param factory Object to create simulators and state spaces.
   * @param hd Container object for gates on the boundary between lattice
   *   sections.
   * @param parts Lattice sections to be simulated.
   * @param fgates0 List of gates from one section of the lattice.
   * @param fgates1 List of gates from the other section of the lattice.
   * @param bitstrings List of output states to simulate, as bitstrings.
   * @param results Output vector of amplitudes. After a successful run, this
   *   will be populated with amplitudes for each state in 'bitstrings'.
   * @return True if the simulation completed successfully; false otherwise.
   */
  template <typename Factory, typename Results>
  bool Run(const Parameter& param, const Factory& factory,
           HybridData& hd, const std::vector<unsigned>& parts,
           const std::vector<GateFused>& fgates0,
           const std::vector<GateFused>& fgates1,
           const std::vector<GateFused>& fgates2,
           const std::vector<uint64_t>& bitstrings, Results& results) const {
    using Simulator = typename Factory::Simulator;
    using StateSpace = typename Simulator::StateSpace;
    using State = typename StateSpace::State;

    std::vector<GateFused> fgates = fgates0;

    fgates.insert(fgates.end(), fgates1.begin(), fgates1.end());
    fgates.insert(fgates.end(), fgates2.begin(), fgates2.end());

    // Compares based on how it's parent GateHybrid is compared in SplitLattice.
    // TODO: This function should use the one from SplitLattice.
    auto compare = [](const GateFused& l, const GateFused& r) -> bool {
      return l.parent->time < r.parent->time || (l.parent->time == r.parent->time &&
          (l.parent->parent < r.parent->parent || (l.parent->parent == r.parent->parent && l.parent->id < r.parent->id)));
    };

    std::sort(fgates.begin(), fgates.end(), compare);

    auto checkpoint_data = CalculateCheckpoints(param, fgates);
    auto loc = checkpoint_data.loc;
    auto p_and_r = checkpoint_data.p_and_r;
    auto loc0 = loc[0];
    auto loc1 = loc[1];
    auto loc2 = loc[2];
    auto p_and_r0t1 = p_and_r[0];
    auto p_and_r1t2 = p_and_r[1];
    auto p_and_r2t0 = p_and_r[2];
    unsigned num_p_gates0t1 = p_and_r0t1[0];
    unsigned num_p_gates1t2 = p_and_r1t2[0];
    unsigned num_p_gates2t0 = p_and_r2t0[0];
    unsigned num_r_gates0t1 = p_and_r0t1[1];
    unsigned num_r_gates1t2 = p_and_r1t2[1];
    unsigned num_r_gates2t0 = p_and_r2t0[1];

    unsigned num_pr_gates0t1 = num_p_gates0t1 + num_r_gates0t1;
    unsigned num_pr_gates1t2 = num_p_gates1t2 + num_r_gates1t2;
    unsigned num_pr_gates2t0 = num_p_gates2t0 + num_r_gates2t0;

    auto bits0t1 = CountSchmidtBits(num_p_gates0t1, num_r_gates0t1, hd.gatexs0t1);
    auto bits1t2 = CountSchmidtBits(num_p_gates1t2, num_r_gates1t2, hd.gatexs1t2);
    auto bits2t0 = CountSchmidtBits(num_p_gates2t0, num_r_gates2t0, hd.gatexs2t0);

    if (bits0t1.num_r_bits > 63 ||
        bits0t1.num_s_bits > 63 ||
        bits1t2.num_r_bits > 63 ||
        bits1t2.num_s_bits > 63 ||
        bits2t0.num_r_bits > 63 ||
        bits2t0.num_s_bits > 63) {
      IO::errorf("Error: r or s bits > 63\nDecrease one of them and try again.\n");
      IO::errorf("  bits0t1.num_r_bits: %d\n", bits0t1.num_r_bits);
      IO::errorf("  bits0t1.num_s_bits: %d\n", bits0t1.num_s_bits);
      IO::errorf("  bits1t2.num_r_bits: %d\n", bits1t2.num_r_bits);
      IO::errorf("  bits1t2.num_s_bits: %d\n", bits1t2.num_s_bits);
      IO::errorf("  bits2t0.num_r_bits: %d\n", bits2t0.num_r_bits);
      IO::errorf("  bits2t0.num_s_bits: %d\n", bits2t0.num_s_bits);
      return false;
    }
    else if ((bits0t1.num_p_bits > 63 || bits1t2.num_p_bits > 63 || bits2t0.num_p_bits > 63) && param.verbosity > 0) {
      IO::messagef("p is very high. 2^%d runs with different values for w0t1 is required for fidelity one.\n", bits0t1.num_p_bits);
      IO::messagef("p is very high. 2^%d runs with different values for w1t2 is required for fidelity one.\n", bits1t2.num_p_bits);
      IO::messagef("p is very high. 2^%d runs with different values for w2t0 is required for fidelity one.\n", bits2t0.num_p_bits);
    }
    else if (param.verbosity > 0) {
      uint64_t pmax0t1 = uint64_t{1} << bits0t1.num_p_bits;
      uint64_t pmax1t2 = uint64_t{1} << bits1t2.num_p_bits;
      uint64_t pmax2t0 = uint64_t{1} << bits2t0.num_p_bits;
      IO::messagef("FOR FIDELITY 1, THE CALLER NEEDS TO RUN pmax0t1 = %d TIMES WITH w0t1 = [0 - %d)\n", pmax0t1, pmax0t1);
      IO::messagef("FOR FIDELITY 1, THE CALLER NEEDS TO RUN pmax1t2 = %d TIMES WITH w1t2 = [0 - %d)\n", pmax1t2, pmax1t2);
      IO::messagef("FOR FIDELITY 1, THE CALLER NEEDS TO RUN pmax2t0 = %d TIMES WITH w2t0 = [0 - %d)\n", pmax2t0, pmax2t0);
    }

    uint64_t rmax0t1 = uint64_t{1} << bits0t1.num_r_bits;
    uint64_t smax0t1 = uint64_t{1} << bits0t1.num_s_bits;
    uint64_t rmax1t2 = uint64_t{1} << bits1t2.num_r_bits;
    uint64_t smax1t2 = uint64_t{1} << bits1t2.num_s_bits;
    uint64_t rmax2t0 = uint64_t{1} << bits2t0.num_r_bits;
    uint64_t smax2t0 = uint64_t{1} << bits2t0.num_s_bits;
    uint64_t rmax_tot = rmax0t1 * rmax1t2 * rmax2t0;
    uint64_t smax_tot = smax0t1 * smax1t2 * smax2t0;

    struct Index {
      unsigned i0;
      unsigned i1;
      unsigned i2;
    };

    std::vector<Index> indices;
    indices.reserve(bitstrings.size());

    // Bitstring indices for part 0, part 1 and part 2. TODO: optimize.
    for (const auto& bitstring : bitstrings) {
      Index index{0, 0, 0};

      for (uint64_t i = 0; i < hd.qubit_map.size(); ++i) {
        unsigned m = ((bitstring >> i) & 1) << hd.qubit_map[i];
        switch (parts[i]) {
        case 0:
          index.i0 |= m;
          break;
        case 1:
          index.i1 |= m;
          break;
        case 2:
          index.i2 |= m;
        }
      }

      indices.push_back(index);
    }

    StateSpace state_space = factory.CreateStateSpace();

    State* rstate0;
    State* rstate1;
    State* rstate2;

    State state0p = state_space.Null();
    State state1p = state_space.Null();
    State state2p = state_space.Null();
    State state0r = state_space.Null();
    State state1r = state_space.Null();
    State state2r = state_space.Null();
    State state0s = state_space.Null();
    State state1s = state_space.Null();
    State state2s = state_space.Null();

    // Create states.

    if (!CreateStates(hd.num_qubits0, hd.num_qubits1, hd.num_qubits2, state_space, true,
                      state0p, state1p, state2p, rstate0, rstate1, rstate2)) {
      return false;
    }

    if (!CreateStates(hd.num_qubits0, hd.num_qubits1, hd.num_qubits2, state_space, (rmax0t1 > 1 || rmax1t2 > 1 ||rmax2t0 > 1),
                      state0r, state1r, state2r, rstate0, rstate1, rstate2)) {
      return false;
    }

    if (!CreateStates(hd.num_qubits0, hd.num_qubits1, hd.num_qubits2, state_space, (smax0t1 > 1 || smax1t2 > 1 ||smax2t0 > 1),
                      state0s, state1s, state2s, rstate0, rstate1, rstate2)) {
      return false;
    }

    state_space.SetStateZero(state0p);
    state_space.SetStateZero(state1p);
    state_space.SetStateZero(state2p);

    Simulator simulator = factory.CreateSimulator();

    std::vector<unsigned> prev0t1(hd.num_gatexs0t1, unsigned(-1));
    std::vector<unsigned> prev1t2(hd.num_gatexs1t2, unsigned(-1));
    std::vector<unsigned> prev2t0(hd.num_gatexs2t0, unsigned(-1));

    // param.prefix0t1, -1t2 and -2t0 encodes the prefixes paths.

    unsigned gatex_index0t1 = SetSchmidtMatrices(
        0, num_p_gates0t1, param.prefix0t1, prev0t1, hd.gatexs0t1);

    unsigned gatex_index1t2 = SetSchmidtMatrices(
        0, num_p_gates1t2, param.prefix1t2, prev1t2, hd.gatexs1t2);

    unsigned gatex_index2t0 = SetSchmidtMatrices(
        0, num_p_gates2t0, param.prefix2t0, prev2t0, hd.gatexs2t0);

    if (gatex_index0t1 == 0 && gatex_index1t2 == 0 && gatex_index2t0 == 0) {
      // Apply gates before the first checkpoint.
      ApplyGates(fgates0, 0, loc0[0], simulator, state0p);
      ApplyGates(fgates1, 0, loc1[0], simulator, state1p);
      ApplyGates(fgates2, 0, loc2[0], simulator, state2p);
    } else {
      IO::errorf("invalid prefix %lu for prefix gate index %u or\n",
                 param.prefix0t1, gatex_index0t1 - 1);
      IO::errorf("invalid prefix %lu for prefix gate index %u or\n",
                 param.prefix1t2, gatex_index1t2 - 1);
      IO::errorf("invalid prefix %lu for prefix gate index %u.\n",
                 param.prefix2t0, gatex_index2t0 - 1);
      return false;
    }

    // Branch over root gates on the cut. r encodes the root path.
    for (uint64_t r0t1 = 0; r0t1 < rmax0t1; ++r0t1) {
    for (uint64_t r1t2 = 0; r1t2 < rmax1t2; ++r1t2) {
    for (uint64_t r2t0 = 0; r2t0 < rmax2t0; ++r2t0) {
      if (rmax_tot > 1) {
        state_space.Copy(state0p, state0r);
        state_space.Copy(state1p, state1r);
        state_space.Copy(state2p, state2r);
      }

      // All needs to be valid. If one of them is invalid, the combination is invalid.
      if (SetSchmidtMatrices(num_p_gates0t1, num_pr_gates0t1,
                             r0t1, prev0t1, hd.gatexs0t1) == 0 &&
          SetSchmidtMatrices(num_p_gates1t2, num_pr_gates1t2,
                             r1t2, prev1t2, hd.gatexs1t2) == 0 &&
          SetSchmidtMatrices(num_p_gates2t0, num_pr_gates2t0,
                             r2t0, prev2t0, hd.gatexs2t0) == 0) {
        // Apply gates before the second checkpoint.
        // TODO: Run these on three different nodes.
        // TODO: Also fgates2 does not depend on r0t1 etc
        ApplyGates(fgates0, loc0[0], loc0[1], simulator, state0r);
        ApplyGates(fgates1, loc1[0], loc1[1], simulator, state1r);
        ApplyGates(fgates2, loc2[0], loc2[1], simulator, state2r);
      } else {
        continue;
      }

      // Branch over suffix gates on the cut. s encodes the suffix path.
      for (uint64_t s0t1 = 0; s0t1 < smax0t1; ++s0t1) {
      for (uint64_t s1t2 = 0; s1t2 < smax1t2; ++s1t2) {
      for (uint64_t s2t0 = 0; s2t0 < smax2t0; ++s2t0) {
        if (smax_tot > 1) {
          state_space.Copy(rmax_tot > 1 ? state0r : state0p, state0s);
          state_space.Copy(rmax_tot > 1 ? state1r : state1p, state1s);
          state_space.Copy(rmax_tot > 1 ? state2r : state2p, state2s);
        }

        if (SetSchmidtMatrices(num_pr_gates0t1, hd.num_gatexs0t1,
                               s0t1, prev0t1, hd.gatexs0t1) == 0 &&
            SetSchmidtMatrices(num_pr_gates1t2, hd.num_gatexs1t2,
                               s1t2, prev1t2, hd.gatexs1t2) == 0 &&
            SetSchmidtMatrices(num_pr_gates2t0, hd.num_gatexs2t0,
                               s2t0, prev2t0, hd.gatexs2t0) == 0) {
          // Apply the rest of the gates.

          ApplyGates(fgates0, loc0[1], fgates0.size(), simulator, state0s);
          ApplyGates(fgates1, loc1[1], fgates1.size(), simulator, state1s);
          ApplyGates(fgates2, loc2[1], fgates2.size(), simulator, state2s);
        } else {
          continue;
        }

        auto f = [](unsigned n, unsigned m, uint64_t i,
                    const StateSpace& state_space,
                    const State& state0, const State& state1, const State& state2,
                    const std::vector<Index>& indices, Results& results) {
          // TODO: make it faster for the CUDA state space.
          auto a0 = state_space.GetAmpl(state0, indices[i].i0);
          auto a1 = state_space.GetAmpl(state1, indices[i].i1);
          auto a2 = state_space.GetAmpl(state2, indices[i].i2);
          results[i] += a0 * a1 * a2;
        };

        // Collect results.
        for_.Run(results.size(), f,
                 state_space, *rstate0, *rstate1, *rstate2, indices, results);
      }
      }
      }
    }
    }
    }

    return true;
  }

 private:

  struct CheckpointData {
    std::array<std::array<unsigned, 2>, 3> loc;
    std::array<std::array<unsigned, 2>, 3> p_and_r;
  };

  /**
   * Identifies when to save "checkpoints" of the simulation state. These allow
   * runs with different cut-index values to reuse parts of the simulation.
   * It also counts the number of prefix and root gates for each cut.
   * This is used to know how many paths there are for each cut.
   * @param param Options for parallelism and logging. Also specifies the size
   *   of the 'prefix' and 'root' sections of the lattice (in total for all cuts).
   * @param fgates Set of all fgates.
   * @return The struct CheckpointData consisting of:
   *  - Three pairs of numbers specifying how many FusedGates to apply before the
   *      first and second checkpoints, respectively.
   *  - Three pairs of numbers specifying p and r for each cut 0t1, 1t2, 2t0.
   */
  static CheckpointData CalculateCheckpoints(
      const Parameter& param, const std::vector<GateFused>& fgates) {
    CheckpointData checkpoint_data;
    checkpoint_data.loc = {{{0, 0}, {0, 0}, {0, 0}}};
    checkpoint_data.p_and_r = {{{0, 0}, {0, 0}, {0, 0}}};

    // Decomposed gates in the cuts 0t1, 1t2 and 0t2 so far.
    std::array<unsigned,3> num_decomposed = {0, 0, 0};
    unsigned num_p_gates = param.num_prefix_gatexs;
    unsigned num_pr_gates = num_p_gates + param.num_root_gatexs;

    // Loop through all fgates.
    for (std::size_t i = 0; i < fgates.size(); ++i) {
      // Check if there is one HybridGate in the fgate that is on a cut.
      auto fgate = fgates[i];
      if (fgate.parent != nullptr) {
        auto gateh = fgate.parent;
        if (gateh->parent != nullptr) {
          if (gateh->parts[0] == 0 && gateh->parts[1] == 1) {
            // Increase number of gates on cut 0t1
            ++num_decomposed[0];
          }
          else if (gateh->parts[0] == 1 && gateh->parts[1] == 2) {
            // Increase number of gates on cut 1t2
            ++num_decomposed[1];
          }
          else if (gateh->parts[0] == 2 && gateh->parts[1] == 0) {
            // Increase number of gates on cut 2t0
            ++num_decomposed[2];
          }
          else {
            IO::errorf("Error: cut not identified\n");
          }

          // Add loc for both parts: parts[1] here, and parts[0] after if (same as 1 qubit gates).
          if (num_decomposed[0] + num_decomposed[1] + num_decomposed[2] <= num_p_gates) {
            checkpoint_data.loc[gateh->parts[1]][0]++;
          }
          if (num_decomposed[0] + num_decomposed[1] + num_decomposed[2] <= num_pr_gates) {
            checkpoint_data.loc[gateh->parts[1]][1]++;
          }

          /**
           * GateHybrids on a cut will be in two parts, and thus appear twise in the list of fgates.
           * Since we sort fgates based on their parent GateHybrid, the next fgate will have the "same"
           * GateHybrid as parent, but they are on different parts.
           * So the parent Gate of the GateHybrids should be identical.
           */
          if (gateh->parent != fgates[i+1].parent->parent) {
            IO::errorf("Fused gates on the cut did not have the same grandparent Gate\n");
            exit(1);
          }

          // We skip the next fgate since we increase loc for both parts in this iteration.
          ++i;
        }
        // FusedGate has HybridGates with one or two parts.
        if (num_decomposed[0] + num_decomposed[1] + num_decomposed[2] <= num_p_gates) {
          if (num_decomposed[0] + num_decomposed[1] + num_decomposed[2] == num_p_gates) {
            // Set number of prefix gates for each cut.
            checkpoint_data.p_and_r[0][0] = num_decomposed[0];
            checkpoint_data.p_and_r[1][0] = num_decomposed[1];
            checkpoint_data.p_and_r[2][0] = num_decomposed[2];
          }
          checkpoint_data.loc[gateh->parts[0]][0]++;
        }
        if (num_decomposed[0] + num_decomposed[1] + num_decomposed[2] <= num_pr_gates) {
          if (num_decomposed[0] + num_decomposed[1] + num_decomposed[2] == num_pr_gates) {
            // Set number of root gates for each cut to num_decomposed - num prefix gates
            checkpoint_data.p_and_r[0][1] = num_decomposed[0] - checkpoint_data.p_and_r[0][0];
            checkpoint_data.p_and_r[1][1] = num_decomposed[1] - checkpoint_data.p_and_r[1][0];
            checkpoint_data.p_and_r[2][1] = num_decomposed[2] - checkpoint_data.p_and_r[2][0];
          }
          checkpoint_data.loc[gateh->parts[0]][1]++;
        }
      }
      else {
        IO::errorf("Error: Fused gate did not have parent\n");
        exit(1);
      }
    }

    return checkpoint_data;
  }

  struct Bits {
    unsigned num_p_bits;
    unsigned num_r_bits;
    unsigned num_s_bits;
  };

  static Bits CountSchmidtBits(
      unsigned num_prefix_gatexs, unsigned num_root_gatexs, const std::vector<GateX>& gatexs) {
    Bits bits{0, 0, 0};

    unsigned num_p_gates = num_prefix_gatexs;
    unsigned num_pr_gates = num_p_gates + num_root_gatexs;

    for (std::size_t i = 0; i < gatexs.size(); ++i) {
      const auto& gatex = gatexs[i];
      if (i < num_p_gates) {
        bits.num_p_bits += gatex.schmidt_bits;
      } else if (i < num_pr_gates) {
        bits.num_r_bits += gatex.schmidt_bits;
      } else {
        bits.num_s_bits += gatex.schmidt_bits;
      }
    }

    return bits;
  }

  static unsigned SetSchmidtMatrices(std::size_t i0, std::size_t i1,
                                     uint64_t path,
                                     std::vector<unsigned>& prev_k,
                                     std::vector<GateX>& gatexs) {
    unsigned shift_length = 0;

    for (std::size_t i = i0; i < i1; ++i) {
      const auto& gatex = gatexs[i];

      if (gatex.schmidt_bits == 0) {
        // Continue if gatex has Schmidt rank 1.
        continue;
      }

      unsigned k = (path >> shift_length) & ((1 << gatex.schmidt_bits) - 1);
      shift_length += gatex.schmidt_bits;

      if (k != prev_k[i]) {
        if (k >= gatex.schmidt_decomp.size()) {
          // Invalid path. Returns gatex index plus one to report error in case
          // of invalid prefix.
          return i + 1;
        }

        FillSchmidtMatrices(k, gatex);

        prev_k[i] = k;
      }
    }

    return 0;
  }

  static void FillSchmidtMatrices(unsigned k, const GateX& gatex) {
    unsigned part0;
    unsigned part1;

    /**
     * If they are in the "natural order" we want part0=0 and part1=1
     * Otherwise the reverse
     * Natural order means that gatex.part_q0 + 1 = gatex.part_q1 (mod 3)
     * TODO: Generalize this to > 3 parts
     */
    if (gatex.part_q0 + 1 == gatex.part_q1 || (gatex.part_q0 == 2 && gatex.part_q1 == 0)) {
      part0 = 0;
      part1 = 1;
    }
    else {
      part0 = 1;
      part1 = 0;
    }

    {
      gatex.decomposed0->matrix.resize(gatex.schmidt_decomp[k][part0].size());
      auto begin = gatex.schmidt_decomp[k][part0].begin();
      auto end = gatex.schmidt_decomp[k][part0].end();
      std::copy(begin, end, gatex.decomposed0->matrix.begin());
    }
    {
      gatex.decomposed1->matrix.resize(gatex.schmidt_decomp[k][part1].size());
      auto begin = gatex.schmidt_decomp[k][part1].begin();
      auto end = gatex.schmidt_decomp[k][part1].end();
      std::copy(begin, end, gatex.decomposed1->matrix.begin());
    }
  }

  template <typename Simulator>
  static void ApplyGates(const std::vector<GateFused>& gates,
                         std::size_t i0, std::size_t i1,
                         const Simulator& simulator,
                         typename Simulator::State& state) {
    for (std::size_t i = i0; i < i1; ++i) {
      if (gates[i].matrix.size() > 0) {
        ApplyFusedGate(simulator, gates[i], state);
      } else {
        auto gate = gates[i];
        CalculateFusedMatrix(gate);
        ApplyFusedGate(simulator, gate, state);
      }
    }
  }

  static unsigned SchmidtBits(unsigned size) {
    switch (size) {
    case 1:
      return 0;
    case 2:
      return 1;
    case 3:
      return 2;
    case 4:
      return 2;
    default:
      // Not supported.
      return 42;
    }
  }

  template <typename StateSpace>
  static bool CreateStates(unsigned num_qubits0,unsigned num_qubits1,unsigned num_qubits2,
                           const StateSpace& state_space, bool create,
                           typename StateSpace::State& state0,
                           typename StateSpace::State& state1,
                           typename StateSpace::State& state2,
                           typename StateSpace::State* (&rstate0),
                           typename StateSpace::State* (&rstate1),
                           typename StateSpace::State* (&rstate2)) {
    if (create) {
      state0 = state_space.Create(num_qubits0);
      state1 = state_space.Create(num_qubits1);
      state2 = state_space.Create(num_qubits2);

      if (state_space.IsNull(state0) || state_space.IsNull(state1) || state_space.IsNull(state2)) {
        IO::errorf("not enough memory: is the number of qubits too large?\n");
        return false;
      }

      rstate0 = &state0;
      rstate1 = &state1;
      rstate2 = &state2;
    }

    return true;
  }

  For for_;
};

}  // namespace qsim

#endif  // HYBRID_H_
