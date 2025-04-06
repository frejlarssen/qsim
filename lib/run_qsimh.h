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

#ifndef RUN_QSIMH_H_
#define RUN_QSIMH_H_

#include <string>
#include <vector>
#include <mpi.h>

#include "hybrid.h"
#include "util.h"

namespace qsim {

/**
 * Helper struct for running qsimh.
 */
template <typename IO, typename HybridSimulator>
struct QSimHRunner final {
  using Gate = typename HybridSimulator::Gate;
  using fp_type = typename HybridSimulator::fp_type;

  using Parameter = typename HybridSimulator::Parameter;
  using HybridData = typename HybridSimulator::HybridData;
  using Fuser = typename HybridSimulator::Fuser;
  using Amplitude = typename std::complex<fp_type>;

  /**
   * Evaluates the amplitudes for a given circuit and set of output states.
   * @param param Options for gate fusion, parallelism and logging. Also
   *   specifies the size of the 'prefix' and 'root' sections of the lattice.
   * @param factory Object to create simulators and state spaces.
   * @param circuit The circuit to be simulated.
   * @param parts Lattice sections to be simulated.
   * @param bitstrings List of output states to simulate, as bitstrings.
   * @param results Output vector of amplitudes. After a successful run, this
   *   will be populated with amplitudes for each state in 'bitstrings'.
   * @return True if the simulation completed successfully; false otherwise.
   */
  template <typename Factory, typename Circuit>
  static bool Run(const Parameter& param, const Factory& factory,
                  const Circuit& circuit, const std::vector<unsigned>& parts,
                  const std::vector<uint64_t>& bitstrings,
                  std::vector<Amplitude>& results_accumulated) {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (circuit.num_qubits != parts.size()) {
      IO::errorf("parts size is not equal to the number of qubits.");
      return false;
    }

    double t0 = 0.0;

    if (param.verbosity > 0) {
      t0 = GetTime();
    }

    HybridData hd;
    bool rc = HybridSimulator::SplitLattice(parts, circuit.gates, hd);

    if (!rc) {
      //TODO: Tell the other rank that there was an error
      return false;
    }

    if (hd.num_gatexs < param.num_prefix_gatexs + param.num_root_gatexs) {
      IO::errorf("error: num_prefix_gates (%u) plus num_root gates (%u) is "
                 "greater than num_gates_on_the_cut (%u).\n",
                 param.num_prefix_gatexs, param.num_root_gatexs,
                 hd.num_gatexs);
      //TODO: Tell the other rank that there was an error
      return false;
    }

    if (param.verbosity > 0 && world_rank == 0) {
      PrintInfo(param, hd);
    }

    std::vector<typename Fuser::GateFused> fgates0;
    std::vector<typename Fuser::GateFused> fgates1;

    if (world_rank == 0) {
      fgates0 = Fuser::FuseGates(param, hd.num_qubits0, hd.gates0);
      if (fgates0.size() == 0 && hd.gates0.size() > 0) {
        //TODO: Tell the other rank that there was an error
        return false;
      }
    } else if (world_rank == 1) {
      fgates1 = Fuser::FuseGates(param, hd.num_qubits1, hd.gates1);
      if (fgates1.size() == 0 && hd.gates1.size() > 0) {
        //TODO: Tell the other rank that there was an error
        return false;
      }
    }

    auto bits = HybridSimulator::CountSchmidtBits(param, hd.gatexs);

    hd.rmax = uint64_t{1} << bits.num_r_bits;
    hd.smax = uint64_t{1} << bits.num_s_bits;

    std::vector<unsigned> indices0, indices1;
    indices0.reserve(bitstrings.size());
    indices1.reserve(bitstrings.size());

    // Bitstring indices for part 0 and part 1. TODO: optimize.
    for (const auto& bitstring : bitstrings) {
      unsigned index0 = 0;
      unsigned index1 = 0;

      for (uint64_t i = 0; i < hd.qubit_map.size(); ++i) {
        unsigned m = ((bitstring >> i) & 1) << hd.qubit_map[i];
        parts[i] ? index1 |= m : index0 |= m;
      }

      indices0.push_back(index0);
      indices1.push_back(index1);
    }

    uint64_t sblock_size = bitstrings.size();
    uint64_t rblock_size = hd.smax * sblock_size;
    uint64_t res_size    = hd.rmax * rblock_size;

    std::vector<Amplitude> results;

    try {
      results.resize(res_size, Amplitude(0.0f, 0.0f));
    } catch (const std::bad_alloc& e) {
      IO::errorf("rank %d: Too many gates on root and suffix cut to resize vector\n", world_rank);
      return false;
    }

    bool rc_part;
    if (world_rank == 0) {
      rc_part = HybridSimulator(param.num_threads).Run(
        param, factory, hd, parts, fgates0, hd.num_qubits0, bitstrings, indices0, results);
    } else if (world_rank == 1) {
      rc_part = HybridSimulator(param.num_threads).Run(
        param, factory, hd, parts, fgates1, hd.num_qubits1, bitstrings, indices1, results);
    }

    bool rc_all = false;
    MPI_Reduce(&rc_part, &rc_all, 1, MPI_C_BOOL, MPI_LAND, 0, MPI_COMM_WORLD);

    if ((world_rank == 0 && !rc_all) || (world_rank > 0 && !rc_part)) {
      return false;
    }

    if (world_rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, results.data(), res_size, 
                   MPI_C_FLOAT_COMPLEX, MPI_PROD, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(results.data(), nullptr, res_size,
                   MPI_C_FLOAT_COMPLEX, MPI_PROD, 0, MPI_COMM_WORLD);
    }

    if (world_rank > 0) {
      return true;
    }

    uint64_t index;
    for (uint64_t r = 0; r < hd.rmax; ++r) {
      for (uint64_t s = 0; s < hd.smax; ++s) {
        for (uint64_t i = 0; i < bitstrings.size(); i++) {
          index = r * rblock_size + s * sblock_size + i;
          results_accumulated[i] += results[index];
        }
      }
    }

    if (param.verbosity > 0) {
      double t1 = GetTime();
      IO::messagef("time elapsed %g seconds.\n", t1 - t0);
    }
    return true;
  }

 private:
  static void PrintInfo(const Parameter& param, const HybridData& hd) {
    unsigned num_suffix_gates =
        hd.num_gatexs - param.num_prefix_gatexs - param.num_root_gatexs;

    IO::messagef("part 0: %u, part 1: %u\n", hd.num_qubits0, hd.num_qubits1);
    IO::messagef("%u gates on the cut\n", hd.num_gatexs);
    IO::messagef("breakup: %up+%ur+%us\n", param.num_prefix_gatexs,
                 param.num_root_gatexs, num_suffix_gates);
  }
};

}  // namespace qsim

#endif  // RUN_QSIM_H_
