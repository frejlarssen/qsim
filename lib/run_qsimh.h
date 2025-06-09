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
  static bool Run(Parameter& param, const Factory& factory,
                  const Circuit& circuit, const std::vector<unsigned>& parts,
                  const std::vector<uint64_t>& bitstrings,
                  std::vector<std::complex<fp_type>>& results) {
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
      return false;
    }

    if (hd.num_gatexs < param.num_prefix_gatexs + param.num_root_gatexs) {
      IO::errorf("error: num_prefix_gates (%u) plus num_root gates (%u) is "
                 "greater than num_gates_on_the_cut (%u).\n",
                 param.num_prefix_gatexs, param.num_root_gatexs,
                 hd.num_gatexs);
      return false;
    }

    auto fgates0 = Fuser::FuseGates(param, hd.num_qubits0, hd.gates0);
    if (fgates0.size() == 0 && hd.gates0.size() > 0) {
      return false;
    }

    auto fgates1 = Fuser::FuseGates(param, hd.num_qubits1, hd.gates1);
    if (fgates1.size() == 0 && hd.gates1.size() > 0) {
      return false;
    }

    unsigned num_p_gates = param.num_prefix_gatexs;

    if (param.auto_num_root_gatexs) {
      unsigned max_r = hd.num_gatexs - num_p_gates;
      unsigned best_r = 0;
      unsigned best_time = std::numeric_limits<unsigned>::max();

      for (unsigned r_guess = 0; r_guess <= max_r; r_guess++) {

        param.num_root_gatexs = r_guess;
        unsigned num_pr_gates = num_p_gates + r_guess;

        auto bits = HybridSimulator::CountSchmidtBits(param, hd.gatexs);

        uint64_t rmax = uint64_t{1} << bits.num_r_bits;
        uint64_t smax = uint64_t{1} << bits.num_s_bits;

        auto loc0 = HybridSimulator::CheckpointLocations(param, fgates0);
        auto loc1 = HybridSimulator::CheckpointLocations(param, fgates1);

        unsigned mr_0 = loc0[1] - loc0[0];
        unsigned mr_1 = loc1[1] - loc1[0];
        unsigned ms_0 = fgates0.size() - loc0[1];
        unsigned ms_1 = fgates1.size() - loc1[1];

        // Compute the time for this guess.
        uint64_t time =
            rmax * ((mr_0 + mr_1) + (smax * (ms_0 + ms_1)));

        if (time < best_time) {
          best_time = time;
          best_r = r_guess;
        }
      }

      param.num_root_gatexs = best_r;
    }

    if (param.verbosity > 0) {
      PrintInfo(param, hd);
    }

    rc = HybridSimulator(param.num_threads).Run(
        param, factory, hd, parts, fgates0, fgates1, bitstrings, results);

    if (rc && param.verbosity > 0) {
      double t1 = GetTime();
      IO::messagef("prefix %d: time elapsed %g seconds.\n", param.prefix, t1 - t0);
      report_memory_usage(param.prefix, "For this prefix");
    }

    return rc;
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
