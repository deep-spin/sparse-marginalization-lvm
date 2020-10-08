#pragma once

#include "sequence.h"

namespace AD3 {

class FactorSequenceBinary : public FactorSequence
{
  protected:
    virtual double GetNodeScore(
      int position,
      int state,
      const vector<double>& variable_log_potentials,
      const vector<double>& additional_log_potentials) override
    {
        if (state == 1) {
            return variable_log_potentials[position];
        } else {
            return variable_log_potentials[length_ + position];
        }
    }

    // The edge connects node[position-1] to node[position].
    virtual double GetEdgeScore(
      int position,
      int previous_state,
      int state,
      const vector<double>& variable_log_potentials,
      const vector<double>& additional_log_potentials) override
    {
        // only consider positive-to-positive transitions;
        // this automatically rules out the initial and final transitions
        if (previous_state == 1 && state == 1) {
            return additional_log_potentials[position - 1];
        } else
            return 0;
    }

    virtual void AddNodePosterior(
      int position,
      int state,
      double weight,
      vector<double>* variable_posteriors,
      vector<double>* additional_posteriors) override
    {
        if (state == 1) {
            (*variable_posteriors)[position] += weight;
        } else {
            (*variable_posteriors)[length_ + position] += weight;
        }
    }

    // The edge connects node[position-1] to node[position].
    virtual void AddEdgePosterior(
      int position,
      int previous_state,
      int state,
      double weight,
      vector<double>* variable_posteriors,
      vector<double>* additional_posteriors) override
    {
        if (previous_state == 1 && state == 1) {
            (*additional_posteriors)[position - 1] += weight;
        }
    }

  public:
    void Initialize(const int length)
    {
        length_ = length;
        num_states_ = vector<int>(length, 2);
        num_additionals_ = length - 1;
    }

  public:
    FactorSequenceBinary() {}
    virtual ~FactorSequenceBinary() { ClearActiveSet(); }

    int length_;
};

} // namespace AD3
