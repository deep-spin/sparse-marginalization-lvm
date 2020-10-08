#pragma once

#include "ad3/GenericFactor.h"

namespace AD3 {

class FactorBernoulli : public GenericFactor
{
  public:
    FactorBernoulli() {}
    virtual ~FactorBernoulli() { ClearActiveSet(); }

    // Obtain the best configuration.
    void Maximize(const vector<double>& variable_log_potentials,
                  const vector<double>&,
                  Configuration& configuration,
                  double* value)
    override
    {
        *value = 0;
        vector<int>* cfg =
          static_cast<vector<int>*>(configuration);
        for (int i = 0; i < length_; ++i)
        {
            if (variable_log_potentials[i] > variable_log_potentials[length_ + i])
            {
                (*cfg)[i] = 1;
                *value += variable_log_potentials[i];
            } else
            {
                (*cfg)[i] = 0;
                *value += variable_log_potentials[length_ + i];
            }
        }
    }

    // Compute the score of a given assignment.
    void Evaluate(const vector<double>& variable_log_potentials,
                  const vector<double>&,
                  const Configuration configuration,
                  double* value)
    override
    {
        const vector<int>* sequence =
          static_cast<const vector<int>*>(configuration);
        *value = 0.0;
        for (int i = 0; i < length_; ++i)
        {
            if ((*sequence)[i] == 1)
                *value += variable_log_potentials[i];
            else
                *value += variable_log_potentials[length_ + i];
        }
    }

    // Given a configuration with a probability (weight),
    // increment the vectors of variable and additional posteriors.
    void UpdateMarginalsFromConfiguration(const Configuration& configuration,
                                          double weight,
                                          vector<double>* variable_posteriors,
                                          vector<double>*)
    override
    {
        const vector<int>* sequence =
          static_cast<const vector<int>*>(configuration);
        for (int i = 0; i < length_; ++i)
        {
            if ((*sequence)[i] == 1)
                (*variable_posteriors)[i] += weight;
            else
                (*variable_posteriors)[length_ + i] += weight;
        }
    }

    // Count how many common values two configurations have.
    int CountCommonValues(const Configuration& configuration1,
                          const Configuration& configuration2)
    override
    {
        const vector<int>* sequence1 =
          static_cast<const vector<int>*>(configuration1);
        const vector<int>* sequence2 =
          static_cast<const vector<int>*>(configuration2);
        assert(sequence1->size() == sequence2->size());
        int count = 0;
        for (int i = 0; i < sequence1->size(); ++i) {
            if ((*sequence1)[i] == (*sequence2)[i])
                ++count;
        }
        return count;
    }

    // Check if two configurations are the same.
    bool SameConfiguration(const Configuration& configuration1,
                           const Configuration& configuration2)
    override
    {
        const vector<int>* sequence1 =
          static_cast<const vector<int>*>(configuration1);
        const vector<int>* sequence2 =
          static_cast<const vector<int>*>(configuration2);

        assert(sequence1->size() == sequence2->size());
        for (int i = 0; i < sequence1->size(); ++i) {
            if ((*sequence1)[i] != (*sequence2)[i])
                return false;
        }
        return true;
    }

    // Delete configuration.
    void DeleteConfiguration(Configuration configuration)
    override
    {
        vector<int>* sequence = static_cast<vector<int>*>(configuration);
        delete sequence;
    }

    Configuration CreateConfiguration()
    override
    {
        vector<int>* sequence = new vector<int>(length_, -1);
        return static_cast<Configuration>(sequence);
    }

  public:
    void Initialize(int length)
    {
        length_ = length;
    }

    virtual size_t GetNumAdditionals() override { return num_additionals_; }

  protected:
    // Number of states for each position.
    int length_;
    int num_additionals_ = 0;
};

} // namespace AD3
