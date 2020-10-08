#pragma once

#include <algorithm>

#include "ad3/Factor.h"
#include "ad3/GenericFactor.h"

#include "bernoulli.h"

namespace AD3 {
class FactorBudget : public FactorBernoulli
{
  public:

    FactorBudget() {}
    virtual ~FactorBudget() { ClearActiveSet(); }

    void Initialize(int length, int budget)
    {
        length_ = length;
        budget_ = budget;
    }

    void Maximize(const vector<double>& variable_log_potentials,
                  const vector<double>& add_,
                  Configuration& configuration,
                  double* value)
    override
    {
        // Create a local copy of the log potentials.
        vector<double> eta(length_);
        vector<int>* y = static_cast<vector<int>*>(configuration);
        *value = 0.0;

        // start with all variables off
        for (int i = 0; i < length_; ++i)
        {
            eta.at(i) = (variable_log_potentials.at(i) -
                         variable_log_potentials.at(length_ + i));
            *value += variable_log_potentials.at(length_ + i);
            y->at(i) = 0;
        }

        double valaux;

        // first, try including everything with positive score
        size_t num_active = 0;
        double sum = 0.0;
        for (size_t i = 0; i < length_; ++i) {
            if (eta[i] > 0) {
                sum += eta[i];
                y->at(i) = 1;
                ++num_active;
            }
        }

        // if we went over budget, we sort, and only include the top
        if (num_active > budget_)
        {
            vector<pair<double, int>> scores(length_);
            for (size_t i = 0; i < length_; ++i) {
                scores.at(i).first = -eta.at(i);
                scores.at(i).second = i;
            }
            sort(scores.begin(), scores.end());
            num_active = 0;
            sum = 0.0;
            for (size_t k = 0; k < budget_; ++k) {
                valaux = -scores[k].first;
                if (valaux < 0)
                    break;
                int i = scores[k].second;
                y->at(i) = 1;
                sum += valaux;
                ++num_active;
            }

            for (size_t k = num_active; k < length_; ++k) {
                int i = scores[k].second;
                y->at(i) = 0;
            }
        }

        *value += sum;
    }

  protected:
    int budget_;
};

}
