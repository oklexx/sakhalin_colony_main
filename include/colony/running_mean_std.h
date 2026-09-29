#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <json.hpp>

namespace colony {

// Welford's online algorithm for running mean/variance.
// Port of stable-baselines3 VecNormalize RunningMeanStd.
class RunningMeanStd {
public:
    RunningMeanStd() = default;

    explicit RunningMeanStd(int size)
        : mean_(size, 0.0), var_(size, 1.0), count_(1.0) {}

    int size() const { return (int)mean_.size(); }
    int64_t count() const { return (int64_t)count_; }
    const double* mean() const { return mean_.data(); }
    const double* var() const { return var_.data(); }

    // Update running statistics with a batch of observations.
    // Each row is [row_stride] floats; only the first `size` elements are used.
    void update(const float* data, int n_rows, int row_stride) {
        if (n_rows <= 0) return;
        int sz = size();
        double old_count = count_;
        double new_count = old_count + (double)n_rows;

        for (int j = 0; j < sz; ++j) {
            double batch_mean = 0.0;
            double batch_var = 0.0;
            int valid = 0;
            for (int i = 0; i < n_rows; ++i) {
                double val = (double)data[i * row_stride + j];
                if (!std::isfinite(val)) continue;
                batch_mean += val;
                batch_var += val * val;
                valid++;
            }
            if (valid == 0) continue;
            batch_mean /= (double)valid;
            if (valid > 1) {
                batch_var = batch_var / (double)valid - batch_mean * batch_mean;
            } else {
                batch_var = 0.0;
            }

            double delta = batch_mean - mean_[j];
            double new_mean = mean_[j] + delta * ((double)valid / new_count);
            double m2 = var_[j] * old_count + batch_var * (double)valid
                        + delta * delta * old_count * (double)valid / new_count;
            mean_[j] = new_mean;
            var_[j] = m2 / new_count;
        }
        count_ = new_count;
    }

    // Update with a single scalar value (for reward stats).
    void update_scalar(double val) {
        if (!std::isfinite(val)) return;  // skip NaN/Inf
        double old_count = count_;
        count_ += 1.0;
        double delta = val - mean_[0];
        double new_mean = mean_[0] + delta / count_;
        double m2 = var_[0] * old_count + delta * (val - new_mean);
        mean_[0] = new_mean;
        var_[0] = m2 / count_;
    }

    // Normalize a batch of observations in-place.
    // obs: flat buffer [n_rows * row_stride], normalize first `size` elements per row.
    void normalize(float* obs, int n_rows, int row_stride, double clip_val = 10.0) const {
        int sz = size();
        double eps = 1e-8;
        for (int i = 0; i < n_rows; ++i) {
            for (int j = 0; j < sz; ++j) {
                double v = (double)obs[i * row_stride + j];
                double denominator = std::sqrt(std::abs(var_[j]) + eps);
                v = (v - mean_[j]) / denominator;
                if (!std::isfinite(v)) v = 0.0;
                if (v > clip_val) v = clip_val;
                if (v < -clip_val) v = -clip_val;
                obs[i * row_stride + j] = (float)v;
            }
        }
    }

    // Normalize a single reward.
    double normalize_reward(double rew, double clip_val = 10.0) const {
        double eps = 1e-8;
        double denominator = std::sqrt(std::abs(var_[0]) + eps);
        double v = rew / denominator;
        if (!std::isfinite(v)) v = 0.0;
        if (v > clip_val) v = clip_val;
        if (v < -clip_val) v = -clip_val;
        return v;
    }

    void set_mean(const std::vector<double>& m) { mean_ = m; }
    void set_var(const std::vector<double>& v) { var_ = v; }
    void set_count(double c) { count_ = c; }

    // JSON serialization (compatible with SB3 vecnormalize.pkl logic).
    nlohmann::json to_json() const {
        nlohmann::json j;
        j["mean"] = mean_;
        j["var"] = var_;
        j["count"] = count_;
        return j;
    }

    void from_json(const nlohmann::json& j) {
        // Валидация до записи: раньше j["mean"] на null/не-объекте это
        // JSON_ASSERT (в release — UB), а mean/var разной длины или пустые
        // массивы доезжали до normalize() и давали обращения вне границ.
        if (!j.is_object() || !j.contains("mean") || !j.contains("var")) {
            throw std::runtime_error(
                "RunningMeanStd::from_json: expected an object with \"mean\" and \"var\"");
        }
        std::vector<double> mean = j["mean"].get<std::vector<double>>();
        std::vector<double> var = j["var"].get<std::vector<double>>();
        if (mean.empty() || mean.size() != var.size()) {
            throw std::runtime_error(
                "RunningMeanStd::from_json: empty mean/var or size mismatch");
        }
        double count = j.value("count", 1.0);
        if (!std::isfinite(count) || count <= 0.0) {
            throw std::runtime_error(
                "RunningMeanStd::from_json: count must be a positive finite number");
        }
        mean_ = std::move(mean);
        var_ = std::move(var);
        count_ = count;
    }

private:
    std::vector<double> mean_;
    std::vector<double> var_;
    double count_ = 1.0;
};

}  // namespace colony
