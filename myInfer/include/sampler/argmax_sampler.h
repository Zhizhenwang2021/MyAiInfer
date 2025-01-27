//
// Created by lx186 on 2025/1/14.
//

#ifndef ARGMAX_SAMPLER_H
#define ARGMAX_SAMPLER_H
#include <base/base.h>
#include "sampler.h"
namespace sampler {
    class ArgmaxSampler : public Sampler {
    public:
        explicit ArgmaxSampler(base::DeviceType device_type) : Sampler(device_type){}

        size_t sample(const float *logits, size_t size, void *stream) override;
    };
}


#endif //ARGMAX_SAMPLER_H
