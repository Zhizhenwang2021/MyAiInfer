//
// Created by lx186 on 2024/12/15.
//

#ifndef MYINFER_INCLUDE_OP_RMSNORM_H
#define MYINFER_INCLUDE_OP_RMSNORM_H
#include "layer.h"
namespace op {
    class RmsNormLayer : public LayerParam {
    public:
        explicit RmsNormLayer(base::DeviceType device_type, int32_t dim);

        base::Status check() const override;

        base::Status forward() override;

    private:
        int32_t dim_ = 0;
    };
}


#endif //MYINFER_INCLUDE_OP_RMSNORM_H
