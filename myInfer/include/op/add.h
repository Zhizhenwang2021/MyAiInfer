//
// Created by lx186 on 2024/12/1.
//

#ifndef MYINFER_INCLUDE_OP_ADD_H_
#define MYINFER_INCLUDE_OP_ADD_H_

#include "base/base.h"
#include "layer.h"
namespace op {
class VecAddLayer : public Layer {
public:
    explicit VecAddLayer(base::DeviceType device_type);

    base::Status check() const override;

    base::Status forward() override;
};
}

#endif //MYINFER_INCLUDE_OP_ADD_H_
