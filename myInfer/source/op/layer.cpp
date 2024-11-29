//
// Created by lx186 on 2024/11/27.
//
#include "op/layer.h"
#include <base/cuda_config.h>
#include <glog/logging.h>
#include <cstdarg>
#include <numeric>
#include <utility>



namespace op {
    BaseLayer::BaseLayer(base::DeviceType device_type, LayerType layer_type, base::DataType data_type,
                         std::string layer_name)
        : device_type_(device_type),
          layer_type_(layer_type),
          data_type_(data_type),
          layer_name_(std::move(layer_name)) {}



}
