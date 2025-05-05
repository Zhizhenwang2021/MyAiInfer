//
// Created by lx186 on 2025/1/14.
//

#ifndef MYINFER_INCLUDE_MODEL_LLAMA3_H
#define MYINFER_INCLUDE_MODEL_LLAMA3_H
#include <base/cuda_config.h>
#include "model.h"
#include "op/add.h"
#include "op/embedding.h"

namespace model {

    struct LLama2Layers {
        std::shared_ptr<op::Layer> add_layer_;
        std::shared_ptr<op::Layer> rope_layer_;
        std::shared_ptr<op::Layer> swiglu_layer_;
        std::shared_ptr<op::Layer> mha_layer_;

        std::vector<std::shared_ptr<op::Layer>> wq_layers_;
        std::vector<std::shared_ptr<op::Layer>> wk_layers_;
        std::vector<std::shared_ptr<op::Layer>> wv_layers_;
        std::vector<std::shared_ptr<op::Layer>> wo_layers_;

        std::vector<std::shared_ptr<op::Layer>> w1_layers_;
        std::vector<std::shared_ptr<op::Layer>> w2_layers_;
        std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;
        std::vector<std::shared_ptr<op::Layer>> w3_layers_;
        std::shared_ptr<op::Layer> cls_layer_;

        std::shared_ptr<op::Layer> embedding_layer_;

        void to_cuda(std::shared_ptr<kernel::CudaConfig> config);


    };

    class LLama2Layers : public Model {
    public:
        explicit LLama2Model(base::TokenizerType tokenizer_type, std::string token_path,
            std::string model_path, bool is_quant_model);

        base::Status init(base::DeviceType device_type) override;

        base::Status predict(const tensor::Tensor &input, const tensor::Tensor &pos_tensor, bool is_prompt, int &next) const override;

        base::Status forward(const tensor::Tensor &input, const tensor::Tensor &pos_tensor, int &next) const override;

        op::EmbeddingOutput embedding(const std::vector<int> &tokens) const override;

    private:
        void init_mem() override;

        base::Status create_layers() override;

        void create_param_layers() override;

        void create_nonparam_layers() override;

        void create_param_quant_layers() override;

        void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

        void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;

        void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;

        void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

        void cls_logits(const tensor::Tensor& input) const;

        int32_t post_processing(const tensor::Tensor &pos, bool is_prompt) const override;

        std::shared_ptr<kernel::CudaConfig> cuda_config_;
        std::unique_ptr<LLama2Layers> llama_layers_;
    };

}
#endif //MYINFER_INCLUDE_MODEL_LLAMA3_H
