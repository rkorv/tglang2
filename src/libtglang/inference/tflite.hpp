#pragma once

#include <iostream>
#include <vector>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/logger.h>
#include <tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h>

#include <tf_model.hpp>   // Generated by `xxd -i model.tflite > model.h`
#include <model_meta.hpp> // Generated by python script

#include "../tglang.h"

#include <chrono>

class TglangModelInference
{
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromBuffer(reinterpret_cast<const char *>(tf_model_tflite), tf_model_tflite_len);
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;

    void init_interpreter()
    {
        tflite::LoggerOptions::SetMinimumLogSeverity(tflite::LogSeverity::TFLITE_LOG_SILENT);
        tflite::InterpreterBuilder(*model, this->resolver)(&this->interpreter);
        this->interpreter->SetNumThreads(0);
    }

public:
    TglangModelInference()
    {
        this->init_interpreter();
    }

    enum TglangLanguage forward(
        std::vector<det_int_t> &encoded_text,
        std::vector<det_int_t> &naming_types,
        std::vector<det_int_t> &group_types,
        std::vector<det_int_t> &lines_num,
        std::vector<det_int_t> &positions_ids)
    {
        /* Allocate buffers */
        std::vector<int> dims = {(int)encoded_text.size()};
        if (this->interpreter->ResizeInputTensor(this->interpreter->inputs()[0], dims) != kTfLiteOk ||
            this->interpreter->ResizeInputTensor(this->interpreter->inputs()[1], dims) != kTfLiteOk ||
            this->interpreter->ResizeInputTensor(this->interpreter->inputs()[2], dims) != kTfLiteOk ||
            this->interpreter->ResizeInputTensor(this->interpreter->inputs()[3], dims) != kTfLiteOk ||
            this->interpreter->ResizeInputTensor(this->interpreter->inputs()[4], dims) != kTfLiteOk)
        {
            std::cerr << "Failed to resize input tensor." << std::endl;
            return TGLANG_LANGUAGE_OTHER;
        }

        if (this->interpreter->AllocateTensors() != kTfLiteOk)
        {
            std::cerr << "Failed to allocate tensors." << std::endl;
            return TGLANG_LANGUAGE_OTHER;
        }

        /* Copy tokens to buffers */
        det_int_t *input0 = this->interpreter->typed_input_tensor<det_int_t>(0);
        det_int_t *input1 = this->interpreter->typed_input_tensor<det_int_t>(1);
        det_int_t *input2 = this->interpreter->typed_input_tensor<det_int_t>(2);
        det_int_t *input3 = this->interpreter->typed_input_tensor<det_int_t>(3);
        det_int_t *input4 = this->interpreter->typed_input_tensor<det_int_t>(4);

        /**
         * Kind of issue in tensorflow-lite, the input tensor order is not the same as in the model.
         * original: 0,1,2,3,4 -> encoded_text, naming_types, group, lines, positions_ids
         *     here: 0,1,2,3,4 -> naming_types, group, positions_ids, lines, encoded_text
         */
        std::copy(naming_types.begin(), naming_types.end(), input0);
        std::copy(group_types.begin(), group_types.end(), input1);
        std::copy(positions_ids.begin(), positions_ids.end(), input2);
        std::copy(lines_num.begin(), lines_num.end(), input3);
        std::copy(encoded_text.begin(), encoded_text.end(), input4);

        /* Call tflite */
        if (this->interpreter->Invoke() != kTfLiteOk)
        {
            std::cerr << "Failed to invoke interpreter." << std::endl;
            return TGLANG_LANGUAGE_OTHER;
        }

        float label_conf = *this->interpreter->typed_output_tensor<float>(1);
        if (label_conf < DETECTION_THRESHOLD)
            return TGLANG_LANGUAGE_OTHER;
        det_int_t label = *this->interpreter->typed_output_tensor<det_int_t>(0);

        return static_cast<TglangLanguage>(label);
    }
};
