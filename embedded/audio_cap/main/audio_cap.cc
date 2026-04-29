#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/i2s_std.h"
#include "driver/gpio.h"

#include "esp_log.h"
#include "esp_timer.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "micro_features_generator.h"
#include "micro_model_settings.h"
#include "recognize_commands.h"

#include "dsps_fft2r.h"
#include "dsps_wind.h"
#include "dsps_view.h"

#include <math.h>
#include <stdio.h>

#define FFT_SIZE 512
#define NUM_MELS 40

#define I2S_SCK_PIN GPIO_NUM_41
#define I2S_WS_PIN  GPIO_NUM_42
#define I2S_SD_PIN  GPIO_NUM_2

#define SAMPLE_RATE 16000
#define DMA_BUF_COUNT 8
#define DMA_BUF_LEN 64
#define READ_BUF_SIZE 1024

#define LOG_EVERY_N 50

constexpr int kTensorArenaSize = 128 * 1024;

static const char* labels[] = {"_silence", "_unknown", "dog", "basketball"};
static const int NUM_LABELS = 4;

static uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));

static int16_t audio_buffer[16000];
static int buffer_pos = 0;

static const tflite::Model* tfl_model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;

static i2s_chan_handle_t rx_handle = NULL;

static float lp_filter = 0;
static const char *TAG = "audio_capture";


void init_dsp() {
    esp_err_t res = dsps_fft2r_init_fc32(NULL, 1024);
    if (res != ESP_OK) {
        ESP_LOGE("DSP", "FFT initialization failed!");
        return;
    }
    ESP_LOGI("DSP", "DSP initialized");
}

static void init_tflite() {

    tfl_model = tflite::GetModel(kws_model_tflite);

    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMaxPool2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddMean();
    resolver.AddMul();
    resolver.AddAdd();
    resolver.AddReduceMax();

    static tflite::MicroInterpreter static_interpreter(
        tfl_model,
        resolver,
        tensor_arena,
        kTensorArenaSize
    );

    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed!");
        return;
    }

    input_tensor = interpreter->input(0);

    if (InitializeMicroFeatures() != kTfLiteOk) {
        ESP_LOGE(TAG, "InitializeMicroFeatures failed!");
        return;
    }

    ESP_LOGI(TAG, "TFLite initialized OK");
}

void init_audio_hardware() {

    i2s_chan_config_t chan_cfg =
        I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);

    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_handle));

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(16000),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_32BIT,
            I2S_SLOT_MODE_MONO
        ),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_SCK_PIN,
            .ws   = I2S_WS_PIN,
            .dout = I2S_GPIO_UNUSED,
            .din  = I2S_SD_PIN,
        },
    };

    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_handle, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_handle));
}

// void run_inference_dsp() {
//     const int num_frames = 49;
//     const int samples_per_frame = 480; 
//     const int step_size = 320;        
//     const int fft_size = 512;          
    
//     static int8_t model_input_buffer[num_frames * 40]; 
//     static float hann_window[samples_per_frame];
//     static bool window_init = false;

//     // Initialize the window 
//     if (!window_init) {
//         dsps_wind_hann_f32(hann_window, samples_per_frame);
//         window_init = true;
//     }

//     static float fft_input[fft_size * 2]; 
//     static int16_t unwrapped_buffer[16000];

//     int samples_to_end = 16000 - buffer_pos;
//     memcpy(unwrapped_buffer, &audio_buffer[buffer_pos], samples_to_end * 2);
//     memcpy(&unwrapped_buffer[samples_to_end], audio_buffer, buffer_pos * 2);

//     for (int f = 0; f < num_frames; f++) {
//         int start_sample = f * step_size;
        
//         for (int i = 0; i < fft_size; i++) {
//             if (i < samples_per_frame) {
//                 fft_input[i * 2] = (float)unwrapped_buffer[start_sample + i] * hann_window[i];
//                 fft_input[i * 2 + 1] = 0; 
//             } else {
//                 fft_input[i * 2] = 0;
//                 fft_input[i * 2 + 1] = 0;
//             }
//         }

//         dsps_fft2r_fc32(fft_input, fft_size);
//         dsps_bit_rev_fc32(fft_input, fft_size);

//         for (int m = 0; m < 40; m++) {
//             float bin_power = 0;
//             int bin_start = m * 4; 
//             for (int j = 0; j < 4; j++) {
//                 float re = fft_input[(bin_start + j) * 2];
//                 float im = fft_input[(bin_start + j) * 2 + 1];
//                 bin_power += sqrtf(re*re + im*im);
//             }
            
//             float val = log1pf(bin_power); 

//             int8_t quantized = (int8_t)(val * 4.0f - 80.0f);
//             model_input_buffer[f * 40 + m] = quantized;
//         }
//     }

//     if (interpreter != nullptr) {
//         memcpy(interpreter->input(0)->data.int8, model_input_buffer, num_frames * 40);
//         printf("INPUT_SCAN: ");
//         int8_t* scan_ptr = (int8_t*)interpreter->input(0)->data.int8;
//         for(int i = 0; i < 10; i++) {
//             printf("%d, ", scan_ptr[i]);
//         }
//         printf("\n");
//         if (interpreter->Invoke() == kTfLiteOk) {
//             int8_t* out = interpreter->output(0)->data.int8;
//             ESP_LOGI("DSP_AI", "SIL:%d, UNK:%d, YES:%d, BSK:%d", out[0], out[1], out[2], out[3]);
//         } else {
//             ESP_LOGE("DSP_AI", "Inference Failed!");
//         }
//     } else {
//         ESP_LOGE("DSP_AI", "Interpreter NULL. Check init_tflite()");
//     }
// }

void run_inference() {

    static int16_t linear_buffer[16000];

    for (int i = 0; i < 16000; i++) {
        linear_buffer[i] =
            audio_buffer[(buffer_pos + i) % 16000];
    }

    Features features;

    if (GenerateFeatures(linear_buffer, 16000, &features) != kTfLiteOk) {
        ESP_LOGE(TAG, "Feature generation failed!");
        return;
    }

    for (int i = 0; i < kFeatureCount; i++) {
        for (int j = 0; j < kFeatureSize; j++) {
            input_tensor->data.int8[i * kFeatureSize + j] =
                features[i][j];
        }
    }

    int64_t start_time = esp_timer_get_time();

    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed!");
        return;
    }

    int64_t end_time = esp_timer_get_time();
    int64_t inference_time_us = end_time - start_time;
    float inference_time_ms = inference_time_us / 1000.0f;

    TfLiteTensor* output = interpreter->output(0);

    // ESP_LOGI(TAG, "SIL:%d UNK:%d YES:%d BSK:%d",
    //     output->data.int8[0],
    //     output->data.int8[1],
    //     output->data.int8[2],
    //     output->data.int8[3]
    // );
    
    // Find highest score
    int max_index = 0;
    int max_value = output->data.int8[0];

    for (int i = 1; i < NUM_LABELS; i++) {
        if (output->data.int8[i] > max_value) {
            max_value = output->data.int8[i];
            max_index = i;
        }
    }

    // Print best prediction
    ESP_LOGI(TAG, "PREDICTION: %s (%d)",
            labels[max_index],
            max_value);
}

void audio_task(void *arg) {

    init_audio_hardware();
    init_dsp();
    init_tflite();

    static uint8_t i2s_raw_buffer[4096];
    static int16_t pcm_output[1024];

    size_t bytes_read = 0;

    while (1) {

        esp_err_t ret = i2s_channel_read(
            rx_handle,
            i2s_raw_buffer,
            sizeof(i2s_raw_buffer),
            &bytes_read,
            portMAX_DELAY
        );

        if (ret == ESP_OK && bytes_read > 0) {

            int sample_count = bytes_read / 4;

            for (int i = 0; i < sample_count; i++) {

                int32_t raw_val = ((int32_t*)i2s_raw_buffer)[i];
                int32_t s = raw_val >> 14;

                lp_filter = (lp_filter * 0.95f) + ((float)s * 0.05f);
                int32_t filtered = (int32_t)((float)s - lp_filter);

                filtered *= 4;

                if (filtered > 32767) filtered = 32767;
                if (filtered < -32768) filtered = -32768;

                pcm_output[i] = (int16_t)filtered;

                audio_buffer[buffer_pos] = pcm_output[i];
                buffer_pos = (buffer_pos + 1) % 16000;

                if (buffer_pos % 3200 == 0) {
                    run_inference();
                }
            }
        }
    }
}

extern "C" void app_main(void) {
    xTaskCreatePinnedToCore(
        audio_task,
        "audio_task",
        32768,
        NULL,
        5,
        NULL,
        1
    );
}