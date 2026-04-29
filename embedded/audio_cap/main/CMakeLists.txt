idf_component_register(SRCS "audio_cap.cc"
                    "micro_features_generator.cc"
                    "recognize_commands.cc"
                    INCLUDE_DIRS "."
                    PRIV_REQUIRES esp-tflite-micro esp-dsp driver esp_timer)

                    # Get the path to the tflm component
idf_component_get_property(tflm_dir espressif__esp-tflite-micro COMPONENT_DIR)

# Force the compiler to look in the folder where we just saw all those .h files
target_include_directories(${COMPONENT_LIB} PRIVATE "${tflm_dir}/signal/src")
