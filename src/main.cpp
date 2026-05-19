#include "FreeRTOS.h"
#include "ei_accelerometer.h"
#include "ei_analogsensor.h"
#include "ei_at_handlers.h"
#include "ei_classifier_porting.h"
#include "ei_device_raspberry_rp2xxx.h"
#include "ei_dht11sensor.h"
#include "ei_inertialsensor.h"
#include "ei_rp2xxx_internal_temperature.h"
#include "ei_run_impulse.h"
#include "ei_ultrasonicsensor.h"
#include "pico/multicore.h"
#include "pico/stdlib.h"
#include "task.h"
#include <stdio.h>
#include <time.h>

// imu
#include <hardware/gpio.h>
#include <hardware/i2c.h>
#include <hardware/pwm.h>
#include <hardware/uart.h>
#include <pico/stdio.h>

// freertos
#include <FreeRTOS.h>
#include <queue.h>
#include <semphr.h>
#include <stdlib.h>
#include <string.h>
#include <task.h>

// // específico
#include "mpu6050.h"
// edited
// -- Adicione estas 3 linhas em main.cpp --
#include "edge-impulse-sdk/classifier/ei_model_types.h"
#include "edge-impulse-sdk/dsp/numpy.hpp"
#include "model-parameters/model_metadata.h"

using namespace ei;

extern "C" EI_IMPULSE_ERROR
run_classifier(ei::signal_t *signal, ei_impulse_result_t *result, bool debug);

static bool debug_nn = false;

const int MPU_ADDRESS = 0x68;
const int I2C_SDA_GPIO = 4;
const int I2C_SCL_GPIO = 5;

typedef struct {
    uint8_t r, g, b;
} color_t;

static QueueHandle_t xQueueColor;

static const uint LED_PIN_BLUE = 13;
static const uint LED_PIN_GREEN = 14;
static const uint LED_PIN_RED = 15;

// Ânodo comum: PWM alto = LED apagado, PWM baixo = LED aceso.
static const uint16_t PWM_WRAP = 255;

static void mpu6050_init()
{
    i2c_init(i2c_default, 400 * 1000);
    gpio_set_function(I2C_SDA_GPIO, GPIO_FUNC_I2C);
    gpio_set_function(I2C_SCL_GPIO, GPIO_FUNC_I2C);
    gpio_pull_up(I2C_SDA_GPIO);
    gpio_pull_up(I2C_SCL_GPIO);

    // Two byte reset. First byte register, second byte data
    // There are a load more options to set up the device in different ways that could be added here
    uint8_t buf[] = { 0x6B, 0x00 };
    i2c_write_blocking(i2c_default, MPU_ADDRESS, buf, 2, false);
}

static void mpu6050_read_raw(int16_t accel[3], int16_t gyro[3], int16_t *temp)
{
    uint8_t buffer[14];

    // Read all data sequentially starting from acceleration registers (0x3B)
    // 0x3B-0x40: acceleration (6 bytes)
    // 0x41-0x42: temperature (2 bytes)
    // 0x43-0x48: gyro (6 bytes)
    uint8_t val = 0x3B;
    i2c_write_blocking(i2c_default, MPU_ADDRESS, &val, 1, true);
    i2c_read_blocking(i2c_default, MPU_ADDRESS, buffer, 14, false);

    // Parse acceleration
    for (int i = 0; i < 3; i++) {
        accel[i] = (buffer[i * 2] << 8 | buffer[(i * 2) + 1]);
    }

    // Parse temperature
    *temp = buffer[6] << 8 | buffer[7];

    // Parse gyro
    for (int i = 0; i < 3; i++) {
        gyro[i] = (buffer[8 + i * 2] << 8 | buffer[8 + (i * 2) + 1]);
    }
}

static void pwm_task(void *p)
{
    gpio_set_function(LED_PIN_BLUE, GPIO_FUNC_PWM);
    gpio_set_function(LED_PIN_GREEN, GPIO_FUNC_PWM);
    gpio_set_function(LED_PIN_RED, GPIO_FUNC_PWM);

    uint slice_b = pwm_gpio_to_slice_num(LED_PIN_BLUE);
    uint slice_g = pwm_gpio_to_slice_num(LED_PIN_GREEN);
    uint slice_r = pwm_gpio_to_slice_num(LED_PIN_RED);

    pwm_config config = pwm_get_default_config();
    pwm_config_set_clkdiv(&config, 4.f);
    pwm_config_set_wrap(&config, PWM_WRAP);
    pwm_init(slice_b, &config, true);
    pwm_init(slice_g, &config, true);
    pwm_init(slice_r, &config, true);

    pwm_set_gpio_level(LED_PIN_BLUE, PWM_WRAP);
    pwm_set_gpio_level(LED_PIN_GREEN, PWM_WRAP);
    pwm_set_gpio_level(LED_PIN_RED, PWM_WRAP);

    color_t color;
    while (true) {
        if (xQueueReceive(xQueueColor, &color, portMAX_DELAY)) {
            pwm_set_gpio_level(LED_PIN_BLUE, PWM_WRAP - color.b);
            pwm_set_gpio_level(LED_PIN_GREEN, PWM_WRAP - color.g);
            pwm_set_gpio_level(LED_PIN_RED, PWM_WRAP - color.r);
        }
    }
}

static void gesture_recognize_task(void *p)
{
    mpu6050_init();
    int16_t accelerometer[3], gyro[3], temp;

    static float buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];

    while (true) {
        for (size_t ix = 0; ix < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; ix += 3) {
            mpu6050_read_raw(accelerometer, gyro, &temp);
            buffer[ix + 0] = accelerometer[0];
            buffer[ix + 1] = accelerometer[1];
            buffer[ix + 2] = accelerometer[2];

            vTaskDelay(pdMS_TO_TICKS(10));
        }

        // Prepara sinal
        ei::signal_t signal;
        int err = numpy::signal_from_buffer(buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
        if (err != 0) {
            ei_printf("Failed to create signal from buffer (%d)\n", err);
            continue;
        }

        // Run the classifier
        ei_impulse_result_t result = { 0 };
        err = run_classifier(&signal, &result, debug_nn);
        if (err != EI_IMPULSE_OK) {
            ei_printf("ERR: Failed to run classifier (%d)\n", err);
            continue;
        }

        // print the predictions
        ei_printf("Predictions ");
        ei_printf(
            "(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
            result.timing.dsp,
            result.timing.classification,
            result.timing.anomaly);
        ei_printf(": \n");
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            ei_printf(
                "teste    %s: %.5f\n",
                result.classification[ix].label,
                result.classification[ix].value);
        }

        size_t best_ix = 0;
        float best_val = result.classification[0].value;
        for (size_t ix = 1; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            if (result.classification[ix].value > best_val) {
                best_val = result.classification[ix].value;
                best_ix = ix;
            }
        }

        const char *label = result.classification[best_ix].label;
        color_t color = { 0, 0, 0 };
        if (strcmp(label, "idle") == 0)
            color.r = 255;
        else if (strcmp(label, "Wave") == 0)
            color.g = 255;
        else if (strcmp(label, "Up and down") == 0)
            color.b = 255;

        xQueueSend(xQueueColor, &color, 0);
        ei_printf(">>> Gesto: %s (%.2f)\n", label, best_val);

#if EI_CLASSIFIER_HAS_ANOMALY == 1
        ei_printf("    anomaly score: %.3f\n", result.anomaly);
#endif
    }
}

int main(void)
{
    stdio_init_all();

    xQueueColor = xQueueCreate(4, sizeof(color_t));

    xTaskCreate(gesture_recognize_task, "gesture_task", 16384, NULL, 1, NULL);
    xTaskCreate(pwm_task, "pwm_task", 2048, NULL, 1, NULL);
    vTaskStartScheduler();

    while (true)
        ;
}
