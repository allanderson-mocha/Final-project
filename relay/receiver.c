#include "system.h"
#include "altera_avalon_pio_regs.h"
#include "sys/alt_stdio.h"
#include <stdint.h>
#include <stdlib.h>

int main() {
    alt_putstr("Receiver started\n");

    while(1) {
        char buffer[4];  // enough for 0-255 + newline
        int idx = 0;

        // read until Enter
        char c;
        while((c = alt_getchar()) != '\n' && idx < 3) {
            buffer[idx++] = c;
        }
        buffer[idx] = '\0';  // null terminate

        int value = atoi(buffer);   // convert string to integer
        if(value >=0 && value <= 255) {
            IOWR_ALTERA_AVALON_PIO_DATA(LEDS_BASE, (uint8_t)value);
        }
    }
}
