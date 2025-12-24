#include "system.h"
#include "altera_avalon_pio_regs.h"
#include "sys/alt_stdio.h"
#include <stdint.h>
#include <unistd.h>  // for usleep

int main() {
    alt_putstr("Sender started\n");

    alt_u8 prev_val = 0xFF;  // initialize to impossible value to detect first change

    while(1) {
        // read 8 switches (SWITCHES_BASE is 8-bit wide)
        alt_u8 sw_val = IORD_ALTERA_AVALON_PIO_DATA(SWITCHES_BASE) & 0xFF;

        // only send if value changed
        if(sw_val != prev_val) {
            prev_val = sw_val;

            // print decimal value as string to JTAG UART
            char buf[4];  // max 3 digits + null
            snprintf(buf, sizeof(buf), "%u", sw_val);
            alt_putstr(buf);
            alt_putstr("\n");
        }

        usleep(200000);  // small delay to debounce / reduce printing
    }
}
