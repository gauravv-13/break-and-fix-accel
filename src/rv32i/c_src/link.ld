OUTPUT_ARCH("riscv")
ENTRY(_start)

SECTIONS
{
    . = 0x80000000;
    .text : {
        *(.text.init)
        *(.text*)
    }
    . = ALIGN(16);
    .rodata : {
        *(.rodata*)
    }
    . = ALIGN(16);
    .data : {
        *(.data*)
        *(.sdata*)
    }
    . = ALIGN(16);
    .bss : {
        *(.bss*)
        *(.sbss*)
        *(COMMON)
    }
    . = 0x8000f000;
    tohost = .;
    .tohost : {
        LONG(0)
    }
}
