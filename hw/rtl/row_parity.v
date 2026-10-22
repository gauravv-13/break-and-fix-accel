// Phase 5: parity checker for one weight row (matches src/defense_checksum.py, kind="parity").
// row packing: byte i of the row (memory order) is row[8*i +: 8]; row bit j = row[j].
// Purely combinational: code = XOR of all row bits; ok = (code == stored parity bit).
module row_parity #(
    parameter N_BYTES = 8
) (
    input  wire [8*N_BYTES-1:0] row,
    input  wire                 stored,
    output wire                 code,
    output wire                 ok
);
    assign code = ^row;
    assign ok   = (code == stored);
endmodule
