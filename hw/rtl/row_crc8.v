// Phase 5: CRC-8 checker for one weight row (matches src/defense_checksum.py, kind="crc8").
// Polynomial x^8 + x^2 + x + 1 (0x07), init 0, no final XOR, bytes in memory order,
// most significant bit first. CRC (init 0) is linear, so each output bit is the XOR of a
// fixed set of row bits: crc_mask(b) marks the row bits whose single-bit CRC has bit b set.
// Writing it this way gives a balanced XOR tree (the plain bit-serial loop synthesized into
// long XOR chains).
module row_crc8 #(
    parameter N_BYTES = 8
) (
    input  wire [8*N_BYTES-1:0] row,
    input  wire [7:0]           stored,
    output wire [7:0]           code,
    output wire                 ok
);
    function [7:0] crc8;
        input [8*N_BYTES-1:0] d;
        integer i, b;
        reg [7:0] c;
        begin
            c = 8'h00;
            for (i = 0; i < N_BYTES; i = i + 1) begin
                c = c ^ d[8*i +: 8];
                for (b = 0; b < 8; b = b + 1)
                    c = c[7] ? ({c[6:0], 1'b0} ^ 8'h07) : {c[6:0], 1'b0};
            end
            crc8 = c;
        end
    endfunction

    function [8*N_BYTES-1:0] crc_mask;
        input integer b;
        integer j;
        reg [8*N_BYTES-1:0] onehot;
        reg [7:0] c;
        begin
            crc_mask = {8*N_BYTES{1'b0}};
            for (j = 0; j < 8 * N_BYTES; j = j + 1) begin
                onehot = {8*N_BYTES{1'b0}};
                onehot[j] = 1'b1;
                c = crc8(onehot);
                crc_mask[j] = c[b];
            end
        end
    endfunction

    genvar gb;
    generate
        for (gb = 0; gb < 8; gb = gb + 1) begin : g_crc
            assign code[gb] = ^(row & crc_mask(gb));
        end
    endgenerate
    assign ok   = (code == stored);
endmodule
