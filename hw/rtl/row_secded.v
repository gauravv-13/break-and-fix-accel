// Phase 5: SECDED (extended Hamming) decoder/corrector for one weight row
// (matches src/defense_secded.py).
//   Data bit j sits at "position" data_pos(j) = the j-th number >= 3 that is not a power of 2.
//   Stored word = {P, h}: h = XOR of positions of the 1 data bits (R bits), P = overall
//   parity of data bits and h.
//   s = h_now ^ h_stored, p = parity(row) ^ parity(h_stored) ^ P_stored
//   s=0,p=0 -> ok ; p=0,s!=0 -> detected ; p=1: s=0 or power of 2 -> corrected (a check
//   bit was hit, data fine) ; s = a data position -> flip that bit (corrected) ;
//   otherwise detected.
// status: 2'd0 ok, 2'd1 corrected, 2'd2 detected. code = encode(row) for the write path.
// Set R = number of Hamming bits: 7 for 8-byte rows, 5 for 2-byte rows.
module row_secded #(
    parameter N_BYTES = 8,
    parameter R       = 7
) (
    input  wire [8*N_BYTES-1:0] row,
    input  wire [R:0]           stored,
    output wire [8*N_BYTES-1:0] fixed,
    output wire [1:0]           status,
    output wire [R:0]           code
);
    localparam K = 8 * N_BYTES;

    function integer data_pos;
        input integer j;
        integer p, cnt;
        begin
            data_pos = 0;
            cnt = 0;
            for (p = 3; p < 1024; p = p + 1)
                if ((p & (p - 1)) != 0) begin
                    if (cnt == j) data_pos = p;
                    cnt = cnt + 1;
                end
        end
    endfunction

    // mask of data bits whose position has bit b set
    function [K-1:0] pos_mask;
        input integer b;
        integer j;
        begin
            pos_mask = {K{1'b0}};
            for (j = 0; j < K; j = j + 1)
                pos_mask[j] = (data_pos(j) >> b) & 1;
        end
    endfunction

    wire [R-1:0] h_st = stored[R-1:0];
    wire         P_st = stored[R];

    wire [R-1:0] h_now;
    genvar gb;
    generate
        for (gb = 0; gb < R; gb = gb + 1) begin : g_h
            assign h_now[gb] = ^(row & pos_mask(gb));
        end
    endgenerate

    wire data_par = ^row;
    wire [R-1:0] s = h_now ^ h_st;
    wire p = data_par ^ (^h_st) ^ P_st;

    localparam integer MAXPOS = data_pos(K - 1);
    wire s_zero  = (s == {R{1'b0}});
    wire s_pow2  = ((s & (s - 1'b1)) == {R{1'b0}}) & ~s_zero;
    wire s_valid = ~s_zero & ~s_pow2 & (s <= MAXPOS);

    assign status = (s_zero & ~p)                  ? 2'd0 :
                    (~p)                           ? 2'd2 :
                    (s_zero | s_pow2 | s_valid)    ? 2'd1 : 2'd2;

    genvar gj;
    generate
        for (gj = 0; gj < K; gj = gj + 1) begin : g_fix
            assign fixed[gj] = row[gj] ^ (p & (s == data_pos(gj)));
        end
    endgenerate

    assign code = {data_par ^ (^h_now), h_now};
endmodule
