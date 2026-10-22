// Phase 5 reference: the multiply-accumulate of one processing element of the systolic
// array (signed 8x8 multiply + 32-bit add). Its logic depth sets the clock period that
// the row checkers are compared against.
module pe_mac (
    input  wire signed [7:0]  a,
    input  wire signed [7:0]  w,
    input  wire signed [31:0] psum_in,
    output wire signed [31:0] psum_out
);
    assign psum_out = psum_in + a * w;
endmodule
