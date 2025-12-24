package nn_pkg;
    /*
    * MLP ARCHITECTURE CONSTANTS
    * Change these to change entire network topology
    */
    parameter int IN_DIM       = 64;
    parameter int HIDDDEN_SIZE = 8;
    parameter int OUTPUT_SIZE  = 10;

    // GLOBAL NUERICAL REPRESENTATIONS
    parameter int DATA_W = 8;
    parameter int ACC_W  = 64; 

    typedef logic signed [DATA_W-1:0] data_t;
    typedef logic signed [ACC_W-1:0]  acc_t;
endpackage