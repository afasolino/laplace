# SystemVerilog ready/valid, AXI4-Lite, and W1C invariant card

This project-authored card records recurring microarchitectural invariants for bounded RTL repairs.

## One-entry ready/valid storage

A full one-entry stage must accept a replacement item when the current item is consumed in the same cycle. A common combinational relationship is:

```systemverilog
assign in_ready = ~full_q | out_ready;
assign out_valid = full_q;
assign out_data = data_q;
```

Sequential updates must distinguish input and output handshakes. During simultaneous dequeue and enqueue, keep the stage full and replace `data_q` with the new payload. While `out_valid && !out_ready`, keep both `out_valid` and `out_data` stable. Reset leaves the stage empty. Avoid a combinational path from `in_valid` to `out_valid` unless the specified architecture deliberately permits bypass.

## AXI4-Lite writes

AW and W are independent channels and may arrive in either order. Capture each channel independently, perform the register write only after both have been accepted, apply `WSTRB` per byte, and issue one B response per completed write. Do not require AWVALID and WVALID in the same cycle. Keep response payloads stable while stalled.

## Write-one-to-clear event state

Apply writes only to byte lanes selected by `WSTRB`. A W1C bit clears pending state when the corresponding written bit is one. Define same-cycle event-set versus software-clear priority explicitly. IRQ is asserted only when both enable and pending are set, and it deasserts after pending is cleared. Intentionally unused upper write-data bits should be consumed or documented in a lint-clean way rather than relying on global warning suppression.
