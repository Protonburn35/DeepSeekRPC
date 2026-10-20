# Returned critical review of 554f155

The three accompanying files are the user-supplied independent review and its
CPU control-flow reproduction. They are retained unchanged as review evidence,
not runtime commands or a current qualification report. The reproduction script
expects the old failures and hard-codes the old 32-character validator for R1;
it is not a success regression for the repair.

R1, R2 and R3 were independently reproduced locally before edits. Permanent
regressions in tests/test_startup_review.py and tests/test_vision_input.py cross
the actual affected call boundaries. Their new startup/image cases failed
before repair and passed afterward. See docs/FINDINGS.md DSR-AUDIT-028 for the
fix scope and limitations. GPU/model execution remains unrun.
