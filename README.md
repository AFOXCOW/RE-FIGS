# RI-FIGS
Repeated Identification with Featured Ion-Guided Stoichiometry (RI-FIGS) is a complete and compact solution on DI-SPA data for more confident identifications and corresponding label free quantifications. 

## RI-FIGS-ID.py
The bootstrap aggregation linear discriminant analysis based identification.
* mzML_file: the mzml file path
* SSM_file: the spectrum spectrum match results file (csodiaq csv format)
* lib_file: the spectral library path
* start_cycle: the cycle No. starts from
* end_cycle: the cycle No. ends with
* good_shared_limit: the threshold to select good target.
* good_cos_sim_limit: the threshold to select good target. range(0,1)
* good_sqrt_cos_sim_limit: the threshold to select good target. range(0,1)
* good_count_within_cycle_limit: the threshold to select good target.
* tol: fragment mass error(tolerance) in ppm.
* scans_per_cycle: the scan number in each cycle.
* seed: the seed to randomly choose decoy.

## RI-FIGS-Quant.py
* mzML_file:the mzml file path
* SSM_file: the final identification results file
* lib_file: the spectral library path
* topN: keep topN peaks in spectral library.
* tol: fragment mass error(tolerance) in ppm.
