#!/usr/bin/env python3


# =================
# ===  Imports  ===
# =================
import os

from crmass.gbdtanalysis import GBDTAnalysis

# ===========================================
# ===  Variable definitions for analysis  ===
# ===========================================

VERBOSE = True
SAVE_OUTPUT = True

lg_e_bin_edges = [16.0, 16.2, 16.4, 16.6, 16.8, 17.0, 17.2, 17.4, 17.6, 17.8, 18.0, 18.2, 18.4,
                  18.6, 18.8, 19.0, 19.2, 19.4, 19.6, 19.8, 20.0, 20.2, 20.4]

#zenith_deg_bin_edges = [(0.0, 30.0), (40.0, 60.0)]
zenith_deg_bin_edges = [(40.0, 60.0)]

observatory_name = "Auger"

#hadronic_model_names = ["EPOS LHC-R", "Sibyll 2.3e", "QGSJETIII-01"]
hadronic_model_names = ["EPOS LHC-R"]

#primaries = ["ProtonIron", "HeliumOxygen", "ProtonHelium"]
primaries = ["ProtonIron"]

smear_values = [True, False]

make_plots = False

input_file_path = "/home/bflaggs/Documents/Research/MassSensitiveObservablesPaper/ASCIIFiles/ForML/NextGenModels/"
output_file_path = "/home/bflaggs/Documents/Research/MassSensitiveObservablesPaper/code/CRMassMultivariateML/model_output/"

for zeniths in zenith_deg_bin_edges:
    if type(zeniths) != tuple:
        raise ValueError("Entries in 'zenith_deg_bin_edges' must be tuples of the zenith angle bin edges in degrees!")
    
    for model in hadronic_model_names:
        if model == "EPOS LHC-R":
            had_model_file = "EPOSLHCR"
        elif model == "Sibyll 2.3e":
            had_model_file = "Sibyll23e"
        elif model == "QGSJETIII-01":
            had_model_file = "QGSJETIII01"
        else:
            raise ValueError("Implement new models in the hadronic model loop!")
        
        for prim in primaries:
            if prim == "ProtonIron":
                proton_iron = True
                helium_oxygen = False
                proton_helium = False
            elif prim == "HeliumOxygen":
                proton_iron = False
                helium_oxygen = True
                proton_helium = False
            elif prim == "ProtonHelium":
                proton_iron = False
                helium_oxygen = False
                proton_helium = True
            else:
                # Will report values for ProtonHeavier!
                proton_iron = False
                helium_oxygen = False
                proton_helium = False

            for smearing in smear_values:

                input_filename = f"{observatory_name}_{had_model_file}_EnergyCorrected_AllEnergiesAndZeniths_ForML.txt"
                input_file = input_file_path + input_filename
                output_file_name = f"{observatory_name}_{had_model_file}_{prim}_zen{zeniths[0]:.0f}_{zeniths[1]:.0f}"

                if smearing == True:
                    output_file = output_file_path + output_file_name + "_Smeared_GBDToutput.txt"
                else:
                    output_file = output_file_path + output_file_name + "_noSmearing_GBDToutput.txt"

                analysis = GBDTAnalysis(save_plots=make_plots, analyze_smeared_test_data=smearing)

                results = analysis.perform_gbdt_analysis(input_file, observatory_name, model, lg_e_bin_edges,
                                                         zenith_bins=zeniths, pFe=proton_iron, HeO=helium_oxygen,
                                                         pHe=proton_helium, smear=smearing, verbose=VERBOSE)
                if VERBOSE:
                    analysis.print_gbdt_results(lg_e_bin_edges, results[0], results[1], results[2], results[3])

                if SAVE_OUTPUT:
                    analysis.save_output(output_file, lg_e_bin_edges, results[0], results[1], results[2], results[3])

                    # Change name of feature importance file
                    if os.path.exists("model_output/feature_importances.txt"):
                        ft_importance_name = output_file_name.rsplit(".", 1)[0]
                        os.rename("model_output/feature_importances.txt", "model_output/" + ft_importance_name + "_feature_importances.txt")
