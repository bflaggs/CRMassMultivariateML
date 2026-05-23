#!/usr/bin/env python3

######################################
# Description of file and file usage #
######################################

# This was supposed to be a file to read in all ASCII files and convert them to a format more suitable for the ML methods

# But now that I'm thinking about it, maybe it would be best if I incorporated a function in the main MultivatiateMass class
# instance that just saved the "vals" from function "GetValues()" to another file and used that...

# --- BSF 22/05/2026 ---
# Once I refactor the code and make it modularized then I should write a function like described above
# It will make this file and the energy correction brother version of this file obsolete :)
# ----------------------

# Run by executing the command...
# ./CondenseASCIIFiles.py PATH_TO_ASCII_FILES --kwargs

# PATH_TO_ASCII_FILES potential location:
# /home/bflaggs/Documents/Research/MassSensitiveObservablesPaper/ASCIIFiles/NextGenModelsAuger/MODEL_NAME/*/*.txt

# NOTE: Can concatenate all output .txt files into a single file using the bash command:
# head -n 1 ONE_OUTPUT_FILE > COMBINED_FILE; tail -n +2 -q ALL_OUTPUT_FILES >> COMBINED_FILE

######################
# End of description #
######################

import numpy as np
import os
import argparse

ABS_PATH_HERE = str(os.path.dirname(os.path.realpath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, nargs="+", default=[], help="List of CORSIKA simulation ASCII files")
parser.add_argument("--observatory", type=str, nargs="?", required=True, default="Auger", help="Name of observatory")
parser.add_argument("--model", type=str, nargs="?", required=True, default="EPOS LHC-R", help="Name of hadronic model")
parser.add_argument("--electronXmaxScaling", action="store_true", help="If set will scale observables based on the electron number at Xmax")
args = parser.parse_args()

if args.observatory == "IceCube":
    muonsHE = "nMu>500GeV"
    muonsHEindex = 6
    observatory = "IceCube"
elif args.observatory == "Auger":
    muonsHE = "nMu>1GeV"
    muonsHEindex = 5
    observatory = "Auger"
else:
    raise ValueError("Can not set '--observatory' to anything other than 'IceCube' or 'Auger'.")

# Define constants for scaling each observable based on the selected model
if args.model == "EPOS LHC-R":
    hadronic_model = "EPOSLHCR"
    scale_correction = 0.01
    eev_norm = 585508499.8882083  # zen = 0-65 deg (Auger, all zenith angles), lgE = 17.9-18.1
    em_obslev_shift = 1.13 + scale_correction
    mu_ann_shift = 0.93 + scale_correction - 0.01
    em_mu_ratio_ann_shift = 0.11 + scale_correction
    xmax_shift = 60.02 + scale_correction
    r_shift = -0.03 + scale_correction
    l_shift = 6.44 + scale_correction
elif args.model == "Sibyll 2.3e":
    hadronic_model = "Sibyll23e"
    scale_correction = 0.00
    eev_norm = 619514957.1267722  # zen = 0-65 deg (Auger, all zenith angles), lgE = 17.9-18.1
    em_obslev_shift = 1.16 + scale_correction
    mu_ann_shift = 0.93  # slightly different for Sibyll 2.3e
    em_mu_ratio_ann_shift = 0.13 + scale_correction
    xmax_shift = 60.71 + scale_correction
    r_shift = -0.03 + scale_correction
    l_shift = 7.75 + scale_correction
elif args.model == "QGSJETIII-01":
    hadronic_model = "QGSJETIII01"
    scale_correction = 0.01
    eev_norm = 621701443.5937785  # zen = 0-65 deg (Auger, all zenith angles), lgE = 17.9-18.1
    em_obslev_shift = 1.14 + scale_correction
    mu_ann_shift = 0.93 + scale_correction - 0.01
    em_mu_ratio_ann_shift = 0.13 + scale_correction
    xmax_shift = 58.17 + scale_correction
    r_shift = -0.03 + scale_correction
    l_shift = 6.28 + scale_correction
else:
    raise ValueError("Can not set '--model' to anything other than 'EPOS LHC-R', 'Sibyll 2.3e', or 'QGSJETIII-01'.")


if args.electronXmaxScaling:
    energy_correction = "_EnergyCorrected"
else:
    energy_correction = ""

def make_condensed_file(file):
    """
    Makes a condensed ASCII file from the full file output from the CorsikaParser scripts.
    Output will be used for training ML models.
    """

    corsikaIDs = [14, 402, 1608, 5626]  # Proton, Helium, Oxygen, and Iron (in this order)

    fileSplit = file.rsplit("/", 1)

    outPath = "/home/bflaggs/Documents/Research/MassSensitiveObservablesPaper/ASCIIFiles/ForML/NextGenModels/"
    outName = outPath + observatory + energy_correction + "_Condensed_" + fileSplit[1]
    outfile = open(outName, "w")
    if args.electronXmaxScaling:
        outfile.write(f"#ParticleID, E(GeV), zenith, nEM_Xmax, nEM_Obslev, nEM800m_NOTCORRECTED, {muonsHE}_NOTCORRECTED, R_eMuHighE_NOTCORRECTED, nMu800m, R_eMu800m, Xmax, SigmaXmax, R, SigmaR, L, SigmaL\n")
    else:
        outfile.write(f"#ParticleID, E(GeV), zenith, nEM_Xmax, nEM_Obslev, nEM800m, {muonsHE}, R_eMuHighE, nMu800m, R_eMu800m, Xmax, SigmaXmax, R, SigmaR, L, SigmaL\n")


    with open(file, "r") as file:
        for line in file:
            if line[0] == "#":
                continue

            cols = line.split()

            if len(cols) != 71:
                continue

            particleID = int(cols[0])
            if particleID not in corsikaIDs:
                continue

            energy = float(cols[1])  # In units of GeV
            if energy == 0.:
                continue

            zenith = float(cols[2])  # In units of radians

            nEMxmax = float(cols[55])

            nEM800m = float(cols[48])
            nEM850m = float(cols[49])
            nEM800mRing = nEM850m - nEM800m

            nMuHighE = float(cols[muonsHEindex])

            nEMObslev = float(cols[32])  # Number of electrons/positrons at ground (at Obslev)
            ratio_eMuHighE = np.log10(nEMObslev) - np.log10(nMuHighE)

            nMu800m = float(cols[27])
            nMu850m = float(cols[28])
            nMu800mRing = nMu850m - nMu800m

            ratio_eMu800mRing = np.log10(nEM800mRing) - np.log10(nMu800mRing)

            xmaxFit = float(cols[69])  # Xmax from Andringa fit to .long file
            sigmaXmax = float(cols[70])  # Uncertainty in Xmax from Andringa fit

            rFit = float(cols[65])  # R from Andringa fit to .long file 
            sigmaR = float(cols[66])  # Uncertainty in R from Andringa fit
            
            lFit = float(cols[67])  # L from Andringa fit to .long file
            sigmaL = float(cols[68])  # Uncertainty in L from Andringa fit

            if args.electronXmaxScaling:
                nEMObslev = np.log10(nEMObslev / (nEMxmax / eev_norm) ** (em_obslev_shift))
                nEM800mRing = nEM800mRing  # Do not correct b/c will not use
                nMuHighE = np.log10(nMuHighE)  # Not corrected for Auger b/c not used (not for now...)        
                ratio_eMuHighE = ratio_eMuHighE  # Not corrected for Auger b/c not used (not for now...)
                nMu800mRing = np.log10(nMu800mRing / (nEMxmax / eev_norm) ** (mu_ann_shift))
                ratio_eMu800mRing = ratio_eMu800mRing - (em_mu_ratio_ann_shift)*np.log10(nEMxmax / eev_norm)
                xmaxFit = xmaxFit - (xmax_shift)*np.log10(nEMxmax / eev_norm)
                rFit = rFit - (r_shift)*np.log10(nEMxmax / eev_norm)
                lFit = lFit - (l_shift)*np.log10(nEMxmax / eev_norm)


            outfile.write(f"{particleID} {energy} {zenith} {nEMxmax} {nEMObslev} {nEM800mRing} {nMuHighE} {ratio_eMuHighE} {nMu800mRing} {ratio_eMu800mRing} {xmaxFit} {sigmaXmax} {rFit} {sigmaR} {lFit} {sigmaL}\n")

    outfile.close()

for filename in args.input:
    make_condensed_file(filename)
