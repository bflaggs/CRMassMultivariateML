#!/usr/bin/env python3

######################################
# Description of file and file usage #
######################################

# This was supposed to be a file to read in all ASCII files and convert them to a format more suitable for the ML methods

# But now that I'm thinking about it, maybe it would be best if I incorporated a function in the main MultivatiateMass class
# instance that just saved the "vals" from function "GetValues()" to another file and used that...

# For now, still work on this...


# Run by executing the command...
# ./CondenseASCIIFiles.py PATH_TO_OG_ASCII_FILES --kwargs

# PATH_TO_OG_ASCII_FILES for diffrent experiments:
# IceCube -> /home/acoleman/gen2-surface/sim/mass/*.txt
# Auger   -> /pbs/home/b/bflaggs/SimulationWork/ParsedData/SIB23c/*/FILES_TO_READ
# Local (IceCube) -> /home/bflaggs/Documents/Research/MassSensitiveObservablesPaper/ASCIIFiles/IceCube/EMParticleProfileFits/*.txt
# Local (Auger)   -> /home/bflaggs/Documents/Research/MassSensitiveObservablesPaper/ASCIIFiles/Auger/EMParticleProfileFits/*/*.txt

# NOTE: Can concatenate all output .txt files into a single file using the bash command:
# head -n 1 ONE_OUTPUT_FILE > COMBINED_FILE; tail -n +2 -q ALL_OUTPUT_FILES >> COMBINED_FILE

######################
# End of description #
######################


import numpy as np
import os
from os import listdir
from os.path import isfile, join

ABS_PATH_HERE = str(os.path.dirname(os.path.realpath(__file__)))

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, nargs="+", default=[], help="List of CORSIKA simulation ASCII files")
parser.add_argument("--observatory", type=str, nargs="?", required=True, default="IceCube", help="Name of observatory (either IceCube or Auger)")
parser.add_argument("--electronXmaxScaling", action="store_true", help="If set will scale observables based on the electron number at Xmax")
args = parser.parse_args()

if args.observatory == "IceCube":
    muonsHE = "nMu>500GeV"
    muonsHEindex = 6
    observatory = "IceCube"
    scaleCorrection = 0.01 # Correction between lg(Ne) vs. lg(E) plot
    EeVnEMNormalization = 605741418.2773747 # zen = 0-72 deg (all zenith angles), lgE = 17.9-18.1
elif args.observatory == "Auger":
    muonsHE = "nMu>1GeV"
    muonsHEindex = 5
    observatory = "Auger"
    scaleCorrection = 0.01 # Correction between lg(Ne) vs. lg(E) plot
    EeVnEMNormalization = 586908936.4969574 # zen = 0-65 deg (Auger, all zenith angles), lgE = 17.9-18.1
else:
    raise ValueError("Can not set '--observatory' to anything other than 'IceCube' or 'Auger'.")

if args.electronXmaxScaling == False:
    raise ValueError("Must make energy corrected observable ASCII files by scaling observables w.r.t. electrons at Xmax. Or can update code...")


def ReadSingleFile(file):

    corsikaIDs = [14, 402, 1608, 5626]

    fileSplit = file.rsplit("/", 1)

    outPath = "/home/bflaggs/Documents/Research/MassSensitiveObservablesPaper/ASCIIFiles/ForML/" + observatory + "/EMParticleProfileFits/"
    outName = outPath + "EnergyCorrected_Condensed_" + fileSplit[1]
    outfile = open(outName, "w")

    if observatory == "IceCube":
        outfile.write(f"#ParticleID, E(GeV), zenith, nEM_Xmax, nEM_Obslev, nEM800m_NOTCORRECTED, {muonsHE}, R_eMuHighE, nMu800m, R_eMu800m, Xmax, SigmaXmax, R, SigmaR, L, SigmaL\n")
    elif observatory == "Auger":
        outfile.write(f"#ParticleID, E(GeV), zenith, nEM_Xmax, nEM_Obslev, nEM800m_NOTCORRECTED, {muonsHE}_NOTCORRECTED, R_eMuHighE_NOTCORRECTED, nMu800m, R_eMu800m, Xmax, SigmaXmax, R, SigmaR, L, SigmaL\n")

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

            energy = float(cols[1]) # In units of GeV
            if energy == 0.:
                continue
            #energyLog10 = np.log10(energy * 1e+9) # In units of log10(E / eV)

            zenith = float(cols[2]) # In units of radians
            #zenithDeg = float(cols[2]) * (180.0 / np.pi) # In units of degrees

            nEMxmax = float(cols[55])

            nEM800m = float(cols[48])
            nEM850m = float(cols[49])
            nEM800mRing = nEM850m - nEM800m

            nMuHighE = float(cols[muonsHEindex])

            nEMObslev = float(cols[32]) # Number of electrons/positrons at ground (at Obslev)
            ratio_eMuHighE = np.log10(nEMObslev) - np.log10(nMuHighE)

            nMu800m = float(cols[27])
            nMu850m = float(cols[28])
            nMu800mRing = nMu850m - nMu800m

            ratio_eMu800mRing = np.log10(nEM800mRing) - np.log10(nMu800mRing)

            xmaxFit = float(cols[69]) # Xmax from Andringa fit to .long file
            sigmaXmax = float(cols[70]) # Uncertainty in Xmax from Andringa fit

            rFit = float(cols[65]) # R from Andringa fit to .long file 
            sigmaR = float(cols[66]) # Uncertainty in R from Andringa fit
            
            lFit = float(cols[67]) # L from Andringa fit to .long file
            sigmaL = float(cols[68]) # Uncertainty in L from Andringa fit


            #==============================================================================
            # CHECK THIS IS RIGHT!!!!!!!!!! Should be log10(e / Mu) then scaled!
            #diffLgNEMObslevLgNMu850m = np.log10(nEM850mRing) - np.log10(nMu850mRing) - (0.09 + scaleCorrection)*np.log10(event.nEmAtXmax / EeVnEMNormalization)

            # ALSO MAKE NEW HISTOGRAMS FOR L VALUES OVER EXTENDED ENERGY RANGE AT AUGER!!!!!!!!!!!!!!
            # Maybe remove cut on L values for all future FOM plots...... but first check to see if there is any major difference
            #==============================================================================

            if observatory == "IceCube":
                nEMObslev = np.log10(nEMObslev / (nEMxmax / EeVnEMNormalization) ** (1.13 + scaleCorrection))
                nEM800mRing = nEM800mRing
                nMuHighE = np.log10(nMuHighE / (nEMxmax / EeVnEMNormalization) ** (0.82 + scaleCorrection))                
                ratio_eMuHighE = ratio_eMuHighE - (0.31 + scaleCorrection)*np.log10(nEMxmax / EeVnEMNormalization)
                nMu800mRing = np.log10(nMu800mRing / (nEMxmax / EeVnEMNormalization) ** (0.93 + scaleCorrection))
                ratio_eMu800mRing = ratio_eMu800mRing - (0.09 + scaleCorrection)*np.log10(nEMxmax / EeVnEMNormalization)
                xmaxFit = xmaxFit - (62.01 + scaleCorrection)*np.log10(nEMxmax / EeVnEMNormalization)
                rFit = rFit - (-0.03 + scaleCorrection)*np.log10(nEMxmax / EeVnEMNormalization)
                lFit = lFit - (7.18 + scaleCorrection)*np.log10(nEMxmax / EeVnEMNormalization)
            
            elif observatory == "Auger":
                nEMObslev = np.log10(nEMObslev / (nEMxmax / EeVnEMNormalization) ** (1.16 + scaleCorrection))
                nEM800mRing = nEM800mRing
                nMuHighE = np.log10(nMuHighE) # Not corrected for Auger b/c not used           
                ratio_eMuHighE = ratio_eMuHighE # Not corrected for Auger b/c not used
                nMu800mRing = np.log10(nMu800mRing / (nEMxmax / EeVnEMNormalization) ** (0.93 + scaleCorrection))
                ratio_eMu800mRing = ratio_eMu800mRing - (0.11 + scaleCorrection)*np.log10(nEMxmax / EeVnEMNormalization)
                xmaxFit = xmaxFit - (62.82 + scaleCorrection)*np.log10(nEMxmax / EeVnEMNormalization)
                rFit = rFit - (-0.03 + scaleCorrection)*np.log10(nEMxmax / EeVnEMNormalization)
                lFit = lFit - (7.47 + scaleCorrection)*np.log10(nEMxmax / EeVnEMNormalization)


            outfile.write(f"{particleID} {energy} {zenith} {nEMxmax} {nEMObslev} {nEM800mRing} {nMuHighE} {ratio_eMuHighE} {nMu800mRing} {ratio_eMu800mRing} {xmaxFit} {sigmaXmax} {rFit} {sigmaR} {lFit} {sigmaL}\n")

    outfile.close()


for filename in args.input:
    ReadSingleFile(filename)




