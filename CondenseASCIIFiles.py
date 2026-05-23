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
#parser.add_argument("--zenithRange", type=float, nargs=2, default=[40.0, 50.0], help="Zenith range of data to plot")
#parser.add_argument("--energyRange", type=float, nargs=2, default=[16.5, 16.9], help="lg(E) energy range of data to plot")
#parser.add_argument("--energyScaling", action="store_true", help="If set, will scale observables based on the energy of the initiating shower")
#parser.add_argument("--electronXmaxScaling", action="store_true", help="If set, will scale observables based on the electron number at Xmax")
#parser.add_argument("--electronObslevScaling", action="store_true", help="If set, will scale observables based on the electron number at ground")
#parser.add_argument("--applyScaling", action="store_true", help="If set, will scale observables based on additional keyword supplied by user")
#parser.add_argument("--applyDataCuts", action="store_true", help="If set, will cut out events with unconstrained/poor profile fit parameters")
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


def ReadSingleFile(file):

    corsikaIDs = [14, 402, 1608, 5626]

    fileSplit = file.rsplit("/", 1)


    outPath = "/home/bflaggs/Documents/Research/MassSensitiveObservablesPaper/ASCIIFiles/ForML/" + observatory + "/EMParticleProfileFits/"
    outName = outPath + "Condensed_" + fileSplit[1]
    outfile = open(outName, "w")
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


            outfile.write(f"{particleID} {energy} {zenith} {nEMxmax} {nEMObslev} {nEM800mRing} {nMuHighE} {ratio_eMuHighE} {nMu800mRing} {ratio_eMu800mRing} {xmaxFit} {sigmaXmax} {rFit} {sigmaR} {lFit} {sigmaL}\n")

    outfile.close()


for filename in args.input:
    ReadSingleFile(filename)








# Save the below so I have an idea of the data cuts...

'''
    def GetValues(self):

        prevZen = 0
        prevAzi = 0

        if not len(self.eventList):
            print("No events were loaded which passed the cuts!")

        for event in self.eventList:
            # Apply cut for xmax because very large xmax values are unphysical

            if not 0 < event.xmax < 1500:
                continue

            if self.observatoryName == "IceCube":
                if event.n500GeVMuObslev < 1:
                    if not self.warn500:
                        print("Warning: found an event without a 500 GeV muon")
                    self.warn500 = True
                    continue

            if event.nMuonsObslev < 1:
                if not self.warnMuAll:
                    print("Warning: found an event without any muons")
                self.warnMuAll = True
                continue

            elif (self.flagDataCuts == True) and (self.flagGHFits == False):
                if event.sigmaXmaxfitAndringa == np.inf or event.sigmaRfitAndringa == np.inf or event.sigmaLfitAndringa == np.inf:
                    continue
                elif event.sigmaXmaxfitAndringa > 5.0 or event.sigmaRfitAndringa > 0.05 or event.sigmaLfitAndringa > 5.0:
                    continue
                elif event.RfitAndringa < 0.0 or event.LfitAndringa > 350.0:  # Maybe also include a cut on L values? (i.e. L < 350 or L < 325???)
                    continue



                elif self.muonEnergyScaling == 0.93 and self.highEmuonEnergyScaling == 0.82:
                    # For these parameters then scale w.r.t. the electron number at Xmax
                    if self.observatoryName == "IceCube":
                        scaleCorrection = 0.01 # Correction between lg(Ne) vs. lg(E) plot
                        EeVnEMNormalization = 605741418.2773747 # zen = 0-72 deg (all zenith angles), lgE = 17.9-18.1
                    elif self.observatoryName == "Auger":
                        scaleCorrection = 0.01 # Correction between lg(Ne) vs. lg(E) plot
                        EeVnEMNormalization = 586908936.4969574 # zen = 0-65 deg (Auger, all zenith angles), lgE = 17.9-18.1

                    lgNMuTotCorr = np.log10(event.nMuonsObslev / (event.nEmAtXmax / EeVnEMNormalization) ** (self.muonEnergyScaling + scaleCorrection))
                    lgNMuHighCorr = np.log10(event.n500GeVMuObslev / (event.nEmAtXmax / EeVnEMNormalization) ** (self.highEmuonEnergyScaling + scaleCorrection))

                    lgNEM = np.log10(event.nEmAtXmax)
                    diffLgNEMLgNMuTot = np.log10(event.nEmAtXmax) - np.log10(event.nMuonsObslev) - (0.07 + scaleCorrection)*np.log10(event.nEmAtXmax / EeVnEMNormalization)
                    diffLgNEMLgNMuHigh = np.log10(event.nEmAtXmax) - np.log10(event.n500GeVMuObslev) - (0.18 + scaleCorrection)*np.log10(event.nEmAtXmax / EeVnEMNormalization)

                    if self.observatoryName == "IceCube":
                        diffLgNEMObslevLgNMuTot = np.log10(event.nEmObslev) - np.log10(event.nMuonsObslev) - (0.20 + scaleCorrection)*np.log10(event.nEmAtXmax / EeVnEMNormalization)
                        lgNEMObslevCorr = np.log10(event.nEmObslev / (event.nEmAtXmax / EeVnEMNormalization) ** (1.13 + scaleCorrection))
                    elif self.observatoryName == "Auger":
                        diffLgNEMObslevLgNMuTot = np.log10(event.nEmObslev) - np.log10(event.nMuonsObslev) - (0.23 + scaleCorrection)*np.log10(event.nEmAtXmax / EeVnEMNormalization)
                        lgNEMObslevCorr = np.log10(event.nEmObslev / (event.nEmAtXmax / EeVnEMNormalization) ** (1.16 + scaleCorrection))

                    diffLgNEMObslevLgNMuHigh = np.log10(event.nEmObslev) - np.log10(event.n500GeVMuObslev) - (0.31 + scaleCorrection)*np.log10(event.nEmAtXmax / EeVnEMNormalization)

                    lgNMu50mCorr = np.log10(nMu50mRing / (event.nEmAtXmax / EeVnEMNormalization) ** (self.muonEnergyScaling + scaleCorrection))
                    lgNMu100mCorr = np.log10(nMu100mRing / (event.nEmAtXmax / EeVnEMNormalization) ** (self.muonEnergyScaling + scaleCorrection))
                    lgNMu150mCorr = np.log10(nMu150mRing / (event.nEmAtXmax / EeVnEMNormalization) ** (self.muonEnergyScaling + scaleCorrection))
                    lgNMu200mCorr = np.log10(nMu200mRing / (event.nEmAtXmax / EeVnEMNormalization) ** (self.muonEnergyScaling + scaleCorrection))
                    lgNMu250mCorr = np.log10(nMu250mRing / (event.nEmAtXmax / EeVnEMNormalization) ** (self.muonEnergyScaling + scaleCorrection))
                    lgNMu300mCorr = np.log10(nMu300mRing / (event.nEmAtXmax / EeVnEMNormalization) ** (self.muonEnergyScaling + scaleCorrection))
                    lgNMu350mCorr = np.log10(nMu350mRing / (event.nEmAtXmax / EeVnEMNormalization) ** (self.muonEnergyScaling + scaleCorrection))
                    lgNMu400mCorr = np.log10(nMu400mRing / (event.nEmAtXmax / EeVnEMNormalization) ** (self.muonEnergyScaling + scaleCorrection))
                    lgNMu450mCorr = np.log10(nMu450mRing / (event.nEmAtXmax / EeVnEMNormalization) ** (self.muonEnergyScaling + scaleCorrection))
                    lgNMu500mCorr = np.log10(nMu500mRing / (event.nEmAtXmax / EeVnEMNormalization) ** (self.muonEnergyScaling + scaleCorrection))
                    lgNMu550mCorr = np.log10(nMu550mRing / (event.nEmAtXmax / EeVnEMNormalization) ** (self.muonEnergyScaling + scaleCorrection))
                    lgNMu600mCorr = np.log10(nMu600mRing / (event.nEmAtXmax / EeVnEMNormalization) ** (self.muonEnergyScaling + scaleCorrection))
                    lgNMu650mCorr = np.log10(nMu650mRing / (event.nEmAtXmax / EeVnEMNormalization) ** (self.muonEnergyScaling + scaleCorrection))
                    lgNMu700mCorr = np.log10(nMu700mRing / (event.nEmAtXmax / EeVnEMNormalization) ** (self.muonEnergyScaling + scaleCorrection))
                    lgNMu750mCorr = np.log10(nMu750mRing / (event.nEmAtXmax / EeVnEMNormalization) ** (self.muonEnergyScaling + scaleCorrection))
                    lgNMu800mCorr = np.log10(nMu800mRing / (event.nEmAtXmax / EeVnEMNormalization) ** (self.muonEnergyScaling + scaleCorrection))
                    lgNMu850mCorr = np.log10(nMu850mRing / (event.nEmAtXmax / EeVnEMNormalization) ** (self.muonEnergyScaling + scaleCorrection))
                    lgNMu900mCorr = np.log10(nMu900mRing / (event.nEmAtXmax / EeVnEMNormalization) ** (self.muonEnergyScaling + scaleCorrection))
                    lgNMu950mCorr = np.log10(nMu950mRing / (event.nEmAtXmax / EeVnEMNormalization) ** (self.muonEnergyScaling + scaleCorrection))
                    lgNMu1000mCorr = np.log10(nMu1000mRing / (event.nEmAtXmax / EeVnEMNormalization) ** (self.muonEnergyScaling + scaleCorrection))

                    # Take 800-850m ring as nominal, partially motivated by muon density paper (arxiv: 2201.12635)
                    diffLgNEMLgNMu850m = np.log10(event.nEmAtXmax) - np.log10(nMu850mRing) - (0.07 + scaleCorrection)*np.log10(event.nEmAtXmax / EeVnEMNormalization)

                    if self.observatoryName == "IceCube":
                        diffLgNEMObslevLgNMu850m = np.log10(nEM850mRing) - np.log10(nMu850mRing) - (0.09 + scaleCorrection)*np.log10(event.nEmAtXmax / EeVnEMNormalization)
                    elif self.observatoryName == "Auger":
                        diffLgNEMObslevLgNMu850m = np.log10(nEM850mRing) - np.log10(nMu850mRing) - (0.11 + scaleCorrection)*np.log10(event.nEmAtXmax / EeVnEMNormalization)



                    if self.observatoryName == "IceCube":
                        XmaxvalCorr = event.XmaxfitAndringa - (62.01 + scaleCorrection)*np.log10(event.nEmAtXmax / EeVnEMNormalization)
                        RvalCorr = event.RfitAndringa - (-0.03 + scaleCorrection)*np.log10(event.nEmAtXmax / EeVnEMNormalization)
                        LvalCorr = event.LfitAndringa - (7.18 + scaleCorrection)*np.log10(event.nEmAtXmax / EeVnEMNormalization)
                    elif self.observatoryName == "Auger":
                        XmaxvalCorr = event.XmaxfitAndringa - (62.82 + scaleCorrection)*np.log10(event.nEmAtXmax / EeVnEMNormalization)
                        RvalCorr = event.RfitAndringa - (-0.03 + scaleCorrection)*np.log10(event.nEmAtXmax / EeVnEMNormalization)
                        LvalCorr = event.LfitAndringa - (7.47 + scaleCorrection)*np.log10(event.nEmAtXmax / EeVnEMNormalization)

                        if XmaxvalCorr == np.nan or XmaxvalCorr == np.inf:
                            print(f"Bad value found! With Xmax={event.XmaxfitAndringa}, EMatXmax={event.nEmAtXmax}")
'''

