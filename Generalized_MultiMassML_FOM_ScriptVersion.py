#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pprint import pprint
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib import rc, rcParams

import os

rc("font", size=18.0)
rcParams["font.family"] = "serif"
rcParams["mathtext.fontset"] = "dejavuserif"

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

from sklearn.model_selection import RandomizedSearchCV

#### In notebook, each function is it's own cell ####

# Function to read in the data and do some renaming of columns
def ReadData(filename):
    # Check file exists
    if os.path.exists(filename):
        df = pd.read_csv(filename, sep=" ")
    else:
        raise ValueError(f"File {filename} does not exist. Check path.")

    # Rename some columns to make things easier for future...
    df.columns = ["ParticleID", "EnergyGeV", "zenith", "nEM_Xmax", "nEM_Obslev", "nEM800m_NOTCORRECTED", "nMuHighE", "R_eMuHighE", "nMu800m", "R_eMu800m", "Xmax", "SigmaXmax", "R", "SigmaR", "L", "SigmaL"]

    return df


# Function to apply same data cuts as the Fisher analysis to the data
def ApplyFisherDataCuts(dataset, ProtonIron=False, HeliumOxygen=False, ProtonHelium=False):
    if ProtonIron + HeliumOxygen + ProtonHelium > 1:
        raise ValueError("Can only set one (or none) of the three keywords for mass comparison.")

    # Drop bad data, same cuts as my Fisher analysis
    dfCuts = dataset[(dataset.nMuHighE != -np.inf) &
                    (dataset.nMu800m != -np.inf) &
                    (dataset.R_eMu800m != -np.inf) &  # Should be same as cutting all events w/ nEM800m <= 0.0
                    (dataset.SigmaXmax != np.inf) &
                    (dataset.SigmaR != np.inf) &
                    (dataset.SigmaL != np.inf) &
                    (dataset.SigmaXmax < 5.0) &
                    (dataset.SigmaR < 0.05) &
                    (dataset.SigmaL < 5.0) &
                    (dataset.R > 0.0) &
                    (dataset.Xmax < 1500.0)]

    if ProtonIron == True:
        # Cut He and O
        dfCuts = dfCuts[(dfCuts.ParticleID != 402) &
                        (dfCuts.ParticleID != 1608)]
        # Convert p and Fe ParticlsIDs to binaries
        dfCuts.ParticleID[dfCuts.ParticleID == 5626] = 0
        dfCuts.ParticleID[dfCuts.ParticleID == 14] = 1
        plotLabels = ['Proton', 'Iron']
        plotColors = ['#CC6677', '#4477AA']

    elif HeliumOxygen == True:
        # Cut p and Fe
        dfCuts = dfCuts[(dfCuts.ParticleID != 14) &
                        (dfCuts.ParticleID != 5626)]
        # Convert He and O ParticlsIDs to binaries
        dfCuts.ParticleID[dfCuts.ParticleID == 1608] = 0
        dfCuts.ParticleID[dfCuts.ParticleID == 402] = 1
        plotLabels = ['Helium', 'Oxygen']
        plotColors = ['#DDCC77', '#117733']

    elif ProtonHelium == True:
        # Cut O and Fe
        dfCuts = dfCuts[(dfCuts.ParticleID != 1608) &
                        (dfCuts.ParticleID != 5626)]
        # Convert p and He ParticlsIDs to binaries
        dfCuts.ParticleID[dfCuts.ParticleID == 402] = 0
        dfCuts.ParticleID[dfCuts.ParticleID == 14] = 1
        plotLabels = ['Proton', 'Helium']
        plotColors = ['#CC6677', '#DDCC77']

    else:
        print("No keyword for separating two primary types. Instead will try separating protons from heavier primaries!")
        # Convert p and heavier ParticleIDs to binaries
        dfCuts.ParticleID[dfCuts.ParticleID != 14] = 0 # Must do this first!
        dfCuts.ParticleID[dfCuts.ParticleID == 14] = 1
        plotLabels = ['Proton', 'Heavier']
        plotColors = ['#CC6677', '#4477AA']

    return dfCuts, plotLabels, plotColors


# Function to apply zenith angle cuts
def ApplyZenithCut(dataset, minZen=40.0, maxZen=60.0):
    dfZenCuts = dataset[((dataset.zenith * 180. / np.pi) >= minZen) &
                        ((dataset.zenith * 180. / np.pi) <= maxZen)]
    return dfZenCuts


# Function to apply energy bin cuts, used to split data in energy so that multiple models can be trained
def BinByEnergy(dataset, minLgE=16.0, maxLgE=20.5):
    dfEnergyBinned = dataset[(np.log10(dataset.EnergyGeV * 1e+9) >= minLgE) &
                            (np.log10(dataset.EnergyGeV * 1e+9) <= maxLgE)]
    return dfEnergyBinned


# Function to drop columns not used in Fisher analysis and separate array of particle labels (goal) from data
def GetDataForGBDT(dataset, dropZenith=False):
    if dropZenith == True:
        dfGBDT = dataset.drop(columns=["EnergyGeV", "zenith", "nEM_Xmax", "nEM800m_NOTCORRECTED", "nMuHighE", "R_eMuHighE", "SigmaXmax", "SigmaR", "SigmaL"])
        smearLevels = [0.1, 0.1, 0.14, 20.0, 0.05, 5.0]
    else:
        dfGBDT = dataset.drop(columns=["EnergyGeV", "nEM_Xmax", "nEM800m_NOTCORRECTED", "nMuHighE", "R_eMuHighE", "SigmaXmax", "SigmaR", "SigmaL"])
        smearLevels = [2.0 * np.pi / 180.0, 0.1, 0.1, 0.14, 20.0, 0.05, 5.0]

    print("WARNING: To keep the high-energy muon number this function should be updated and the observable studied in more detail.")

    # Define another array which contains the labels of the particles
    particleIDs = np.array(dfGBDT["ParticleID"])

    # Remove the IDs for the particles from the dataset, so that model can be trained
    dfGBDT.drop(columns=["ParticleID"], inplace=True)

    return dfGBDT, particleIDs, smearLevels


# Function to get the feature importances and print them nicely to the screen
def PrintFeatureImportances(importance_list, feature_list):
    if type(feature_list) != list:
        feature_list = list(feature_list)

    ft_importances = [(feature, round(importance, 3)) for feature, importance in zip(feature_list, importance_list)]
    ft_importances = sorted(ft_importances, key = lambda x:x[1], reverse=True)
    [print("Observable: {:20} Importance: {}".format(*pair)) for pair in ft_importances]


# Function to smear data (used to simulate detector response, i.e. true reconstructions)
def SmearData(original_array, smear_values):
    if len(smear_values) != original_array.shape[1]:
        raise ValueError("Shape of smearing array values and data to smear is not the same!")

    smeared_array = np.copy(original_array)

    for i in range(original_array.shape[0]):
        for j in range(original_array.shape[1]):
            smeared_array[i][j] += stats.norm.rvs(loc=0.0, scale=smear_values[j])

    return smeared_array


# Function to extract FOM value
def GetFOM(GBDT, testData, testIDs, plotLabels, smearValues=False, smearArray=None, printOutput=False):
    if smearValues == True:
        testData_smeared = SmearData(testData, smearArray)
        gbdt_pred_test = GBDT.predict(testData_smeared)
    else:
        gbdt_pred_test = GBDT.predict(testData)

    pop1_gbdt_pred = gbdt_pred_test[np.nonzero(testIDs == 1)]
    pop2_gbdt_pred = gbdt_pred_test[np.nonzero(testIDs == 0)]

    reverseSorted_pop2 = np.sort(pop2_gbdt_pred)[::-1]
    pop2_5perOverlap = reverseSorted_pop2[:int(len(pop2_gbdt_pred)* 0.05)]
    pop1_5perContamination = pop1_gbdt_pred[np.nonzero(pop1_gbdt_pred >= pop2_5perOverlap[-1])]

    pop2_1perOverlap = reverseSorted_pop2[:int(len(pop2_gbdt_pred)* 0.01)]
    pop1_1perContamination = pop1_gbdt_pred[np.nonzero(pop1_gbdt_pred >= pop2_1perOverlap[-1])]

    # Could update these to quantiles like so:
    #pop2_thresh5 = np.quantile(pop2_gbdt_pred, 0.95)
    #pop1_5perContamination = pop1_gbdt_pred[pop1_gbdt_pred >= pop2_thresh5]

    frac_sep_pop1_5perContamination = len(pop1_5perContamination) / len(pop1_gbdt_pred)
    frac_sep_pop1_1perContamination = len(pop1_1perContamination) / len(pop1_gbdt_pred)

    mean_pop1 = np.mean(pop1_gbdt_pred)
    mean_pop2 = np.mean(pop2_gbdt_pred)
    std_pop1 = np.std(pop1_gbdt_pred)
    std_pop2 = np.std(pop2_gbdt_pred)

    fom_gbdt = np.abs(mean_pop1 - mean_pop2) / np.sqrt(std_pop1**2 + std_pop2**2)

    # NOTE: Should also calculated an AUC score and directly compare the GBDT score to the linear discriminant score

    if printOutput == True:
        if smearValues == True:
            print("For smeared values...")
        else:
            print("For true values...")

        print(f"    The GBDT FOM between {plotLabels[0]} and {plotLabels[1]} is: {fom_gbdt:.3f}")
        print(f"    The separable fraction of {plotLabels[0]} with 5% contamination from {plotLabels[1]} is: {frac_sep_pop1_5perContamination:.3f}")
        print(f"    The separable fraction of {plotLabels[0]} with 1% contamination from {plotLabels[1]} is: {frac_sep_pop1_1perContamination:.3f}")

    return fom_gbdt, frac_sep_pop1_5perContamination, frac_sep_pop1_1perContamination, pop1_gbdt_pred, pop2_gbdt_pred

###### Make function here that trains a GBDT (w/ option for training on smeared or true)
###### and then loops through the test sets (w/ diff smear or true option) and
###### then calculates the FOM and separation for each energy bin that the test
###### set can be split into (should be in bin widths of 0.2 in lgE). Can then
###### go ahead and have another function which makes the histogram for these
###### binned values......

###### New histogram fucntion should be generalized to the point where it only
###### takes in the test populations data set run through the ML algorithm and
###### then can calculate the FOM and contamination factors individually
###### Maybe then it would be best to have a function that just runs the populations
###### through the ML algorithm then returns the result...

# Function that does the above... (histograms)
#def GeneralizedHistogram(pop1, pop2):


# Function to make the output histogram
def MakeHistogram(observatory, hadronic_model, pop1_pred, pop2_pred, fom, cont_5per, cont_1per, plotLabels, plotColors, lgEbins=[16.0,20.5], smearValues=False, smearArray=None):

    plt.figure(figsize=(18.0 / 2.54, 15.0 / 2.54))
    n2, bins2, patches2 = plt.hist(pop2_pred, bins=100, histtype="step", color=plotColors[1], label=plotLabels[1])
    n1, bins1, patches1 = plt.hist(pop1_pred, bins=100, histtype="step", color=plotColors[0], label=plotLabels[0])

    if max(n2) > max(n1):
        max_counts = max(n2)
    else:
        max_counts = max(n1)

    if plotLabels[0] == "Proton":
        sep_pop_string = "p"
    elif plotLabels[0] == "Helium":
        sep_pop_string = "He"

    reverseSorted_pop2 = np.sort(pop2_pred)[::-1]
    pop2_5perOverlap = reverseSorted_pop2[:int(len(pop2_pred)* 0.05)]
    pop2_1perOverlap = reverseSorted_pop2[:int(len(pop2_pred)* 0.01)]

    plt.vlines(pop2_5perOverlap[-1], 0, (max_counts / 3), colors="black", linestyle="dashed", label="5% Contamination")
    plt.vlines(pop2_1perOverlap[-1], 0, (max_counts / 3), colors="black", linestyle="dotted", label="1% Contamination")
    plt.text(0.1, (max_counts / 2), f"Frac. Sep. {sep_pop_string} = {cont_5per:.3f} @ 5% cont.", fontsize=12)
    plt.text(0.1, (max_counts / 2.2), f"Frac. Sep. {sep_pop_string} = {cont_1per:.3f} @ 1% cont.", fontsize=12)
    plt.text(0.05, (max_counts / 1.5), f"{hadronic_model} (FOM = {fom:.2f})")
    plt.text(0.05, (max_counts / 1.5) - 0.06, f"{observatory}")
    plt.legend(loc="best", fontsize=14)
    if smearValues == True:
        plt.xlabel("GBT Regressor Output (Smeared)")
    else:
        plt.xlabel("GBT Regressor Output (True)")
    plt.ylabel("Counts")
    plt.title(rf"log$_{{10}}$(E / eV) = {lgEbins[0]:.1f}-{lgEbins[1]:.1f}")
    plt.show()


# Setup another function which will be used to initialize the GBDT w/ a keyword option for setting up a smeared GBDT or not.
def SetupGBDT(og_df, primaries, smearedGBDT=False, analyzeModel=True, analyzeForSmearedTestData=False):

    print(f"Model trained to separate {primaries[0]} from {primaries[1]}.")

    # Define another array which contains the labels of the particles
    objectives = np.array(og_df["ParticleID"])

    # First do the train-test split of the data
    trainData, testData, trainObjectives, testObjectives = train_test_split(og_df.values, objectives, test_size=0.2, shuffle=True, random_state=42)

    # Now need to convert arrays to DataFrames and drop non-used things in analysis
    df_train = pd.DataFrame(trainData, columns=list(og_df.columns))
    df_test = pd.DataFrame(testData, columns=list(og_df.columns))

    df_train, train_obj, smearLevels = GetDataForGBDT(df_train, dropZenith=True)
    df_test, test_obj, smearLevels = GetDataForGBDT(df_test, dropZenith=True)

    if (trainObjectives != train_obj).any() or (testObjectives != test_obj).any():
        raise ValueError("Inconsistent objectives for train/test sets in train-test splitting and data cuts phases.")

    train_array = df_train.to_numpy()
    test_array = df_test.to_numpy()

    if smearedGBDT == True:
        print("Training the GBDT on smeared data to simulate detector resolutions...")
        train_array = SmearData(train_array, smearLevels)

    gbt_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)
    print("Training the model...")
    gbt_model.fit(train_array, trainObjectives)

    if analyzeModel == True:
        AnalyzeGBDT(gbt_model, train_array, trainObjectives, pd.DataFrame(testData, columns=list(og_df.columns)), smearedModel=smearedGBDT, smearTestData=analyzeForSmearedTestData)

    if smearedGBDT == True:
        print("\n")
        print("WARNING: Returned train and test data frames will not be smeared! (But returned training array will be.)")

    return gbt_model, pd.DataFrame(trainData, columns=list(og_df.columns)), pd.DataFrame(testData, columns=list(og_df.columns)), smearLevels, train_array, trainObjectives


# Function to analyze the GBDT, i.e. output model scores and list feature importances
def AnalyzeGBDT(trained_model, trainingData, trainingObjectives, original_testData_df, smearedModel=False, smearTestData=False):

    df_testdata, testingObjectives, smearLevels = GetDataForGBDT(original_testData_df, dropZenith=True)
    testingData = df_testdata.to_numpy()

    if smearTestData == True:
        testingData = SmearData(testingData, smearLevels)

    train_score = trained_model.score(trainingData, trainingObjectives)
    test_score = trained_model.score(testingData, testingObjectives)

    print(f"The scores of the GBT Regressor: {train_score:.4f} (training set), {test_score:.4f} (testing set)")

    if smearedModel == True and smearTestData == True:
        print("(model trained and tested on smeared data)")
    elif smearedModel == True and smearTestData == False:
        print("(model trained on smeared data but tested on exact knowledge data)")
    elif smearedModel == False and smearTestData == True:
        print("(model trained on exact knowledge data but tested on smeared data)")
    else:
        print("(model trained and tested on exact knowledge data)")

    print("Calculating model feature importances...")
    PrintFeatureImportances(list(trained_model.feature_importances_), list(df_testdata.columns))

# In above function, remove the part where I smear both train and test data and instead also return the train_array and trainObjectives (maybe instead of the pandas traindata df)
# Also, take out where I calculate the scores and importances and place this in a new function called "AnalyzeGBDT" or something like this
# Then in one of the other functions below, I can actually add in that the test_data can be smeared and so now we can calculate FOMs from true train but smeared test data (and vice-versa)
# Will need to also save the two different populations output from the predictions of the GBDT, will then use these in a more general histogram function


# Function to calculate a list of FOM values in different energy bins based on the energy binned test data
def FOMComparisonToFisher(observatory, hadronic_model, trained_gbt, df_test_data, lgEBinEdges, plotLabels, plotColors, smearing=False, smearLevels=None, printVals=False, makeHistograms=False):
    FOM_list = []
    Contamination_5per = []
    Contamination_1per = []
    events_per_lgEbin = []

    for i in range(1, len(lgEBinEdges)):

        lgEBinned_test_data = df_test_data[(np.log10(df_test_data.EnergyGeV * 1e+9) >= lgEBinEdges[i-1]) &
                                        (np.log10(df_test_data.EnergyGeV * 1e+9) <= lgEBinEdges[i])]
        lgEBinned_test_IDs = np.array(lgEBinned_test_data["ParticleID"])

        if observatory in ["Auger"] and hadronic_model in ["Sibyll 2.3e", "EPOS LHC-R", "QGSJETIII-01"]:
            lgEBinned_test_data = lgEBinned_test_data.drop(columns=["ParticleID", "EnergyGeV", "zenith", "nEM_Xmax", "nEM800m_NOTCORRECTED", "nMuHighE", "R_eMuHighE", "SigmaXmax", "SigmaR", "SigmaL"])
        else:
            raise ValueError("Unstudied choices for 'observatory' and 'hadronic_model' combination.")

        if printVals == True:
            print(f"For test data between log10(E / eV) = {lgEBinEdges[i-1]:.1f} - {lgEBinEdges[i]:.1f} ({len(lgEBinned_test_IDs)} total events)")

        fom, cont_5per, cont_1per, pop1_pred, pop2_pred = GetFOM(trained_gbt, lgEBinned_test_data.values, lgEBinned_test_IDs, plotLabels, smearValues=smearing, smearArray=smearLevels, printOutput=printVals)

        if makeHistograms == True:
            MakeHistogram(observatory, hadronic_model, pop1_pred, pop2_pred, fom, cont_5per, cont_1per, plotLabels, plotColors, lgEbins=[lgEBinEdges[i-1],lgEBinEdges[i]], smearValues=smearing, smearArray=smearLevels)

        FOM_list.append(fom)
        Contamination_5per.append(cont_5per)
        Contamination_1per.append(cont_1per)
        events_per_lgEbin.append(len(lgEBinned_test_IDs))

    return FOM_list, Contamination_5per, Contamination_1per, events_per_lgEbin


# Function that does whole analysis that take the observatory, energy bins, and primary type as input
def PerformGBTAnalysis(filename, observatory, hadronic_model, lgEbinsFOM, pFe=False, HeO=False, pHe=False, smear=False, verbose=False):
    dataframe = ReadData(filename)
    dataframe, plotLabels, plotColors = ApplyFisherDataCuts(dataframe, ProtonIron=pFe, HeliumOxygen=HeO, ProtonHelium=pHe)
    print("\n")

    dataframe_zenCut = ApplyZenithCut(dataframe, minZen=40., maxZen=60.) # Apply zenith cut same as Fisher analysis
    dataframe_ECut = BinByEnergy(dataframe_zenCut, minLgE=16.0, maxLgE=20.5) # Apply energy cut same as Fisher analysis

    gbtReg, df_train, df_test, smearLevels, _junkarray, _junkobjectives = SetupGBDT(dataframe_ECut, observatory, plotLabels, smearedGBDT=smear, analyzeModel=True, analyzeForSmearedTestData=True)

    FOMs, contam_5per, contam_1per, events_per_bin = FOMComparisonToFisher(observatory, hadronic_model, gbtReg, df_test, lgEbinsFOM, plotLabels, plotColors,
                                                                            smearing=smear, smearLevels=smearLevels, printVals=verbose, makeHistograms=True)

    return FOMs, contam_5per, contam_1per, events_per_bin


# Function to print output from GBDT analysis in a clear concise way
def PrintGBTResults(lgEbins, foms, cut_5per, cut_1per, num_events):

    lgEbins_centers = []
    for i in range(1, len(lgEbins)):
        cent = (lgEbins[i-1] + lgEbins[i]) / 2
        cent = round(cent, 1)
        lgEbins_centers.append(cent)

    eng_binned_output = [(lgE, fom, cut5, cut1, round(events, 0)) for lgE, fom, cut5, cut1, events in zip(lgEbins_centers, foms, cut_5per, cut_1per, num_events)]
    eng_binned_output = sorted(eng_binned_output, key = lambda x:x[0], reverse=False)
    print("\n")
    print("lgE Bin (Center),   FOM,                5% Contamination,   1% Contamination,   # Events")
    [print("{:<20.1f} {:<20.2f} {:<20.3f} {:<20.3f} {:<20}".format(*bin_output)) for bin_output in eng_binned_output]

#lgEBinEdges = [16.0, 16.2, 16.4, 16.6, 16.8, 17.0, 17.2, 17.4, 17.6, 17.8, 18.0, 18.2, 18.4]
lgEBinEdges = [16.0, 16.2, 16.4, 16.6, 16.8, 17.0, 17.2, 17.4, 17.6, 17.8, 18.0, 18.2, 18.4,
               18.6, 18.8, 19.0, 19.2, 19.4, 19.6, 19.8, 20.0, 20.2, 20.4]


"""# IceCube (p Fe Separation)"""

#ic_pFe_FOM_true, ic_pFe_5per_true, ic_pFe_1per_true, ic_pFe_numEvents_true = PerformGBTAnalysis("IceCube", lgEBinEdges, pFe=True, HeO=False, pHe=False, smear=False, verbose=False)
#PrintGBTResults(lgEBinEdges, ic_pFe_FOM_true, ic_pFe_5per_true, ic_pFe_1per_true, ic_pFe_numEvents_true)

#ic_pFe_FOM_smear, ic_pFe_5per_smear, ic_pFe_1per_smear, ic_pFe_numEvents_smear = PerformGBTAnalysis("IceCube", lgEBinEdges, pFe=True, HeO=False, pHe=False, smear=True, verbose=False)
#PrintGBTResults(lgEBinEdges, ic_pFe_FOM_smear, ic_pFe_5per_smear, ic_pFe_1per_smear, ic_pFe_numEvents_smear)

"""# IceCube (He O Separation)"""
'''
ic_HeO_FOM_true, ic_HeO_5per_true, ic_HeO_1per_true, ic_HeO_numEvents_true = PerformGBTAnalysis("IceCube", lgEBinEdges, pFe=False, HeO=True, pHe=False, smear=False, verbose=False)
PrintGBTResults(lgEBinEdges, ic_HeO_FOM_true, ic_HeO_5per_true, ic_HeO_1per_true, ic_HeO_numEvents_true)

ic_HeO_FOM_smear, ic_HeO_5per_smear, ic_HeO_1per_smear, ic_HeO_numEvents_smear = PerformGBTAnalysis("IceCube", lgEBinEdges, pFe=False, HeO=True, pHe=False, smear=True, verbose=False)
PrintGBTResults(lgEBinEdges, ic_HeO_FOM_smear, ic_HeO_5per_smear, ic_HeO_1per_smear, ic_HeO_numEvents_smear)

"""# IceCube (p He Separation)"""

ic_pHe_FOM_true, ic_pHe_5per_true, ic_pHe_1per_true, ic_pHe_numEvents_true = PerformGBTAnalysis("IceCube", lgEBinEdges, pFe=False, HeO=False, pHe=True, smear=False, verbose=False)
PrintGBTResults(lgEBinEdges, ic_pHe_FOM_true, ic_pHe_5per_true, ic_pHe_1per_true, ic_pHe_numEvents_true)

ic_pHe_FOM_smear, ic_pHe_5per_smear, ic_pHe_1per_smear, ic_pHe_numEvents_smear = PerformGBTAnalysis("IceCube", lgEBinEdges, pFe=False, HeO=False, pHe=True, smear=True, verbose=False)
PrintGBTResults(lgEBinEdges, ic_pHe_FOM_smear, ic_pHe_5per_smear, ic_pHe_1per_smear, ic_pHe_numEvents_smear)

"""# Auger (p Fe Separation)"""

auger_pFe_FOM_true, auger_pFe_5per_true, auger_pFe_1per_true, auger_pFe_numEvents_true = PerformGBTAnalysis("Auger", lgEBinEdges, pFe=True, HeO=False, pHe=False, smear=False, verbose=False)
PrintGBTResults(lgEBinEdges, auger_pFe_FOM_true, auger_pFe_5per_true, auger_pFe_1per_true, auger_pFe_numEvents_true)

auger_pFe_FOM_smear, auger_pFe_5per_smear, auger_pFe_1per_smear, auger_pFe_numEvents_smear = PerformGBTAnalysis("Auger", lgEBinEdges, pFe=True, HeO=False, pHe=False, smear=True, verbose=False)
PrintGBTResults(lgEBinEdges, auger_pFe_FOM_smear, auger_pFe_5per_smear, auger_pFe_1per_smear, auger_pFe_numEvents_smear)

"""# Auger (He O Separation)"""

auger_HeO_FOM_true, auger_HeO_5per_true, auger_HeO_1per_true, auger_HeO_numEvents_true = PerformGBTAnalysis("Auger", lgEBinEdges, pFe=False, HeO=True, pHe=False, smear=False, verbose=False)
PrintGBTResults(lgEBinEdges, auger_HeO_FOM_true, auger_HeO_5per_true, auger_HeO_1per_true, auger_HeO_numEvents_true)

auger_HeO_FOM_smear, auger_HeO_5per_smear, auger_HeO_1per_smear, auger_HeO_numEvents_smear = PerformGBTAnalysis("Auger", lgEBinEdges, pFe=False, HeO=True, pHe=False, smear=True, verbose=False)
PrintGBTResults(lgEBinEdges, auger_HeO_FOM_smear, auger_HeO_5per_smear, auger_HeO_1per_smear, auger_HeO_numEvents_smear)

"""# Auger (p He Separation)"""

auger_pHe_FOM_true, auger_pHe_5per_true, auger_pHe_1per_true, auger_pHe_numEvents_true = PerformGBTAnalysis("Auger", lgEBinEdges, pFe=False, HeO=False, pHe=True, smear=False, verbose=False)
PrintGBTResults(lgEBinEdges, auger_pHe_FOM_true, auger_pHe_5per_true, auger_pHe_1per_true, auger_pHe_numEvents_true)

auger_pHe_FOM_smear, auger_pHe_5per_smear, auger_pHe_1per_smear, auger_pHe_numEvents_smear = PerformGBTAnalysis("Auger", lgEBinEdges, pFe=False, HeO=False, pHe=True, smear=True, verbose=False)
PrintGBTResults(lgEBinEdges, auger_pHe_FOM_smear, auger_pHe_5per_smear, auger_pHe_1per_smear, auger_pHe_numEvents_smear)

'''