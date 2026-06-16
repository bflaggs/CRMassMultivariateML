# Use this module for the GBDT analysis for the PRD paper/thesis/ISVHECRI proceeding

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

import os

rc("font", size=18.0)
rcParams["font.family"] = "serif"
rcParams["mathtext.fontset"] = "dejavuserif"

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor

# Add computation of AUC curves, maybe use this??
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import roc_curve


class GBDTAnalysis(object):

    def __init__(self, save_plots=False, analyze_smeared_test_data=False, head_directory=None):
        
        self.save_plots = save_plots
        self.analyze_smeared_test_data = analyze_smeared_test_data
        self.head_directory = head_directory

        if head_directory == None:
            self.head_directory = str(os.path.dirname(os.path.realpath(__file__)))


    # Function to read in the data and do some renaming of columns
    def read_data(self, filename):
        # Check file exists
        if os.path.exists(filename):
            df = pd.read_csv(filename, sep=" ")
        else:
            raise ValueError(f"File {filename} does not exist. Check path.")

        # Rename some columns to make things easier for future...
        df.columns = ["ParticleID", "EnergyGeV", "zenith", "nEM_Xmax", "nEM_Obslev", "nEM800m_NOTCORRECTED", "nMuHighE", "R_eMuHighE", "nMu800m", "R_eMu800m", "Xmax", "SigmaXmax", "R", "SigmaR", "L", "SigmaL"]

        return df


    # Function to apply same data cuts as the Fisher analysis to the data
    def apply_fisher_data_cuts(self, dataset, proton_iron=False, helium_oxygen=False, proton_helium=False):
        if proton_iron + helium_oxygen + proton_helium > 1:
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

        if proton_iron == True:
            # Cut He and O
            dfCuts = dfCuts[(dfCuts.ParticleID != 402) &
                            (dfCuts.ParticleID != 1608)]
            # Convert p and Fe ParticlsIDs to binaries
            dfCuts.loc[dfCuts.ParticleID == 5626, "ParticleID"] = 0
            dfCuts.loc[dfCuts.ParticleID == 14, "ParticleID"] = 1
            plotLabels = ['Proton', 'Iron']
            plotColors = ['#CC6677', '#4477AA']

        elif helium_oxygen == True:
            # Cut p and Fe
            dfCuts = dfCuts[(dfCuts.ParticleID != 14) &
                            (dfCuts.ParticleID != 5626)]
            # Convert He and O ParticlsIDs to binaries
            dfCuts.loc[dfCuts.ParticleID == 1608, "ParticleID"] = 0
            dfCuts.loc[dfCuts.ParticleID == 402, "ParticleID"] = 1
            plotLabels = ['Helium', 'Oxygen']
            plotColors = ['#DDCC77', '#117733']

        elif proton_helium == True:
            # Cut O and Fe
            dfCuts = dfCuts[(dfCuts.ParticleID != 1608) &
                            (dfCuts.ParticleID != 5626)]
            # Convert p and He ParticlsIDs to binaries
            dfCuts.loc[dfCuts.ParticleID == 402, "ParticleID"] = 0
            dfCuts.loc[dfCuts.ParticleID == 14, "ParticleID"] = 1
            plotLabels = ['Proton', 'Helium']
            plotColors = ['#CC6677', '#DDCC77']

        else:
            print("No keyword for separating two primary types. Instead will try separating protons from heavier primaries!")
            # Convert p and heavier ParticleIDs to binaries
            dfCuts.loc[dfCuts.ParticleID != 14, "ParticleID"] = 0 # Must do this first!
            dfCuts.loc[dfCuts.ParticleID == 14, "ParticleID"] = 1
            plotLabels = ['Proton', 'Heavier']
            plotColors = ['#CC6677', '#4477AA']

        return dfCuts, plotLabels, plotColors


    # Function to apply zenith angle cuts
    def apply_zenith_cut(self, dataset, min_zen=40.0, max_zen=60.0):
        dfZenCuts = dataset[((dataset.zenith * 180. / np.pi) >= min_zen) &
                            ((dataset.zenith * 180. / np.pi) <= max_zen)]
        return dfZenCuts


    # Function to apply energy bin cuts, used to split data in energy so that multiple models can be trained
    def bin_by_energy(self, dataset, minLgE=16.0, maxLgE=20.5):
        dfEnergyBinned = dataset[(np.log10(dataset.EnergyGeV * 1e+9) >= minLgE) &
                                (np.log10(dataset.EnergyGeV * 1e+9) <= maxLgE)]
        return dfEnergyBinned


    # Function to drop columns not used in Fisher analysis and separate array of particle labels (goal) from data
    def get_data_for_gbdt(self, dataset, dropZenith=False):
        if dropZenith == True:
            dfGBDT = dataset.drop(columns=["EnergyGeV", "zenith", "nEM_Xmax", "nEM800m_NOTCORRECTED", "nMuHighE", "R_eMuHighE", "SigmaXmax", "SigmaR", "SigmaL"])
            smearLevels = [0.1, 0.1, 0.14, 20.0, 0.05, 5.0]
        else:
            dfGBDT = dataset.drop(columns=["EnergyGeV", "nEM_Xmax", "nEM800m_NOTCORRECTED", "nMuHighE", "R_eMuHighE", "SigmaXmax", "SigmaR", "SigmaL"])
            smearLevels = [2.0 * np.pi / 180.0, 0.1, 0.1, 0.14, 20.0, 0.05, 5.0]

        #print("WARNING: To keep the high-energy muon number this function should be updated and the observable studied in more detail.")

        # Define another array which contains the labels of the particles
        particleIDs = np.array(dfGBDT["ParticleID"])

        # Remove the IDs for the particles from the dataset, so that model can be trained
        dfGBDT.drop(columns=["ParticleID"], inplace=True)

        return dfGBDT, particleIDs, smearLevels


    # Function to get the feature importances and print them nicely to the screen
    def print_feature_importances(self, importance_list, feature_list, save_importances=False):
        if type(feature_list) != list:
            feature_list = list(feature_list)

        ft_importances = [(feature, round(importance, 3)) for feature, importance in zip(feature_list, importance_list)]
        ft_importances = sorted(ft_importances, key = lambda x:x[1], reverse=True)
        [print("Observable: {:20} Importance: {}".format(*pair)) for pair in ft_importances]

        if save_importances:
            file = open(f"{self.head_directory}/model_output/feature_importances.txt", "w")
            for pair in ft_importances:
                file.write("Observable: {:20} Importance: {}\n".format(*pair))
            file.close()


    # Function to smear data (used to simulate detector response, i.e. true reconstructions)
    def smear_data(self, original_array, smear_values):
        if len(smear_values) != original_array.shape[1]:
            raise ValueError("Shape of smearing array values and data to smear is not the same!")

        smeared_array = np.copy(original_array)

        for i in range(original_array.shape[0]):
            for j in range(original_array.shape[1]):
                smeared_array[i][j] += stats.norm.rvs(loc=0.0, scale=smear_values[j])

        return smeared_array


    # Function to extract FOM value
    def get_fom(self, GBDT, testData, testIDs, plotLabels, smearValues=False, smearArray=None, printOutput=False):
        if smearValues == True:
            testData_smeared = self.smear_data(testData, smearArray)
            gbdt_pred_test = GBDT.predict(testData_smeared)
        else:
            gbdt_pred_test = GBDT.predict(testData)

        pop1_gbdt_pred = gbdt_pred_test[np.nonzero(testIDs == 1)]
        pop2_gbdt_pred = gbdt_pred_test[np.nonzero(testIDs == 0)]

        pop2_thresh5 = np.quantile(pop2_gbdt_pred, 0.95)
        pop1_5perContamination = pop1_gbdt_pred[pop1_gbdt_pred >= pop2_thresh5]

        pop2_thresh1 = np.quantile(pop2_gbdt_pred, 0.99)
        pop1_1perContamination = pop1_gbdt_pred[pop1_gbdt_pred >= pop2_thresh1]

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


    # Function to make the output histogram
    def make_histogram(self, observatory, hadronic_model, pop1_pred, pop2_pred, fom, cont_5per, cont_1per, plotLabels, plotColors, lgEbins=[16.0,20.5], smearValues=False, smearArray=None):

        plt.figure(figsize=(18.0 / 2.54, 15.0 / 2.54))
        nbins = int(np.sqrt(max(len(pop1_pred), len(pop2_pred))))
        n2, bins2, patches2 = plt.hist(pop2_pred, bins=nbins, histtype="step", linewidth=2, color=plotColors[1], label=plotLabels[1])
        n1, bins1, patches1 = plt.hist(pop1_pred, bins=nbins, histtype="step", linewidth=2, color=plotColors[0], label=plotLabels[0])

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

        plt.vlines(pop2_5perOverlap[-1], 0, (max_counts / 1.5), colors="black", linestyle="dashed", linewidth=2, label="5% Contamination")
        plt.vlines(pop2_1perOverlap[-1], 0, (max_counts / 1.5), colors="black", linestyle="dotted", linewidth=2, label="1% Contamination")
        plt.ylim(0, max_counts * 1.3)
        plt.text(0.2, (max_counts * 1.0), f"{hadronic_model} (FOM = {fom:.2f})", fontsize=14)
        plt.text(0.2, (max_counts * 0.92), rf"$\lg$(E / eV) = {lgEbins[0]:.1f}-{lgEbins[1]:.1f}, {observatory}", fontsize=14) # Add zenith range?
        plt.text(0.2, (max_counts * 0.85), f"Frac. Sep. {sep_pop_string} = {cont_5per:.3f} @ 5% cont.", fontsize=12)
        plt.text(0.2, (max_counts * 0.80), f"Frac. Sep. {sep_pop_string} = {cont_1per:.3f} @ 1% cont.", fontsize=12)
        plt.legend(loc="best", fontsize=14, ncol=2)
        if smearValues == True:
            plt.xlabel("GBT Regressor Output (Smeared)")
            file_ending = "_Smeared.pdf"
        else:
            plt.xlabel("GBT Regressor Output (True)")
            file_ending = ".pdf"
        plt.ylabel("Counts")

        if hadronic_model == "EPOS LHC-R":
            had_mod = "EPOSLHCR"
        elif hadronic_model == "Sibyll 2.3e":
            had_mod = "Sibyll23e"
        elif hadronic_model == "QGSJETIII-01":
            had_mod = "QGSJETIII01"

        figure_name = f"{self.head_directory}/plots/GBDT_{observatory}_{had_mod}_{plotLabels[0]}{plotLabels[1]}_lgE_{lgEbins[0]:.1f}_{lgEbins[1]:.1f}" + file_ending
        plt.savefig(figure_name, bbox_inches="tight")


    # Setup another function which will be used to initialize the GBDT w/ a keyword option for setting up a smeared GBDT or not.
    def setup_gbdt(self, og_df, primaries, smearedGBDT=False, analyzeModel=True, analyzeForSmearedTestData=False):

        print(f"Model trained to separate {primaries[0]} from {primaries[1]}.")

        # Define another array which contains the labels of the particles
        objectives = np.array(og_df["ParticleID"])

        # First do the train-test split of the data
        trainData, testData, trainObjectives, testObjectives = train_test_split(og_df.values, objectives, test_size=0.2, shuffle=True, random_state=42)

        # Now need to convert arrays to DataFrames and drop non-used things in analysis
        df_train = pd.DataFrame(trainData, columns=list(og_df.columns))
        df_test = pd.DataFrame(testData, columns=list(og_df.columns))

        df_train, train_obj, smearLevels = self.get_data_for_gbdt(df_train, dropZenith=True)
        df_test, test_obj, smearLevels = self.get_data_for_gbdt(df_test, dropZenith=True)

        if (trainObjectives != train_obj).any() or (testObjectives != test_obj).any():
            raise ValueError("Inconsistent objectives for train/test sets in train-test splitting and data cuts phases.")

        train_array = df_train.to_numpy()
        test_array = df_test.to_numpy()

        if smearedGBDT == True:
            print("Training the GBDT on smeared data to simulate detector resolutions...")
            train_array = self.smear_data(train_array, smearLevels)

        gbt_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)
        print("Training the model...")
        gbt_model.fit(train_array, trainObjectives)

        if analyzeModel == True:
            self.analyze_gbdt(gbt_model, train_array, trainObjectives, pd.DataFrame(testData, columns=list(og_df.columns)), smearedModel=smearedGBDT, smearTestData=analyzeForSmearedTestData)

        if smearedGBDT == True:
            print("\n")
            print("WARNING: Returned train and test data frames will not be smeared! (But returned training array will be.)")

        return gbt_model, pd.DataFrame(trainData, columns=list(og_df.columns)), pd.DataFrame(testData, columns=list(og_df.columns)), smearLevels, train_array, trainObjectives


    # Function to analyze the GBDT, i.e. output model scores and list feature importances
    def analyze_gbdt(self, trained_model, trainingData, trainingObjectives, original_testData_df, smearedModel=False, smearTestData=False):

        df_testdata, testingObjectives, smearLevels = self.get_data_for_gbdt(original_testData_df, dropZenith=True)
        testingData = df_testdata.to_numpy()

        if smearTestData == True:
            testingData = self.smear_data(testingData, smearLevels)

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
        self.print_feature_importances(list(trained_model.feature_importances_), list(df_testdata.columns), save_importances=True)


    # Function to calculate a list of FOM values in different energy bins based on the energy binned test data
    def fom_comparison_to_fisher(self, observatory, hadronic_model, trained_gbt, df_test_data, lgEBinEdges, plotLabels, plotColors, smearing=False, smearLevels=None, printVals=False, makeHistograms=False):
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

            fom, cont_5per, cont_1per, pop1_pred, pop2_pred = self.get_fom(trained_gbt, lgEBinned_test_data.values, lgEBinned_test_IDs, plotLabels, smearValues=smearing, smearArray=smearLevels, printOutput=printVals)

            if makeHistograms == True:
                self.make_histogram(observatory, hadronic_model, pop1_pred, pop2_pred, fom, cont_5per, cont_1per, plotLabels, plotColors, lgEbins=[lgEBinEdges[i-1],lgEBinEdges[i]], smearValues=smearing, smearArray=smearLevels)

            FOM_list.append(fom)
            Contamination_5per.append(cont_5per)
            Contamination_1per.append(cont_1per)
            events_per_lgEbin.append(len(lgEBinned_test_IDs))

        return FOM_list, Contamination_5per, Contamination_1per, events_per_lgEbin


    # Function that does whole analysis that take the observatory, energy bins, and primary type as input
    def perform_gbdt_analysis(self, filename, observatory, hadronic_model, lgEbinsFOM, zenith_bins=(40.0, 60.0), pFe=False, HeO=False, pHe=False, smear=False, verbose=False):
        if len(zenith_bins) != 2:
            raise ValueError("Zenith bins must be defined with form like so: zenith_bins=(min_deg, max_deg)")

        print("\n")
        print(f"Setting up GBDT analysis for location {observatory} with hadronic model {hadronic_model}.")
        
        dataframe = self.read_data(filename)
        dataframe, plotLabels, plotColors = self.apply_fisher_data_cuts(dataframe, proton_iron=pFe, helium_oxygen=HeO, proton_helium=pHe)

        dataframe_zenCut = self.apply_zenith_cut(dataframe, min_zen=zenith_bins[0], max_zen=zenith_bins[1]) # Apply zenith cut same as Fisher analysis
        dataframe_ECut = self.bin_by_energy(dataframe_zenCut, minLgE=16.0, maxLgE=20.5) # Apply energy cut same as Fisher analysis

        gbtReg, df_train, df_test, smearLevels, _junkarray, _junkobjectives = self.setup_gbdt(dataframe_ECut, plotLabels, smearedGBDT=smear, analyzeModel=True, analyzeForSmearedTestData=self.analyze_smeared_test_data)

        FOMs, contam_5per, contam_1per, events_per_bin = self.fom_comparison_to_fisher(observatory, hadronic_model, gbtReg, df_test, lgEbinsFOM, plotLabels, plotColors,
                                                                                smearing=smear, smearLevels=smearLevels, printVals=verbose, makeHistograms=self.save_plots)

        return FOMs, contam_5per, contam_1per, events_per_bin


    # Function to print output from GBDT analysis in a clear concise way
    def print_gbdt_results(self, lgEbins, foms, cut_5per, cut_1per, num_events):

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


    def save_output(self, output_file, lg_e_bin_edges, foms, cut_5per, cut_1per, num_events):
        if type(lg_e_bin_edges) == list:
            lg_e_bin_edges = np.array(lg_e_bin_edges)

        bin_centers = (lg_e_bin_edges[:-1] + lg_e_bin_edges[1:]) / 2

        outfile = open(output_file, "w")
        outfile.write("#lgE_bin_center FOM contamination_5percent contamination_1percent num_events\n")

        for i in range(len(bin_centers)):
            outfile.write(f"{bin_centers[i]:.1f} {foms[i]} {cut_5per[i]} {cut_1per[i]} {num_events[i]}\n")

        outfile.close()
