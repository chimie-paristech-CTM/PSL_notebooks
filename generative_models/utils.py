import numpy as np 
import pandas as pd
import random 
from random import seed
from sklearn.preprocessing import StandardScaler
from itertools import *
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem import rdMolDescriptors
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, PredictionErrorDisplay
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict



# Visualization
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})


#   DATA PRE-PROCESSING

def canonical_smiles(df, smiles):

    """
    Checks if the column exists, and then canonicalizes the SMILES of the column.

    Args:
        df (pandas dataframe): dataframe
        smiles (str): the name of the column containing the SMILES to be canonicalized

    returns:
        df : the dataframe with canonicalized SMILES
    """

    if smiles not in df.columns:
        print(f"Column {smiles} not found in DataFrame.")
        return

    df[smiles] = df[smiles].apply(lambda x: Chem.MolFromSmiles(x))
    df[smiles] = df[smiles].apply(lambda x: Chem.MolToSmiles(x))

    return df


def get_ligands_dict(row):

    """
    Counts occurrences of strings in the L1, L2 and L3 columns in a row and creates a dictionary
    
    Args:
        row: a row of a dataframe
    
    returns:
        string_count (dict): a dictionnary containing the SMILES of the L1, L2 and L3 columns with their count of appearance
    """

    string_count = {}
    for col in ['L1', 'L2', 'L3']:
        string = row[col]
        if string in string_count:
            string_count[string] += 1
        else:
            string_count[string] = 1
    return string_count

chosen_descriptors = ['BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 
                      'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 
                      'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'ExactMolWt', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 
                      'FractionCSP3', 'HallKierAlpha', 'HeavyAtomCount', 'HeavyAtomMolWt', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 
                      'MaxAbsEStateIndex', 'MaxAbsPartialCharge', 'MaxEStateIndex', 'MaxPartialCharge', 'MinAbsEStateIndex', 'MinAbsPartialCharge', 
                      'MinEStateIndex', 'MinPartialCharge', 'MolLogP', 'MolMR', 'MolWt', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 
                      'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 
                      'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRadicalElectrons', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 
                      'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumValenceElectrons', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 
                      'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 
                      'RingCount', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 
                      'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 
                      'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 
                      'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 
                      'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 
                      'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 
                      'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 
                      'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 
                      'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 
                      'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 
                      'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 
                      'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 
                      'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 
                      'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea', 'qed']


def get_rdkit_descriptors(mol):
    """
    Gets the RDKit molecular descriptors of the ligands

    Args:
        mol (rdkit mol): the ligands RDKit mol objects

    Returns:
        np.array: the molecular descriptors array
    """
    mol_descriptor_calculator = MolecularDescriptorCalculator(chosen_descriptors)

    return np.array(mol_descriptor_calculator.CalcDescriptors(mol))

def concatenate_float_lists(row):

    """
    Concatenates 3 rows.

    Args:
        row: a dataframe row
    
    Returns:
        np.concatenate() (numpy array) : a numpy array, the concatenation of the 3 arrays for this row
    """

    desc1 = row['Desc1']
    desc2 = row['Desc2']
    desc3 = row['Desc3']
    return np.concatenate((desc1, desc2, desc3))


def calc_desc(df):

    """
    
    Adds and fills the RDKit Descriptors column in a dataframe of compounds.

    Args:
        df: the dataframe of compounds one wants to compute the descriptors of.

    Returns:
        df: the dataframe with a new 'Descriptors' column.

    """

    df['Desc1'] = df['MOL1'].apply(lambda x: 
            get_rdkit_descriptors(x))
    df['Desc2'] = df['MOL2'].apply(lambda x: 
            get_rdkit_descriptors(x))
    df['Desc3'] = df['MOL3'].apply(lambda x: 
            get_rdkit_descriptors(x))

    #Getting the final descriptor of the complex
    df['Descriptors'] = df.apply(concatenate_float_lists, axis=1)

    return df

def get_morgan_fp(mol, n, bits):

    """
    This function calculates the Morgan fingerprint of a mol object

    Args : 
        mol (RDKit mol): a RDKit mol object
        n (int): the radius value of the Morgan fingerprint
        bits (int): the nBits value of the Morgan fingerprint

    Returns:
        np.array : the Morgan fingerprint
    """

    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, n, nBits=bits))
    
def get_rdkit_fp(mol, nbits=2048):
    
    """
    This function calculates the RDKit fingerprint of a mol object

    Args:
        mol (RDKit mol): a RDKit mol object
        nbits (int): the size of the fingerprint

    Returns:
        np.array: the RDKit fingerprint
    """

    return np.array(AllChem.RDKFingerprint(mol, fpSize=nbits))

def get_morgan_fp_from_smiles(smiles, n, bits):

    """
    Gets Morgan fingerprints of the ligands from SMILES string

    Args:
        smiles (str): the ligands SMILES
        rad (int): the radii of the fingerprints
        nbits (int): the number of bits for the fingerprints

    Returns:
        np.array: the fingerprint
    """
    return np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius=n, nBits=bits))
    
def get_rdkit_fp_from_smiles(smiles, bits):
    
   """
    Get RDKit fingerprint of the ligands from SMILES

    Args:
        smiles (str): the ligands SMILES
        bits (int): the number of bits for the fingerprints

    Returns:
        np.array: the fingerprint
        """
   
   return np.array(AllChem.RDKFingerprint(Chem.MolFromSmiles(smiles, nBits=bits)))


def convert_to_float(value):

    """
    This function converts a string to float. It handles strings with > or < , common in biological IC50 values. 
    
    Args:
        value (str or float): a string or a float
    
    returns:
        value (float): the float version of the string, or just the value if it was already a float
    """

    if isinstance(value, float):
        return value
    elif isinstance(value, str):
        if value.startswith('>'):
            value_cleaned = value[1:]
        elif value.startswith('<'):
            value_cleaned = value[1:]
        else:
            value_cleaned = value
        try: 
            return float(value_cleaned)
        except (ValueError, TypeError):
            print(value_cleaned)
            return None


def drop_duplicates(df, column, print_length=True):

    """
    InPlace function, modifies the dataframe when called. If there is duplicates in the column 'column', 
    it only keeps the first occurence row and delete the others. 

    Args:
        df: the dataframe 
        column (str): the name of the column where duplicates have to be dropped
    """

    df[column] = df[column].apply(tuple)
    df.drop_duplicates(subset=[column], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df[column] = df[column].apply(list)

    if print_length: 
        print(len(df))


def average_duplicates(df, column_todrop, column_values):

    """
    This function takes a dataframe as input. If there is duplicates in the 'column_todrop' column with various values
    in the 'column_values' column, it allows to keep only one row with the 'column_todrop' item, and the corresponding
    value in the 'column_values' column is an average of all previous the duplicates of the 'column_todrop'.

    Args:
        df: the dataframe 
        column_to_drop (str): the name of the column where duplicates have to be dropped
        column_values (str): the name of the column containing the values we want to keep an average of
        
    returns:
        df: the filtered dataframe
    """

    df[column_todrop+'_dup'] = df[column_todrop]
    df[column_todrop+'_dup'] = df[column_todrop+'_dup'].apply(tuple)
    df[column_values] = df.groupby(column_todrop+'_dup')[column_values].transform('mean')
    df = df.groupby(column_todrop+'_dup', as_index=False).first()
    df.drop(columns=[column_todrop+'_dup'])
    df.reset_index(inplace=True)
    print(f'Length of training dataset after cleaning duplicates, before adding permutations : {len(df)}')
    return df


def prepare_df_morgan(df_original, r, bits):

    """
    This function takes a dataframe (formated like our original dataset of ruthenium complexes).
    It performs all necessary processing to make it usable in the code, including :
                 - canonicalising the SMILES
                 - creating the MOL columns with the RDKit mol representations
                 - creating a dictionnary and a set of the ligands SMILES (necessary for later functions)
                 - creating a set of the RDKit mol representations
                 - standardize the complexes with 2 identical ligands to AAB representation
                 - calculates the Morgan fingerprints and adding them in a Fingerprint column 
                 - renames the columns
                 - associates an ID to each complex (necessary for indicing functions)
                 - calculates the pIC50 drom the IC50 column

    Args:
        df: the dataframe 
        r (int): the radius of the Morgan fingerprint
        bits (int): the nBits value of the Morgan fingerprint
    
    returns:
        df: the processed dataframe with Morgan Fingerprints 
    """

    df = df_original.copy()

    df.dropna(subset=['L1', 'L2', 'L3'], how='any', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = canonical_smiles(df, 'L1')
    df = canonical_smiles(df, 'L2')
    df = canonical_smiles(df, 'L3')

    df['MOL1'] = df.L1.apply(Chem.MolFromSmiles)
    df['MOL2'] = df.L2.apply(Chem.MolFromSmiles)
    df['MOL3'] = df.L3.apply(Chem.MolFromSmiles)
    df['Ligands_Dict'] = df.apply(get_ligands_dict, axis=1)
    df['Ligands_Set'] = df.apply(lambda row: set([row['L1'], row['L2'], row['L3']]), axis=1)
    df['Mols_Set'] = df.apply(lambda row: set([row['MOL1'], row['MOL2'], row['MOL3']]), axis=1)

    #AAB standardization
    df = swap_identical_ligands(df)

    df['ECFP4_1'] = df.MOL1.apply(lambda mol: get_morgan_fp(mol, r, bits))
    df['ECFP4_2'] = df.MOL2.apply(lambda mol: get_morgan_fp(mol, r, bits))
    df['ECFP4_3'] = df.MOL3.apply(lambda mol: get_morgan_fp(mol, r, bits))

    df.rename(columns={'IC50 (μM)': 'IC50', 'Incubation Time (hours)': 'IncubationTime', 'Partition Coef logP': 'logP', 'Cell Lines ': 'Cells'}, inplace=True)
    add_lists = lambda row: [sum(x) for x in zip(row['ECFP4_1'], row['ECFP4_2'], row['ECFP4_3'])]
    df['Fingerprint'] = df.apply(add_lists, axis=1)

    df['ID']=[i for i in range(len(df))]

    df['IC50'] = df['IC50'].apply(convert_to_float)
    df['pIC50'] = df['IC50'].apply(lambda x: - np.log10(x * 10 ** (-6)))

    return df 

def prepare_df_rdkit(df_original, nbits=2048):

    """
    Takes a dataframe (formated like our original dataset of ruthenium complexes).
    It performs all necessary processing to make it usable in the code, including :
                 - canonicalising the SMILES
                 - creating the MOL columns with the rdkit mol representations
                 - creating a dictionnary and a set of the ligands SMILES (necessary for later functions)
                 - creating a set of the RDKit mol representations
                 - standardize the complexes with 2 identical ligands to AAB representation
                 - calculates the RDKit fingerprints and adding them in a Fingerprint column 
                 - renames the columns
                 - associates an ID to each complex (necessary for indicing functions)
                 - calculates the pIC50 drom the IC50 column

    Args:
        df: the dataframe 
        nbits (int): the nBits value of the RDKit fingerprint
    
    returns: 
        df: the processed dataframe with RDKit Fingerprints 
    """

    df = df_original.copy()

    df.dropna(subset=['L1', 'L2', 'L3'], how='any', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = canonical_smiles(df, 'L1')
    df = canonical_smiles(df, 'L2')
    df = canonical_smiles(df, 'L3')

    df['MOL1'] = df.L1.apply(Chem.MolFromSmiles)
    df['MOL2'] = df.L2.apply(Chem.MolFromSmiles)
    df['MOL3'] = df.L3.apply(Chem.MolFromSmiles)
    df['Ligands_Dict'] = df.apply(get_ligands_dict, axis=1)
    df['Ligands_Set'] = df.apply(lambda row: set([row['L1'], row['L2'], row['L3']]), axis=1)
    df['Mols_Set'] = df.apply(lambda row: set([row['MOL1'], row['MOL2'], row['MOL3']]), axis=1)

    #AAB standardization
    df = swap_identical_ligands(df)

    df['RDKIT_1'] = df.MOL1.apply(lambda mol: get_rdkit_fp(mol, nbits))
    df['RDKIT_2'] = df.MOL2.apply(lambda mol: get_rdkit_fp(mol, nbits))
    df['RDKIT_3'] = df.MOL3.apply(lambda mol: get_rdkit_fp(mol, nbits))

    df.rename(columns={'IC50 (μM)': 'IC50', 'Incubation Time (hours)': 'IncubationTime', 'Partition Coef logP': 'logP', 'Cell Lines ': 'Cells'}, inplace=True)
    add_lists = lambda row: [sum(x) for x in zip(row['RDKIT_1'], row['RDKIT_2'], row['RDKIT_3'])]
    df['Fingerprint'] = df.apply(add_lists, axis=1)

    df['ID']=[i for i in range(len(df))]

    df['IC50'] = df['IC50'].apply(convert_to_float)
    df['pIC50'] = df['IC50'].apply(lambda x: - np.log10(x * 10 ** (-6)))

    return df 


def prepare_input(df):
    """
    Cleaning and preprocessing the data.

    Args:
        df (pandas dataframe)

    Returns:
        df: the dataframe with no duplicates and canonicalized SMILES.

    """

    #Dropping lines with missing smiles
    df.dropna(subset=['L1', 'L2', 'L3'], how='any', inplace=True)
    df.reset_index(drop=True, inplace=True)

    #Making smiles canonical - this is only for data analysis purposes
    canonical_smiles(df, 'L1')
    canonical_smiles(df, 'L2')
    canonical_smiles(df, 'L3')

    df['ID'] = df.index

    return df



#   CROSS VALIDATION AND RESULTS 

#Determine MAE and RMSE metrics from two arrays containing the labels and the corresponding predicted values
def obtain_metrics(y_data, y_predictions):

    """
    Takes the real and predicted values and returns the metrics associated, to get an evaluation
    of the model. The metrics chosen here are MAE, RMSE, Ratio and R² Score.

    Args:
        y_data (array): an array containging the real target values of the dataset 
        y_predictions (array): an array containging the predicted (by the model) target values of the dataset 
    
    returns:
        str: a little text displaying the metrics values
    """
    
    mae = mean_absolute_error(y_data, y_predictions)
    mse = mean_squared_error(y_data, y_predictions)
    rmse = np.sqrt(mse)
    ratio = rmse/mae
    r2 = r2_score(y_data, y_predictions)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'Ratio': ratio,
        'R² Score': r2
    }


# Set global font size for tick labels
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16



#Plot the scatter plot as a Figure
def plot_cv_results(y_data, y_predictions, log=False): 
    """
    This function plots the correlation between the real and predicted values of either the target data (IC50), or the
    pvalue of the target data (pIC50).
    It also prints the residual errors in a second plot frame. 

    Args:
        y_data (array): an array containging the real target values of the dataset 
        y_predictions (array): an array containging the predicted (by the model) target values of the dataset 
        log (bool): set to True if we want the pIC50 instead of the IC50
    
    returns:
        two plots
    """
    
    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))

    PredictionErrorDisplay.from_predictions(
        y_true=y_predictions,
        y_pred=y_data,
        # We exchange data and predictions to get the true values as the abscissa and predictions as ordinate
        kind="actual_vs_predicted",
        scatter_kwargs={"alpha": 0.5},
        ax=axs[0],
    )
    axs[0].axis("square")
    if log:
        axs[0].set_xlabel("True pIC50", fontsize=16)
        axs[0].set_ylabel("Predicted pIC50", fontsize=16)
    else:
        axs[0].set_xlabel("True IC50 (μM)", fontsize=16)
        axs[0].set_ylabel("Predicted IC50 (μM)", fontsize=16)
    
    max_value = int(np.max(np.concatenate((y_data, y_predictions)))) 
    min_value = max(0, int(np.min(np.concatenate((y_data, y_predictions)))))
    x_ticks = [i for i in range(min_value, max_value + 1, 100)] + [max_value]
    y_ticks = [i for i in range(min_value, max_value + 1, 100)] + [max_value]
    axs[0].set_xticks(x_ticks)
    axs[0].set_yticks(y_ticks)
    axs[0].tick_params(axis='both', which='major', labelsize=16)

    PredictionErrorDisplay.from_predictions(
        y_true=y_predictions,
        y_pred=y_data,
        # Same here (cf previous comment)
        kind="residual_vs_predicted",
        scatter_kwargs={"alpha": 0.5},
        ax=axs[1],
    )
    axs[1].axis("square")
    
    if log:
        axs[1].set_xlabel("True pIC50", fontsize=16)
        axs[1].set_ylabel("Residuals (predicted pIC50 - true pIC50)", fontsize=16)
    else:
        axs[1].set_xlabel("True IC50 (μM)", fontsize=16)
        axs[1].set_ylabel("Residuals (predicted IC50 - true IC50) (μM)", fontsize=16)
    
    axs[1].tick_params(axis='both', which='major', labelsize=16)

    plt.subplots_adjust(wspace=0.4)  # Adjust horizontal space between plots
    
    _ = fig.suptitle(
        "Regression displaying correlation between true and predicted data", y=0.9
    )
    #plt.savefig("./figures/figure.png", dpi=300, transparent=True)
    plt.show()


def has_two_identical_ligands(l_dict):

    """
    Takes a dictionnary of strings. If one of the strings as an occurence of 2, it return True. 
    
    Args:
        l_dict : dictionnary
    
    returns:
        result(bool)
    """

    result = []
    for d in l_dict:
        if any(count == 2 for count in d.values()):
            result.append(True)
        else:
            result.append(False)
    return result


def swap_identical_ligands(df):
    
    """
    Converts the metals complexes with 2 identical ligands to the AAB structure. 
    
    Args:
        df : dataframe containing at least a column 'Ligands_Dict' of dicts, and 3 columns L1, L2 and L3 of strings
    
    returns:
       df: updated dataframe
    """

    dict_list = df['Ligands_Dict']
    results = has_two_identical_ligands(dict_list)
    for i, result in enumerate(results):
        if result:
            ligand_dict = dict_list[i]
            twice_ligand = None
            once_ligand = None
            for ligand, count in ligand_dict.items():
                if count == 2:
                    twice_ligand = ligand
                elif count == 1:
                    once_ligand = ligand
            df.at[i, 'L1'] = twice_ligand
            df.at[i, 'L2'] = twice_ligand
            df.at[i, 'L3'] = once_ligand
    return df


# Data Preparing 

def ligands_permutation(df):

    """
    This function allows to include every permutation of ligands for each unique complex. For example, if the complex
    ABC is in the dataframe, it will create new lines ACB, BAC, BCA, CBA, CAB (with all the other column values same
    as the ABC original row).
    
    Args:
        df : pd.dataframe 
    
    returns:
        expanded_df: expanded dataframe with all the new permutations
    """

    new_data = []
    other_columns = df.columns.difference(['L1', 'L2', 'L3', 'Desc1', 'Desc2', 'Desc3', 'Descriptors']) # We retain columns other than the ligands SMILES
    
    for index, row in df.iterrows():
        other_values = row[other_columns] # We retain values from the original ligand for the other columns

        # Generate permutations for L1, L2, and L3 indices
        for perm_indices in permutations([0, 1, 2]):
            # Apply permutation indices to L1, L2, and L3
            permuted_L = [row['L{}'.format(i+1)] for i in perm_indices]
            # Apply the same permutation to Desc1, Desc2, and Desc3
            permuted_Desc = [row['Desc{}'.format(i+1)] for i in perm_indices]
            # Combine other values, permuted L, and permuted Desc into a new row
            new_row = other_values.tolist() + permuted_L + permuted_Desc
            new_data.append(new_row)

    expanded_df = pd.DataFrame(new_data, columns=list(other_columns) + ['L1', 'L2', 'L3', 'Desc1', 'Desc2', 'Desc3'])
    
    #Getting the final descriptor of the permutated complex
    expanded_df['Descriptors'] = expanded_df.apply(concatenate_float_lists, axis=1)

    # Now we have all 6 permutations of L1, L2 and L3 for all complexes. In many cases, though, we have
    # two identical ligands (at least) such as L1 = L2. This means that some permutations are actually identical
    # and we have introduced duplicates in our dataset. We need to drop those.
    
    expanded_df['Ligands_Sum'] = expanded_df.L1 + expanded_df.L2 + expanded_df.L3
    drop_duplicates(expanded_df, 'Ligands_Sum', print_length=False)
    expanded_df.reset_index(drop=True, inplace=True)

    return expanded_df



# Different Splittings 


def df_split(df, sizes=(0.9, 0.1), seed=1):

    """
    This function splits the dataset intro a train and a test set. It makes sure that all the permutations for the same complex
    are stored in the same set. 
    
    Args:
        df : dataframe 
    
    returns:
        train (list): list of indices of the train set
        test (list): list of indices of the test set
    """

    assert sum(sizes) == 1

    ID = list(set(df.ID)) # We extract the unique IDs for each unique complex / each permutation group

    # Split
    train, val, test = [], [], []

    random.seed(seed)
    random.shuffle(ID) # Randomly shuffle unique IDs
    train_range = int(sizes[0] * len(ID))
    #val_range = int(sizes[1] * len(ID))

    for i in range(train_range):
        selected = df[df['ID'] == ID[i]]
        indices = selected.index.tolist() # The absolute indices of these entries (which are consecutive in the dataframe) are added to the train set
        for i in indices:
            train.append(i)

    #for i in range(train_range, train_range + val_range):
    #    selected = df[df['ID'] == ID[i]]
    #    indices = selected.index.tolist()
    #    for i in indices:
    #        val.append(i)

    for i in range(train_range, len(ID)):
        selected = df[df['ID'] == ID[i]]
        indices = selected.index.tolist()
        for i in indices:
            test.append(i)

    print(f'train length : {len(train)} | test length : {len(test)}')

    return train, test


# Cross Validation and Scaling the data 

def prepare_train_set(df, train_idx, test_idx, permutation=True): 

    """
    This function expands the training set with non-redundant permutations, fits the scaler on the train set, 
    scales the train set and then the test set. Scaling the data is necessary when working with molecular descriptors,
    which often contain values that are too large for the model. We automatically permutate then ligands when calling 
    the descriptors, but it is still possible to set 'permutations' to False, for code analysis purposes.

    Args:
        df : pd.dataframe 
        train_idx (list): train indices
        test_idx (list): test indices
        permutation (bool): whether the train set should be augmented with permutations of ligands or not

    returns:
        X_train (array): train set of features
        X_test (array): test set of features
        y_train (array): target array of the train set 
    """
    
    scaler = StandardScaler()
    
    train_set = df.iloc[train_idx] # We select the complexes assigned to training
    if permutation == True:
        train_set = ligands_permutation(train_set) # We augment the dataset with their permutations
        #print(train_set)
    
    X_train = np.array(train_set['Descriptors'].values.tolist()) # The actual augmented training set
    X_train = scaler.fit_transform(X_train) # Fit the scaler and scale the training set
    y_train = np.array(train_set['pIC50'].values.tolist())

    test_set = df.iloc[test_idx] # Select the complexes assigned to testing
    X_test = np.array(test_set['Descriptors'].values.tolist()) # Actual test set
    X_test = scaler.transform(X_test) # Scale the test set with the parameters acquired on the training set
    
    return X_train, y_train, X_test


def cross_validation(df, indices, X, y, rf, descriptors=False, permutation=True):

    """
    This function runs the cross validation of a model on a dataframe. It is is used solely in the notebooks 
    and outputs result metrics along with the true and predicted data as 2 arrays, to allow plotting the data afterwards.
    
    Another cross validation function is defined in the cross_val.py file. This cross validation function is not
    used in the notebooks, it is only called within the model_fp_selection folder, and is used for hyperparameter
    optimisation. It is defined separately because it outputs different information than this one. 
    
    Args:
        df : dataframe 
        indices (list) : list of lists of indices for the splitting
        X (array): array of the features
        y (array): array of the target 
        rf (model): model architecture
        descriptors (bool):
                - True for Molecular Descriptors. It will pass through the 'prepare_train_set' function.
                - False for Fingerprints
        permutation (bool): whether the train set should be augmented with permutations of ligands or not

    returns:
        y_data (array): array of real target values
        y_predictions (array): array of predicted target values
    """

    y_data= []
    y_predictions = []

    for i, (train_idx, test_idx) in enumerate(indices):
        print("CV iteration", i)
       
        if descriptors==True :
            # Getting the scaled and augmented training set, and the scaled test set
            X_train, y_train, X_test = prepare_train_set(df, train_idx, test_idx, permutation) 
        else : 
            X_train, y_train, X_test = X[train_idx], y[train_idx], X[test_idx]

        rf.fit(X_train, y_train)   # Fit model to data
        y_pred = rf.predict(X_test) # Predict values
        y_data.extend(y[test_idx])
        y_predictions.extend(y_pred) # Update lists
    
    y_data = np.array(y_data)
    y_predictions = np.array(y_predictions)
    
    metrics = obtain_metrics(y_data, y_predictions)
    print(metrics)
    return y_data, y_predictions



def get_indices(df, CV, sizes=(0.9, 0.1)):
    
    """
    This function gets the indices for the cross validation splitting.
    
    Args:
        df : dataframe 
        CV (int): number of folds in the cross validation 
        sizes (tuple): sizes of the train and test sets as percentage

    returns:
        indices (list): list of lists with the splitting indices
    """

    indices = []
    
    for seed in range(CV):
        random.seed(seed)
        train_indices, test_indices = df_split(df, sizes=sizes, seed=seed)
        print(len(train_indices), len(test_indices))
        indices.append([train_indices, test_indices])
    
    return indices


def get_indices_doi(df, CV, sizes=(0.9, 0.1), seeder=0):

    """
    This function gets the indices for cross validation DOI splitting. The complexes issued from the same paper end up in the same set
    (either train or test), so that predictions are always made on complexes from a different paper than those in the train (similar
    to real conditions). End to end, the test set outputs cover the entierity of the dataset, no more, no less. 
    
    Args:
        df : dataframe 
        CV (int): number of folds in the cross validation 
        sizes (tuple): sizes of the train and test sets as percentages
    
    returns:
        indices_final (list): list of lists with the DOI splitting indices
    """
    
    indices_final = []
    DOI = list(set(df.DOI)) # We extract the unique DOIs for each unique complex / each permutation group
    k = 1 # This counts the cross-validation iteration, allowing to displace the sample used for test set at each iteration
    random.seed(seeder)
    random.shuffle(DOI) # Randomly shuffle unique DOIs
    test_range = sizes[1] * len(DOI)
    
    for seed in range(CV):
        
        # Split
        train, test = [], []
        #val_range = int(sizes[1] * len(ID))

        for i in range(round((k-1) * test_range)):
            selected = df[df['DOI'] == DOI[i]] # We choose a DOI at random and select all permutations of the all complexes with this DOI
            indices = selected.index.tolist() # The indices of these entries are added to the training set
            for j in indices:
                train.append(j)

        for i in range(round((k-1) * test_range), round(k * test_range)): 
            # This k-dependent interval ensures we go over the whole dataset : at each iteration we take the next 10% of the DOI list, in the end we go
            # through the whole DOI list, hence through the whole dataset.
              
            selected = df[df['DOI'] == DOI[i]] # We choose a DOI at random and select all permutations of the all complexes with this DOI
            indices = selected.index.tolist() # The indices of these entries are added to the test set
            for j in indices:
                test.append(j)

        for i in range(round(k * test_range), len(DOI)):
            selected = df[df['DOI'] == DOI[i]]
            indices = selected.index.tolist()
            for j in indices:
                train.append(j)
        
            
        print(f'train length : {len(train)} | test length : {len(test)}')

        k+=1

        indices_final.append([train, test])
    
    return indices_final


def get_indices_scaff(mols, CV, sizes=(0.9, 0.1), seeder=0):

    """
    This function gets the indices for cross validation Scaffold splitting. The complexes sharing a similar scaffold end up in the 
    same set (either train or test), so that predictions are always made on complexes from a different paper than those in the train (similar
    to real conditions). The test sets are never overlapping and put end to end they cover exactly the totality of the data points.
    
    Args:
        df : dataframe 
        CV (int): number of folds in the cross validation 
        sizes (tuple): sizes of the train and test sets as percentages
    
    returns:
        indices_final (list): list of lists with the scaffold splitting indices
    """

    indices_final = []
    scaff_dict = scaffold_to_smiles(mols, use_indices=True) # We extract the scaffolds and the indices of the complexes having each scaffold
    scaffolds = list(scaff_dict.keys()) # Define a list of the possible scaffolds
    k = 1 # Iteration count
    random.seed(seeder)
    random.shuffle(scaffolds) # Randomly shuffle scaffolds
    test_range = sizes[1] * len(scaffolds)
    
    for seed in range(CV):
        
        # Split
        train, test = [], []
        #val_range = int(sizes[1] * len(ID))

        for i in range(round((k-1) * test_range)):
            scaff = scaffolds[i] # Choose a random scaffold in the shuffled list
            indices = list(scaff_dict[scaff]) # Get indices of all complexes having that scaffold 
            for j in indices:
                train.append(j)

        for i in range(round((k-1) * test_range), round(k * test_range)): 
            
            # This k-dependent interval ensures every single point of the dataset ends up in one of the test sets
            
            scaff = scaffolds[i]
            indices = list(scaff_dict[scaff])
            for j in indices:
                test.append(j)

        for i in range(round(k * test_range), len(scaffolds)):
            scaff = scaffolds[i]
            indices = list(scaff_dict[scaff])
            for j in indices:
                train.append(j)
        
            
        print(f'train length : {len(train)} | test length : {len(test)}')

        k+=1

        indices_final.append([train, test])
    
    return indices_final




#Scaffold Functions 

def generate_scaffold(mol, include_chirality=False):
    """
    Computes the Bemis-Murcko scaffold for a SMILES string.

    Args:
        mol (RDkit mol): A SMILES string or an RDKit molecule.
        include_chirality (bool): Whether to include chirality.
    
    returns:
        scaffold (RDKit mol)
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold


def scaffold_to_smiles(mols,
                       use_indices=False):
    """
    Computes scaffold for each smiles string and returns a mapping from scaffolds to sets of SMILES.

    Args:
        mols (list): A list of SMILES strings or RDKit molecules.
        use_indices (bool): Whether to map to the SMILES' index in all_smiles rather than mapping
        to the SMILES string itself. This is necessary if there are duplicate SMILES. Default is False

    returns:
        scaffolds (dict): A dictionary mapping each unique scaffold to all SMILES (or SMILES indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds




# Functions for the SHAP Anlaysis

def get_bit_smiles_pairs(df, top_features):

    """
    Looks for top features in each SMILES string and returns a mapping from these bits to SMILES.

    Args:
        df: pd.dataframe
        top_features (list): a list of bits corresponding to the top features (see the SHAP analysis in feature analysis notebook)

    returns:
        bit_smiles_pair (list): a list of 2-element lists with the top-feature bit and the corresponding sub-structure SMILES
    """

    all_smiles = list(df['L1']) + list(df['L2']) + list(df['L3'])
    all_smiles_unique = list(set(all_smiles))

    bit_smiles_pairs = []

    for bit in top_features:
        for smiles in all_smiles_unique:
            info = {}
            fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius=2, nBits=2048, bitInfo=info)
            if bit in info.keys():
                bit_smiles_pairs.append((smiles, bit))
                break 
                
    return bit_smiles_pairs

def gethighlight(bit_smiles_pairs) :

    """
    Stores list of molecule substructures to highlight the features of bit_smiles_pairs in a molecule. 

    Args:
        bit_smiles_pairs (list): a list of 2-element lists with the top-feature bit and the corresponding sub-structure SMILES

    returns:
        highlights (list): a list of molecule substructures 
        highlights_texts (list): a list of strings containging feature bit indices
    """

    highlights = []
    highlights_texts = []
    
    for pair in bit_smiles_pairs: 
        smiles = pair[0]
        bit = pair[1]
    
        mol = Chem.MolFromSmiles(smiles)
    
        info = {}
        rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, bitInfo=info)
        atomID, radius = info[bit][0]
  
        if radius>0:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atomID)
            atomsToUse=[]
            for b in env:
                atomsToUse.append(mol.GetBondWithIdx(b).GetBeginAtomIdx())
                atomsToUse.append(mol.GetBondWithIdx(b).GetEndAtomIdx())
            atomsToUse = list(set(atomsToUse))
        else:
            atomsToUse = [atomID]
        highlights.append(atomsToUse)
        highlights_texts.append('Feature number' + str(bit))
    
    return highlights, highlights_texts









# Fonction pour selection.py - calcul des fingerprints 


def get_complexes_fingerprints(df, rad, nbits):
    """
    Prepare the input dataframe to compute morgan fingerprints and filter any duplicate by taking the average of different target values.

    Args:
        df (pd.DataFrame): input dataframe of metal complexes to be predicted
        rad (int): the radii of the fingerprints
        nbits (int): the number of bits for the fingerprints

    Returns:
        pd.DataFrame: the update dataframe
    """

    #Getting the Fingerprint of each ligand
    df['FP1'] = df['L1'].apply(lambda x: 
            get_morgan_fp_from_smiles(x, rad, nbits))
    df['FP2'] = df['L2'].apply(lambda x: 
            get_morgan_fp_from_smiles(x, rad, nbits))
    df['FP3'] = df['L3'].apply(lambda x: 
            get_morgan_fp_from_smiles(x, rad, nbits))

    #Getting the final fingerprint of the complex
    add_lists = lambda row: row['FP1'] + row['FP2'] + row['FP3']
    df['Fingerprint'] = df.apply(add_lists, axis=1)
    
    return df




# Distance functions 


def tanimoto_distance(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_squared_a = np.dot(vector1, vector1)
    norm_squared_b = np.dot(vector2, vector2)
    
    tanimoto_dist = 1 - dot_product / (norm_squared_a + norm_squared_b - dot_product)
    
    return tanimoto_dist

from scipy.spatial.distance import cosine

def cosine_distance(fingerprint1, fingerprint2): 
    cos_d = cosine(fingerprint1, fingerprint2)
    
    return cos_d


def get_distances(df, fingerprint1, distance, desc = False): 
    """
    Returns the list of distances between a complex and the complexes of the dataset.
    
    param : dataset : a dataframe with a column 'Fingerprint'
    param : fingerprint1: the fingerprint of the generated compound
    param : distance : the name of the distance to be used, either cosine_distance or tanimoto_distance
    return : a list of floats (the distances)
    """

    distances = []

    for index, row in df.iterrows():
        
        if desc == True : 
            fingerprint2 = row['Descriptors']
        else : 
            fingerprint2 = row['Fingerprint']
        
        dis = distance(fingerprint1, fingerprint2)
        if dis != 0 : 
            distances.append(dis)
      
    return distances



def get_minimal_distances_dataset(df, dataset, distance, desc = False): 
    """
    Returns the list of minimal distances between each complex of the dataset and the rest of the complexes.
    param : df : the dataframe of generated compounds 
    param : dataset : the dataframe of the original dataset of compounds. 
    param : distance : the name of the distance to be used, either cosine_distance or tanimoto_distance
    return : a list of floats (the distances).
    """
    
    minimal_distances = []
    
    for index, row in df.iterrows():
        if desc == True : 
            fingerprint = row['Descriptors']
            #we get the distance between a complexe to-be-tested and the entire dataset
            list = get_distances(dataset, fingerprint, distance, desc = True)
            if len(list) !=0 :
                minimal_distance = min(list) 
                minimal_distances.append(minimal_distance)
        else : 
            fingerprint = row['Fingerprint']
            #we get the distance between a complexe to-be-tested and the entire dataset
            minimal_distance = min(get_distances(dataset, fingerprint, distance)) 
            minimal_distances.append(minimal_distance)
        
    return minimal_distances


def get_mean_distances_dataset(df, dataset, distance, desc = False): 
    """
    Returns the list of mean distances between each complex of the dataset and the rest of the complexes.
    param : df : the dataframe of generated compounds 
    param : dataset : the dataframe of the original dataset of compounds. 
    param : distance : the name of the distance to be used, either cosine_distance or tanimoto_distance
    return : a list of floats (the distances).
    """
    
    mean_distances = []
    
    for index, row in df.iterrows():
        if desc == True : 
            fingerprint = row['Descriptors']
            #we get the distance between a complexe to-be-tested and the entire dataset
            list_dist = get_distances(dataset, fingerprint, distance, desc = True)
            if len(list) !=0 :
                mean_distance = np.mean(list_dist) 
                mean_distances.append(mean_distance)
        else : 
            fingerprint = row['Fingerprint']
            #we get the distance between a complexe to-be-tested and the entire dataset
            mean_distance = np.mean(get_distances(dataset, fingerprint, distance)) 
            mean_distances.append(mean_distance)
        
    return mean_distances