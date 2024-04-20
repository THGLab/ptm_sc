'''

File: chi_phi.py
Author: Shubhankar Abhang Naik
Labs and Partners: Teresa Head-Gordon Lab @ UC Berkeley and Julie Forman-Kay Lab @ University of Toronto
Contact: shubhankarnaik@berkeley.edu, Teresa Head-Gordon Lab
Description: Given an input of pdb files of protein residues, it creates a new .pkl file with all of the phi-psi bins in the [-180,180) range
             and the associated chi angles.

==========

NOTE (Prior to running code):
    1. Ensure that interested residue and its associated atoms to calculate chi angles are updated in the chi_atoms dictionary
    2. Ensure that correct chi values are being extracted in the "for filename in list_of_files" for loop
    3. Ensure that all data are saved to correct files -- these need to be manually changed at the code's foot

'''




import numpy as np
import os
from Bio.PDB import PDBParser
import gzip
from Bio.PDB import calc_dihedral, PDBParser
import pickle

# Dictionary defining chi atoms for different residues
# *** Refer to Note #1 at code's head
chi_atoms = {
        "chi1" : {
            "SEP" : ['N', 'CA', 'CB', 'OG'],
            "PTR": ['N', 'CA', 'CB', 'CG'],
            "TPO": ['N','CA','CB','OG1'],
			"SER" : ['N', 'CA', 'CB', 'OG'],
			"THR" : ['N', 'CA', 'CB', 'OG1'],
            "M3L" : ['N', 'CA', 'CB', 'CG'],
            "TYR" : ['N', 'CA', 'CB', 'CG'],
            "THR" : ['N', 'CA', 'CB', 'OG1'],
            "AGM" : ['N', 'CA', 'CB', 'CG'],
            "ALY" : ['N', 'CA', 'CB', 'CG'],
            "LYS" : ['N', 'CA', 'CB', 'CG'],
            "TRP" : ['N', 'CA', 'CB', 'CG']
            },

        "chi2" : {
            "SEP" : ['CA', 'CB', 'OG', 'P'],
            "PTR" : ['CA', 'CB', 'CG', 'CD1'],
            "TPO" : ['CA', 'CB', 'OG1', 'P'],
            "M3L" : ['CA', 'CB', 'CG', 'CD'],
            "TYR" : ['CA', 'CB', 'CG', 'CD1'],
            "AGM" : ['CA', 'CB', 'CG', 'CD'],
            "THR" : ['CA', 'CB', 'CG', 'CD1'],
            "ALY" : ['CA', 'CB', 'CG', 'CD'],
            "LYS" : ['CA', 'CB', 'CG', 'CD'],
            "TRP" : ['CA', 'CB', 'CG', 'CD1']
            },

        "chi3" : {
            "SEP" : ['CB', 'OG', 'P', 'O1P'],
            "PTR": ['CE1', 'CZ', 'OH', 'P'],
			"TPO": ['CB','OG1','P','O1P'],
            "M3L" : ['CB', 'CG', 'CD', 'CE'],
            "AGM" : ['CB', 'CG', 'CD', 'NE1'],
            "ALY" : ['CB', 'CG', 'CD', 'CE'],
            "LYS" : ['CB', 'CG', 'CD', 'CE']
            },

        "chi4" : {
		    "PTR": ['CZ','OH','P','O1P'],
            "M3L" : ['CG', 'CD', 'CE', 'NZ'],
            "AGM" : ['CG', 'CD', 'NE1', 'CZ'],
            "ALY" : ['CG', 'CD', 'CE', 'NZ'],
            "LYS" : ['CG', 'CD', 'CE', 'NZ']
        },

        "chi5" : {
            "M3L" : ['CD', 'CE', 'NZ', 'CM1']
        }
}

# Converts angle in [-pi, pi) to [-180, 180)
# A function that converts an angle from radians in the range [-pi, pi) to degrees in the range [-180, 180).
def angle_convert_180(angle):
    """
    Converts angle in [-pi, pi) to [0, 360)

    Parameters:
    ==========
    angle: angle in [-pi, pi) range

    Returns:
    ==========
    angle: angle in [-180, 180) range

    """
    angle *= 180/np.pi
    return angle

# Converts angle in [-pi, pi) to [0, 360)
# A function that converts an angle from radians in the range [-pi, pi) to degrees in the range [0, 360).
# Stores angles in list called update
def angle_convert_360(list):
    """
    Converts angle in [-pi, pi) to [0, 360)

    Parameters:
    ==========
    list: list of angles in [-pi, pi) range

    Returns:
    ==========
    update: list of angles in [0, 360) range

    """
    update = []
    for angle in list:
        angle *= 180/np.pi
        if angle < 0:
            angle += 360.
        update.append(angle)    
    return update


# Function to extract chi angle values for specific residues
# Extracts chi angle values for a given residue and specified chi angles.
def get_chi_value(structure, resname, chis):
    """
    Extract torsion angle values of specified chi angles of residue with resname.

    Parameters:
    ==========
    structure: PDBParser structure object, the protein structure to query from.
    resname: str, 3 letter code of PTM to query.
    chis: list of int or int, the index/indices of chi torsion angle to query.
          ex. [1, 2, 3] for [chi1, chi2, chi3]

    Returns:
    ==========
    chi_values: list of chi torsion angle values of shape (number of residues in structure, length of chis);
                angles in [-180, 180) degrees.
    """

    chi_values = []
    chi_names = ["chi%1d" % int(i) for i in chis]

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() == resname:
                    v = []
                    for chi in chi_names:
                        try:
                            atoms = (residue[a].get_vector() for a in chi_atoms[chi][resname])
                            r = calc_dihedral(*atoms)
                            v.append(angle_convert_180(r))
                        except KeyError:
                            break

                    chi_values.append(v)

    return chi_values


# results in array of backbone structure of that file
# calculates coordinates of all 5 atoms in molecule we are analyzing
def get_backbone_pdb(models, protein):
    '''
    Calculate coordinates of 5 atoms in molecule(s) we are analyzing to later calculate dihedral angles

    Parameters:
    ==========
    models:  structural model of molecule
    protein: PTM 3 letter abbreviation

    Returns:
    ==========
    atoms: list of coordinates of atoms where len(atoms) % 5 == 0 in molecule
    
    '''
    i_count = 0
    n_atoms = 0
    residue_list = []
    time = 0
    counter = 0

    model = next(models)
    i_count += 1
    atoms = []

    try:
        for residue in model.get_residues():
            residue_list.append(residue.get_resname())

        for residue in model.get_residues():
            number = 0
            if residue.get_resname() == protein.upper():
                time += 1

                atoms += [residue['N'].get_coord(), residue['CA'].get_coord(), 
                      residue['C'].get_coord()]

                for residue in model.get_residues():
                    if number == (counter - 1) and residue_list[counter] == protein.upper():
                        if time > 1:
                            atoms.insert(-3, residue['C'].get_coord())
                        else:
                            atoms[:0] = [residue['C'].get_coord()]
                    if number == (counter + 1) and residue_list[counter] == protein.upper():
                        atoms.append(residue['N'].get_coord())
                    number += 1

                counter += 1
            else:
                counter += 1
    
    except KeyError:
        return()
    atoms = np.array(atoms)
    if i_count == 1:
        n_atoms = atoms.shape[0]
    else:
        assert atoms.shape == (n_atoms, 3)

    return atoms


# calculates torsion angles [phi, psi]
def calc_torsion_angles(
        coords,
        ARCTAN2 = np.arctan2,
        CROSS = np.cross,
        DIAGONAL = np.diagonal,
        MATMUL = np.matmul,
        NORM = np.linalg.norm):
    
    """
    Extract torsion angle values of specified chi angles of residue with resname.

    Parameters:
    ==========
    coords: a list of coordinates of atoms in molecule where len(atoms) % 5 == 0

    Returns:
    ==========
    a pair of a phi and psi angle

    """

    # requires
    assert coords.shape[0] >= 3
    assert coords.shape[1] == 3

    crds = coords.T
    q_vecs = crds[:,1:] - crds[:, :-1]
    cross = CROSS(q_vecs[:, :-1], q_vecs[:, 1:], axis = 0)
    unitary = cross/NORM(cross, axis = 0)

    # components
    u0 = unitary[:, :-1]
    u1 = unitary[:, 1:]
    u3 = q_vecs[:, 1:-1]/NORM(q_vecs[:, 1:-1], axis = 0)
    u2 = CROSS(u3, u1, axis = 0)
    cos_theta = DIAGONAL(MATMUL(u0.T, u1))
    sin_theta = DIAGONAL(MATMUL(u0.T, u2))

    # torsion angles
    return -ARCTAN2(sin_theta, cos_theta)


# Calculates torsion angles and correctly identifies phi and psi angles from list
# Converts angles into [-180,180) scale
def calc_phi(list):
    """
    Calculates and identifies torsion angles (phi,psi). Converts angles into [-180, 180) scale
    
    Parameters:
    ==========
    coords: a list of coordinates of atoms in molecule where len(atoms) % 5 == 0

    Returns:
    ==========
    angles: list containing list(s) of phi/psi angle pairings, depending on number of 5 atom molecules inputed via coords

    """
    angles = []
    results = []
    length = len(list)

    if length >= 3:
        degrees = angle_convert_360(calc_torsion_angles(list))
    track = 0
    for i in range(int(length/5)):
        results = []
        phi = degrees[track]
        psi = degrees[track + 1]
        track += 5

        if phi > 180:
            phi -= 360
        if psi > 180:
            psi -= 360

        results.append(phi)
        results.append(psi)
        angles.append(results)
    return angles



# ********* Beginning Of Executable Code*********

list_of_files = []  # List to store all PDB files that need to be analyzed


print("Provide Path Where Files Stored")
path = input().encode('unicode_escape').decode()

# Using path of where PDB files are stored, accesses files and creates a list of file names
for root,dirs,files in os.walk(path):
    for file in files:
        if(file.endswith(".gz")):
            list_of_files.append(os.path.join(root,file))

print("Provide 3 Letter Protein Abbreviation")
protein = input().upper()


'''
Creates a list of lists using bin size inputed
Each list contains the phi bin number and the psi bin number
i.e. if bin size = 10, then the first bin would be [-180, -180], where the phi angle would be from -180 to -170 and psi would be -180 to -170
    Last bin would be [170, 170], where phi angle is 170 to 180 and psi angle is 170 to 180
'''
bin_list_phi = []
print("How large do you want bins to be in degrees?")
bin_size = int(input())
for x in range(int(2*180/bin_size + 1)):
    number = -180+bin_size*x
    bin_list_phi.append(number)


phi_bins = [] # list of lists. Each list represents a different backbone bin, containing values in for loop below
for i in range(len(bin_list_phi) - 1):
    for j in range(len(bin_list_phi) - 1):
        phi_psi = []
        phi_psi.append(bin_list_phi[i]) # Phi angle bin number
        phi_psi.append(bin_list_phi[j]) # Psi angle bin number
        phi_psi.append([]) # Bin where chi 1/2/3 angle pairings are stored that have been calculated from same molecules of phi/psi in this bin
        phi_psi.append(0) # Number of phi/psi angle pairings in backbone bin
        phi_psi.append([]) # Bin where phi/psi angles are stored
        phi_bins.append(phi_psi) # Storing a list of above values into global list


count = 0
all_phi = []
all_chi = []
# Runs through list of files and unzips them
# Calculates phi/psi and chi angles based on molecules and aotm coordinates
# Updates list of phi bins according to torsion angles
for filename in list_of_files:
    print(filename, "PHI/PSI RUNNING")
    counter = 0
    phi_angles = []
    chi_angles = []

    # unzipping files to access data
    unzipfile = open("unzipped.ent", "wb")
    with gzip.open(filename, "rb") as Q:
        bindata = Q.read()
    unzipfile.write(bindata)
    unzipfile.close()
    unzippedfilename = "unzipped.ent"

    parser = PDBParser(PERMISSIVE = True, QUIET = True)
    data = parser.get_structure('data', unzippedfilename)
    model = data.get_models()
    coords = get_backbone_pdb(model, protein) # calculating coordinates of atoms in molecules

    # *** Refer to Note #2 at code's head
    chi_list = [1,2,3,4,5] # Chi values that we want to extract from pdb files
    chi = get_chi_value(parser.get_structure("2fat", "unzipped.ent"), protein, chi_list) # Calculating chi values/pairings



    if len(coords)/5 != 0 and (len(coords)/5) == len(chi) and all(len(sublist) == len(chi_list) for sublist in chi):
        phi_angles = calc_phi(coords) # calculating phi/psi angle pairings
        all_phi.append(phi_angles)
        count += len(phi_angles)
        chi_angles = chi
        all_chi.append(chi)
    else:
        continue

    # Apppropriately bins chi angles based on phi/psi angle values
    # Adds phi/psi and chi values to respective lists in bin and increases counter of number of phi/psi pairings in bin
    for a in phi_angles:
        for num in range(len(phi_bins)):
            if (a[0] > phi_bins[num][0]) and (a[0] < (phi_bins[num][0] + bin_size)) and (a[1] > phi_bins[num][1]) and (a[1] < (phi_bins[num][1] + bin_size)):
                phi_bins[num][2].append(chi_angles[counter])   
                phi_bins[num][3] += 1 
                phi_bins[num][4].append(a)       
        counter += 1
    
# If there are no phi/psi pairings calculated in a bin, then 'NA' is inputed in bin's list as its value
for x in range(len(phi_bins)):
	if len(phi_bins[x][2]) == 0:
		phi_bins[x][2].append("NA")
     

# Stores final list of bins into a .pkl file
directory = r"C:\Users\shubhankarnaik\Desktop\PTM_Project"
file_name = f"{protein}5_30bins.pkl"
folder_name = f"{protein}_Results"
file_path = os.path.join(directory, folder_name, file_name)

with open(file_path, "wb") as file:
    pickle.dump(phi_bins, file)

# Stores final list of chi values into a .pkl file
file_name = f"{protein}5_chi.pkl"
folder_name = f"{protein}_Results"
file_path = os.path.join(directory, folder_name, file_name)

with open(file_path, "wb") as file:
    pickle.dump(all_chi, file)

# Stores final list of phi/psi values into a .pkl file
file_name = f"{protein}5_phi.pkl"
folder_name = f"{protein}_Results"
file_path = os.path.join(directory, folder_name, file_name)

with open(file_path, "wb") as file:
    pickle.dump(all_phi, file)