# Simulation-of-GOx-binding-to-substrates-or-inhibitors
A theoretical study on the interactions between Glucose oxidase (GOx) and the substrates Glu, SP and MC.

In this work, we plumbed the interactions between Glucose oxidase (GOx) and the substrates Glu, SP and MC via molecular docking and molecular dynamics simulations, with focus on the potential inhibitory mechanisms of SP and MC at the GOx active site, to reveal the underlying mechanism regarding the photo-unlocking the catalytic activity of GOx. Specifically, the structures of GOx complexes with Glu, SP, and MC were constructed via molecular docking strategy firstly and then optimized via a 50-ns molecular dynamics simulation for enhancing accuracy and stability. The free binding energies of Glu, SP and MC to GOx, along with the energy decomposition data pertaining to interactions with specific residues, were determined via the Molecular Mechanics/Generalized Born Surface Area (MM/GBSA) method. The detailed calculation process and methodology were shown below:

（1）Molecular docking methods and parameters. 
Molecular docking was primarily performed using the AutoDock Vina program. The structure files of the target proteins (Glucose oxidase from Aspergillus niger) were obtained from the Protein Data Bank (PDB ID: 1CF3). Firstly, the structure files of both target protein and ligand molecules (Glu, SP, MC) were subjected to adding Gasteiger-Hücker empirical charges, merging nonpolar hydrogens, and setting rotatable bonds by AutoDockTools software. All σ-bonds between heavy atoms of the ligand and active site residues (TYR68, THR110, ARG176, THR333, ARG335, GLN347, PHE414, ASP416, SER422, ASP424, ARG512, ASN514, and TYR515) residues were set to rotatable bonds. Subsequently, a 30 × 30 × 30 Å docking box was set up at the active site of the target protein by the AutoDock Vina program. Inside the docking box, conformational search and global energy optimization of the ligand molecules were performed. Ultimately, after 200 independent runs of molecular docking, the result with the highest docking score was selected as the ligand-target protein complex. Default parameter values were used for running the AutoDock Vina program during the calculation process.

（2）Molecular dynamics simulation. 
In the present study, the Amber18 program package was primarily used for equilibrium molecular dynamics simulation. The ligand-target protein complex structures predicted by AutoDock Vina were used as the initial structures for molecular dynamics simulations. After all the ionizable residues of protein receptors were set to the standard protonated or deprotonated states, the hydrogen atoms were added by using the Tleap module of AMBER 18 program. Structural optimization of all ligands was conducted with HF/6-31G* using Gaussian09 software package. The atomic partial charges for all ligands were the restrained electrostatic potential (RESP) charges determined by fitting with the standard RESP procedure implemented in the Antechamber module of the AMBER 18 program. The FF14SB force field was chosen to describe the protein receptor residues, and the GAFF force field describes the ligand. Complexes were immersed in an octahedral box of TIP3P waters, using a 10 Å minimal distance from the protein receptor's surface to the box. The counterions (Na+ or Cl-) were added to the solvent to keep the system neutral. 50 ns molecular dynamics simulations of all complexes were performed using PMEMD.CUDA of AMBER 18 program.
The geometry of the system was minimized in three steps before MD simulation. First, the water molecules were refined through 4000 steps of steepest descent minimization followed by 2000 steps of conjugate gradient minimization, while the protein receptor was kept fixed with a constraint of 10.0 kcal mol^-1 Å^-2. Second, the solvent molecules and side chain of protein were relaxed through 4000 steps of steepest descent minimization followed by 2000 steps of conjugate gradient minimization, while the residue backbone was kept fixed with a constraint of 10.0 kcal mol^-1 Å^-2. Finally, all atoms were relaxed by 10000 cycles of minimization procedure (5000 cycles of steepest descent and 5000 cycles of conjugate gradient minimization).
The whole system was heated from 0 K to 310 K during 60 ps molecular dynamics and then the system was stabilized at 310 K using a temperature-coupling algorithm. Subsequently, a total of 50 ns molecular dynamics simulation was performed using an NVT ensemble. During the simulation, the particle mesh Ewald method was employed to calculate the long-range electrostatic interactions, while the SHAKE method was applied to constrain all covalent bonds involving hydrogen atoms to allow a time step of 2 fs. A 10 Å cutoff value was used for the non-bonded interactions.

（3）Trajectory analysis. 
After the simulation was completed, trajectories generated from MD simulations were analyzed via Cpptraj module of AmberTools18 software package. The root-mean-square deviation (RMSD) values were used to quantify the conformational changes of the ligand-target protein complex. The root-mean-square fluctuations (RMSF) plots were used to determine the relative fluctuation and flexibility of residues in the active pocket of target protein during the simulation. PyMOL software was employed to visualize the trajectories and to depict structural representations.

（4）Calculation of the binding energy. 
MM/GBSA (Molecular Mechanics/Generalized Born Surface Area) technique implemented in AMBER18 was used to calculate the binding energy of various ligands to target protein according to the following equation: 

∆E_bind = E_complex-(E_protein+E_ligand ) 
        = ∆E_vdw+ ∆E_ele+ ∆E_GB+ ∆E_SA

where ΔE_bind represents the binding energy in the solution consisting of the molecular mechanic's energy (ΔE_MM), and the solvation energy containing polar contribution (ΔE_GB) and nonpolar contribution (ΔE_SA). The ΔE_MM term includes ΔE_ele (electrostatic) and ΔE_vdw (van der Waals) energies and was calculated by the sander module of AMBER18. The polar contribution was calculated by using the GB mode, with the solvent and the solute dielectric constants set to 80 and 4, respectively. Additionally, the nonpolar energy was estimated, with a solvent-probe radius of 1.4 Å: ΔE_SA = 0.0072 × ΔSASA, by the LCPO method based on the SASA model. For each ligand-target protein complex, 500 snapshots were taken from 30 to 50 ns on the MD trajectories. Further, the total binding energy of various ligands to target protein was decomposed into ligand-residue pairs using the MM/GBSA decomposition analysis by the mm_pbsa program in AMBER18. The energy contribution for each ligand-residue pair also has four parts: van der Waals term (ΔE_vdw), electrostatic term (ΔE_ele), polar desolvation term (ΔE_GB), and nonpolar desolvation term (ΔE_SA).


