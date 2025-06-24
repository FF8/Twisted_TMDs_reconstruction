# Twisted TMDs atomic reconstruction #



## General info

This code computes the lattice reconstruction of twisted homobilayers of TMDs. 
More details can be found in the following references:

 * Phys. Rev. Lett. 124, 206101 –20 May 2020<br />
 * Phys. Rev. B 104, 125440 – 28 September 2021<br />
 * Scientific Reports volume 11, Article number: 13422 (2021)<br /> 
 * Nature Reviews Physics volume 4, page 632 (2022)<br />





## Instructions

### Create a virtual environment

Create a virtual enviroment:

`python -m venv venv`

### Activate your virtual enviroment:

`source venv/bin/activate`

### Install python packages:

`pip install -r requirements.txt`



## Tutorial

    This tutorial shows how to run a calculations for AP-MoS2 with a twist angle of 0.6º:



    Command to run a calculation (works in Ubuntu terminal):

        `python3 -u reconstruction.py > output_reconst_MoS2_AP0.6`


    Command to run a calculation in background (works in Ubuntu)

        `nohup python3 -u reconstruction.py > output_reconst_MoS2_AP0.6 &`




Input files:

    *   config.ini: Paramaters for the reconstuction calculations
    *   reconstruction.py: This is the main code, reconstruction for homobilayers
    *   derivative_adhenergy.py: This is the code with the derivative of the Adhesion energy w.r.t displacement field vectors (it will be imported in reconstruction.py)

Output files:

    *   ubx_array0.6AP_MoS2.npy, uby_array0.6AP_MoS2.npy, utx_array0.6AP_MoS2.npy, uty_array0.6AP_MoS2.npy: displacement field numpy arrays
    *   vecfield_sol_bot0.6AP_MoS2.dat, vecfield_sol_top0.6AP_MoS2.dat: displacement field vectors readeable forma
    *   equations_writer_MoS2_0.6AP.py:  File created by reconstruction.py that contains all information about the system of equations that gekko needs to read
    *   Output_vec.txt: this is just a file with the points of the grid and moire supercell
    *   output_reconst_MoSe2_AP0.6: This is the main output file.  Important to see the what is the current progress of the calculation.
