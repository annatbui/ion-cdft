This folder contains the trained Keras models for the c1 functionals of
the mimic short-range fluids for the RPM, the size-asymmetric PM and
the multivalent 2:1 PM.
The files include:

	RPM:
		 RPM_H.keras
		 RPM_O.keras
		 extra/RPM_H_withpc.keras
		 extra/RPM_O_withpc.keras
	PM:	 
                 PM_H.keras
                 PM_O.keras

	21PM:
                 21PM_H.keras
                 21PM_O.keras

Models without 'withpc' are trained with randomised inhomogeneous
profiles and electric fields. Models with 'withpc' are trained also
with pair correlation matching. The labels 'H' and 'O' are to 
distinguish between two ionic species. 

The models were trained with Tensorflow/Keras installed with Nvidia 
GPU support. If you run into Tensorflow/Keras compatibility problems
due to hardware or module versions, it might be easiest to 
train new models with the datasets at https://zenodo.org/records/15085645. 


