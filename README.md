# U2PM

## What is it in a nutshell?
__A 2D morphing-based load alleviation algorithm based on the unsteady panel method__ 

## Features
- Unsteady 2D panel code, using linear-strength vortex lines as singularities, including a constant-strength panel at the trailing edge; the implemented numerical method is described in:
<p align="center">
Dimitriadis, G. Unsteady aerodynamics: Potential and vortex methods. <em> Wiley, 2023. </em><a href="https://www.wiley.com/en-us/Unsteady+Aerodynamics%3A+Potential+and+Vortex+Methods-p-9781119762553">Book link</a>
</p>

- (To be implemented) An iterative morphing algorithm that takes as input:
    - the original aerodynamic state of the airfoil (shape, pressure distribution, inflow conditions)
    - the target pressure distribution desired for the same inflow conditions,
    
    and which then gives as output:
    - the shape that achieves the said pressure distribution

    This approach was described in:
    <p align="center">
    Lane, K., & Marshall, D. (2010). Inverse airfoil design utilizing CST parameterization. In <em> 48th AIAA Aerospace Sciences Meeting Including the New Horizons Forum and Aerospace Exposition, 2010</em>. <a href="https://doi.org/10.2514/6.2010-1228">Article link</a>
    </p>
## Assumptions
1. The airfoil coordinates of the flap are those downstream of the hinge point

## Prerequisites
**The results were obtained using python 3.11.9**. The workflow was only tested on Windows 10.

## Folder Structure 
```
└── 📁config ← files necessary for reproducible research
└── 📁data ← inputs and outputs of the model, as well as intermediate files
└── 📁docs 
└── 📁results ← results generated during the postprocessing
└── 📁tests
└── 📁src ← source files used for computing and plotting results  
```

## Installation
1. For results' reproducibility, install the recommended python libraries:

    ```pip install -r config/requirements.txt```
2. Test the separate functioning of the 2D panel code by running ```2D_airfoil.py```; you should obtain an animation in ```results/figures``` that is exactly the same as the reference one being provided

## Contribute
__TO DO__
## Support
__TO DO__
## License
__TO DO__