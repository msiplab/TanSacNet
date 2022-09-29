# TanSacNet
Project for developing tangent space adaptive control networks

## Summary

TanSacNet is an abbreviation of Tangent Space Adaptive Control Networks. 
This package is developed for

* Experiments,
* Development and
* Implementation

of sparsity-aware image and volumetric data restoraition algorithms.

We have prepared custom layer classes with 
Deep Learning Toolbox. It is now easy to incorporate them into flexible 
configurations and parts of your network.

Information about TanSacNet Package is given in Contents.m. The HELP command can 
be used to see the contents as follows:

       >> help TanSacNet
        
       Project for developing tangent space adaptive control networks
         
           Files
             mytest     - Script of unit testing for SaivDr Package
             quickstart - Quickstart of *SaivDr Package*
             setpath    - Path setup for *SaivDr Package*
          
           * Package structure
               
               + saivdr -+- testcase -+- dcnn
                         |            |
                         |            +- sparserep 
                         |            |                         
                         |            +- embedded                          
                         |            |
                         |            +- dictionary  -+- nsolt     -+- design
                         |            |               |
                         |            |               +- nsoltx    -+- design
                         |            |               |
                         |            |               +- nsgenlot  -+- design
                         |            |               |
                         |            |               +- nsgenlotx -+- design
                         |            |               |
                         |            |               +- olaols
                         |            |               |
                         |            |               +- olpprfb
                         |            |               |
                         |            |               +- udhaar 
                         |            |               |
                         |            |               +- generalfb
                         |            |               |
                         |            |               +- mixture
                         |            |               |
                         |            |               +- utility
                         |            |
                         |            +- restoration -+- ista
                         |            |               |
                         |            |               +- pds
                         |            |               |
                         |            |               +- metricproj
                         |            |               |
                         |            |               +- denoiser
                         |            |
                         |            +- degradation -+- linearprocess
                         |            |               |
                         |            |               +- noiseprocess
                         |            |
                         |            +- utility 
                         |
                         +- dcnn
                         |
                         +- sparserep
                         |                         
                         +- embedded
                         |
                         +- dictionary  -+- nsolt     -+- design
                         |               |             |
                         |               |             +- mexsrcs
                         |               |        
                         |               +- nsoltx    -+- design
                         |               |             |
                         |               |             +- mexsrcs
                         |               |
                         |               +- nsgenlot  -+- design
                         |               |         
                         |               +- nsgenlotx -+- design
                         |               |         
                         |               +- olaols
                         |               |         
                         |               +- olpprfb
                         |               |         
                         |               +- udhaar 
                         |               |
                         |               +- generalfb
                         |               |
                         |               +- mixture
                         |               |
                         |               +- utility
                         |
                         +- restoration -+- ista  
                         |               |
                         |               +- pds
                         |               |
                         |               +- metricproj
                         |               |
                         |               +- denoiser
                         |            
                         +- degradation -+- linearprocess
                         |               |
                         |               +- noiseprocess
                         |
                         +- utility
    
## Requirements
 
* MATLAB R2022b is recommended.
 * Signal Processing Toolbox
 * Image Processing Toolbox
 * Optimization Toolbox

## Recomendation
 
 * Deep Learning Toolbox
 * Global Optimization Toolbox 
 * Parallel Computing Toolbox
 * MATLAB Coder
 * GPU Coder

## Brief introduction
 
1. Change current directory to where this file contains on MATLAB.
2. Set the path by using the following command:

        >> setpath

3. Build MEX codes if you have MATLAB Coder.

        >> mybuild

4. Several example codes are found under the second layer directory 
   'examples' of this package. Change current directory to one under 
   the second layer directiory 'examples' and execute an M-file of 
   which name begins with 'main,' such as
 
        >> main_xxxx
 
   and then execute an M-file of which name begins with 'disp,' such as
 
        >> disp_xxxx
 
## Contact address
 
     Shogo MURAMATSU,
     Faculty of Engineering, Niigata University,
     8050 2-no-cho Ikarashi, Nishi-ku,
     Niigata, 950-2181, JAPAN
     http://msiplab.eng.niigata-u.ac.jp/
 
## References

* 
 
## Acknowledgement
 
This work was supported by JSPS KAKENHI Grant Number JP22H00512.
 
## Contributors

### Developpers
* Yasas GODAGE,  2022-
* Eisuke KOBAYASHI, 2022-
 
### Test contributers
* 
