[![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=msiplab/TanSacNet) 

# TanSacNet
Project for developing tangent space adaptive control networks

## Summary

TanSacNet is an abbreviation of Tangent Space Adaptive Control Networks. 
This package is developed for

* Experiments,
* Development and
* Implementation

of sampling embedded manifold of image and volumetric data as a set of 
tangent space bases.

We have prepared custom layer classes with the Deep Learning Toolbox. 
It is easy to incorporate them into flexible configurations and 
parts of your network.

## Package structure
               
           tansacnet -+- results
                      |
                      +- data
                      |
                      +- code -+- examples -+- ...
                               |
                               +- tansacnet -+- testcase -+- lsun
                                             |            |
                                             |            +- utility 
                                             |
                                             +- lsun
                                             |
                                             +- utility       

## Requirements
 
 * MATLAB R2022b/R2023b is recommended.
 * Signal Processing Toolbox
 * Image Processing Toolbox
 * Optimization Toolbox

## Recommendation
 
 * Deep Learning Toolbox
 * Global Optimization Toolbox 
 * Parallel Computing Toolbox
 * MATLAB Coder
 * GPU Coder

## Brief introduction
 
1. Change the current directory to directory 'code' on MATLAB.
    
        >> cd code

2. Set the path by using the following command:

        >> setpath

3. Build MEX codes if you have MATLAB Coder.

        >> mybuild

4. Several example codes are found under the second layer directory 
   'examples' of this package. Change the current directory to one under 
   the second layer directory 'examples' and executes an M-file of 
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

* Yasas GODAGE, Eisuke KOBAYASHI, Shogo MURAMATSU, “Locally-Structured Unitary Network,“ APSIPA Transactions on Signal and Information Processing, https://nowpublishers.com/article/Details/SIP-2024-0008, May 2024
* Yasas GODAGE, Shogo MURAMATSU: Tangent Space Sampling of Video Sequence with Locally Structured Unitary Network, Proc. of 2023 IEEE International Conference on Visual Communications and Image Processing, DOI:10.1109/VCIP59821.2023.10402611, Dec. 2023
 
## Acknowledgement
 
This work was supported by JSPS KAKENHI Grant Number JP22H00512.
 
## Contributors

### Developpers
* Yasas GODAGE,  2022-
* Eisuke KOBAYASHI, 2022-
 
### Test contributers
* 

