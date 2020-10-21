### Scalable and Robust Tensor Decomposition of Spontaneous Stereotactic EEG data 
This is a Python implementation of the Scalable and Robust Sequential Canonical Polyadic Decomposition (SRSCPD) framework presented in 

- J. Li, J. P. Haldar, J. C. Mosher, D. R. Nair, J. Gonzalez-Martinez, R. M. Leahy,
"Scalable and robust tensor decomposition of spontaneous stereotactic EEG data",
IEEE Transactions on Biomedical Engineering, vol. 66, no. 6, pp. 1549–1558, 2019.
https://doi.org/10.1109/TBME.2018.2875467

- J. Li, J. C. Mosher, D. R. Nair, J. Gonzalez-Martinez, R. M. Leahy,
"Robust tensor decomposition of resting brain networks in stereotactic EEG",
IEEE 51st Asilomar Conference on Signals, Systems and Computers, Pacific Grove, CA, Oct. 2017, pp. 1544–1548.
https://doi.org/10.1109/ACSSC.2017.8335616

Please cite these two papers if you use this code and/or its derivatives in your own work.

#### Notes
- Regularization yet to be implemented (PR welcome). Subproblems are simply least squares problem with no regularization, so matrix left inverse is used instead of TFOCS solver (which currently has no Python binding and has a large overhead to transfer data between Python and MATLAB) as in [ref [1]](https://doi.org/10.1109/TBME.2018.2875467). The demo code works on the simulated data as in *Sec III.B Simulation* in [ref [1]](https://doi.org/10.1109/TBME.2018.2875467).
- The official project page with a MATLAB implementation can be found on the first author's website [here](https://silencer1127.github.io/software/SRSCPD_ALS/srscpd_als_main)

#### TODO
- Stem plot for channel mode. Current ```matplotlib.pyplot.stem``` doesn't support multichannel data, so use ```pyplot.plot``` instead.

#### Main Dependencies

- python 3.6
- numpy 1.15.1
- scipy 1.1.0
- h5py 2.10.0
- matplotlib 3.3.1
- os
- copy
- math
- unittest (for module test purposes)


#### Run
Navigate to the code folder and do ```python demo.py```


#### Disclaimer
IN NO EVENT SHALL THE AUTHORS, THE CONTRIBUTORS, THE DISTRIBUTORS AND THE UNIVERSITY OF SOUTHERN CALIFORNIA (“AUTHORS”) BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS CODE, EVEN IF THE AUTHORS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. THE AUTHORS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. FOR RESEARCH PURPOSE ONLY. THIS CODE IS PROVIDED ON A “AS IS” BASIS AND THE AUTHORS HAVE NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
