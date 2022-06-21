# AutoMS

<div align="center">
<img src="https://github.com/hcji/AutoMS/blob/master/AutoMS.png" width="50%">
</div>

AutoMS is a peak picking and quality estimation tool for LC-MS data processing, which utilizes a denoising autoencoder (DAE) to find the common characteristics of chromatographic peaks, and predict the noise-free peaks from input ROIs. AutoMS integrates HPIC for ROI extraction in order to accept raw data directly and output quantitative results. It also supports peak lists obtained from other tools with little adjustment.


## Depends:

  1. [Anaconda](https://www.anaconda.com/) >= 3.8
  2. [Tensorflow](https://www.tensorflow.org/) >= 2.9.1
  3. [pymzml](https://pymzml.readthedocs.io/en/latest/)


## Installation

  1. Clone the repository and enter:

                git clone https://github.com/hcji/AutoMS.git
                cd ./AutoMS

  2. Run python

                 python


## Usage (with HPIC)

        from AutoMS import hpic
        from AutoMS import peakeval

        file = 'data/600mix_pos.mzML'
        peaks = AutoMS(file, min_intensity = 5000)
        

## Usage (with XCMS/MZMine)

  Coming soon ...


## Contact

Ji Hongchao   
E-mail: ji.hongchao@foxmail.com    
<div itemscope itemtype="https://schema.org/Person"><a itemprop="sameAs" content="https://orcid.org/0000-0002-7364-0741" href="https://orcid.org/0000-0002-7364-0741" target="orcid.widget" rel="me noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon">https://orcid.org/0000-0002-7364-0741</a></div>
    
WeChat public account: Chemocoder    
<img align="center" src="https://github.com/hcji/hcji/blob/main/img/qrcode.jpg" width="20%"/>
