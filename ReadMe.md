# AutoMS  ![GitHub](https://img.shields.io/github/license/hcji/AutoMS)

<div align="center">
<img src="https://github.com/hcji/AutoMS/blob/main/TOC.png" width="70%">
</div>

AutoMS is a peak picking and quality estimation tool for LC-MS data processing, which utilizes a denoising autoencoder (DAE) to find the common characteristics of chromatographic peaks, predict the noise-free peaks from input ROIs, and evaluate peak qualities. AutoMS integrates HPIC for ROI extraction in order to accept raw data directly and output quantitative results. It also supports peak lists obtained from other tools with little adjustment.


## Depends:

  [Anaconda](https://www.anaconda.com/)
  

## Installation

  1. Create conda environment and activate:
          
                conda create -n automs python=3.8
                conda activate automs
  
  2. Clone the repository and enter:

                git clone https://github.com/hcji/AutoMS.git
                cd ./AutoMS
                
  3. Install dependency:

                pip install -r requirements.txt
                
  4. Run python

                python


## Usage (with HPIC)

Please download the [demo data](https://figshare.com/articles/dataset/Dataset_of_AutoMS/20499687), unzip to the *data* folder and run the following codes.  

        from AutoMS.automs import AutoMS

        file = 'data/600mix_pos.mzML'
        peaks = AutoMS(file, min_intensity = 5000)
        

## Usage (with XCMS/MZMine)

Please download the [demo data](https://figshare.com/articles/dataset/Dataset_of_AutoMS/20499687), unzip to the *data* folder and run the following codes.
        
        # need install rpy2: pip install rpy2,
        # and set R_HOME first (R version >= 3.4.1 and <= 4.1.1).
        
        import pandas as pd
        from AutoMS.automs_external import AutoMS_External
        
        file = 'data/600mix_pos.mzML'
        
        ## Need reset the column name of xcms/mzmine output refer to demo file.
        peaks = pd.read_csv('data/xcms_mzmine_input_demo.csv')
        peaks = AutoMS_External(file, peaks)


## Paper Reproduce

All the results in the paper can be reproduced by the scripts in the *notebook* dir. 
Please download the dataset in the [figshare](https://figshare.com/articles/dataset/Dataset_of_AutoMS/20499687),
and unzip them to the *data* dir. 


## Contact

Ji Hongchao   
E-mail: ji.hongchao@foxmail.com    
<div itemscope itemtype="https://schema.org/Person"><a itemprop="sameAs" content="https://orcid.org/0000-0002-7364-0741" href="https://orcid.org/0000-0002-7364-0741" target="orcid.widget" rel="me noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon">https://orcid.org/0000-0002-7364-0741</a></div>
    
WeChat public account: Chemocoder    
<img align="center" src="https://github.com/hcji/hcji/blob/main/img/qrcode.jpg" width="30%"/>
