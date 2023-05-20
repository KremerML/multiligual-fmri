import pandas as pd
from nilearn import image 
import os

def fmri2words(brain, text_data, section, delay = 6) :
  chunks = []
  text = text_data[text_data['section'] == section]
  for (tr,b) in enumerate(image.iter_img(brain)) :
    onset = tr*2-delay
    # if onset < 0 :
    #   continue
    offset = onset + 2
    #print(onset)
    #print(text[(text['onset']>= onset) & (text['offset']< offset)])
    
    chunk_data = text[(text['onset']>= onset) & (text['offset']< offset)]

    chunks.append(list(chunk_data['word']))
  return chunks

def main() :
    fmri_file_1 = 'D:/Le-Petite-Prince-fMRI-dataset/derivatives/sub-EN057/func/sub-EN057_task-lppEN_run-15_space-MNIColin27_desc-preproc_bold.nii.gz'
    info = "D:/Le-Petite-Prince-fMRI-dataset/annotation/EN/lppEN_word_information.csv"

    pwd = os.getcwd()
    os.chdir(os.path.dirname(fmri_file_1))
    nii_img_1  = image.load_img(fmri_file_1)

    os.chdir(os.path.dirname(info))
    en_words = pd.read_csv(info, index_col=[0])
    
    os.chdir(pwd)

    words = fmri2words(nii_img_1, en_words, 1, 0)
    print(words)

if __name__ == '__main__' :
  main()