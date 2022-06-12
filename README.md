# lip-reading
## Lip Reading using Machine Learning || Final Year Project - Bachelor's Degree in Applied Statistics

This project is for my Bachelor's thesis (Applied Statistics). We are using a lip reading dataset, each instance is a speaker uttering a phrase or a word,
the dataset used is https://sites.google.com/site/achrafbenhamadou/-datasets/miracl-vc1. 

### description of the scripts and files
tfg.pdf is my thesis document in spanish

get_lips.py is the script that crops the lips region and saves it

end_to_end_nn_phrases.py is the script that takes the cropped images and implement an end-to-end neural network (3DCONV + LSTM), phrases in the title means that uses the dataset of phrases of the link above, meanwhile words means that it uses the dataset with words

hmm_phrases.py pca + hidden markov model

svm_phrases.py pca + svm
