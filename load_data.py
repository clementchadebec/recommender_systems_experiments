import os
import urllib.request

if True: #not os.path.exists('./data/ml20m/train.csv'):
    print('Downloading data... \n')
    with open("./data_url.txt") as file:
        url = file.read()
        urllib.request.urlretrieve(url, './data.zip')
os.remove("./data_url.txt")

if True:#not os.path.exists('./data/models/best_model_Multi_our_VAE_data_foursquare_K_10_N_1_learnreverse_False_anneal_True_lrdec_0.003_lrenc_0.001_learntransitions_False_initstepsize_0.005_learnscale_True.pt'):
    print('Downloading models... \n')
    with open("./models_url.txt") as file:
        url = file.read()
        urllib.request.urlretrieve(url, './models.zip')
os.remove("./models_url.txt")
print('Success! \n')