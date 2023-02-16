#python3 main_pythae.py -data gowalla -model MultiVAMP -annealing False -n_val_samples 30 -gpu 0
#-train_batch_size 500 -n_epoches 300
python3 main_pythae.py -data gowalla -model MultiCIWAE -annealing False -n_val_samples 30 -gpu 0 -train_batch_size 500 -n_epoches 100
python3 main_pythae.py -data gowalla -model MultiCIWAE -annealing False -n_val_samples 30 -gpu 0 -train_batch_size 500 -n_epoches 100 -beta 1.
python3 main_pythae.py -data gowalla -model MultiCIWAE -annealing False -n_val_samples 30 -gpu 0 -train_batch_size 500 -n_epoches 100 -beta 0.2
python3 main_pythae.py -data gowalla -model MultiVAELinNF -annealing False -n_val_samples 30 -gpu 0 -train_batch_size 500 -n_epoches 100
python3 main_pythae.py -data gowalla -model MultiVAEIAF -annealing False -n_val_samples 30 -gpu 0 -train_batch_size 500 -n_epoches 100 -n_made_blocks 3
python3 main_pythae.py -data gowalla -model MultiVAEIAF -annealing False -n_val_samples 30 -gpu 0 -train_batch_size 500 -n_epoches 100 -n_made_blocks 5
python3 main_pythae.py -data gowalla -model MultiVAEIAF -annealing False -n_val_samples 30 -gpu 0 -train_batch_size 500 -n_epoches 100 -n_made_blocks 2
python3 main_pythae.py -data gowalla -model MultiVAE -annealing False -n_val_samples 30 -gpu 0 -train_batch_size 500 -n_epoches 100


#python3 main_pythae.py -data foursquare -model MultiVAMP -annealing False -n_val_samples 30 -gpu 0 -train_batch_size 500 -n_epoches 100
python3 main_pythae.py -data foursquare -model MultiCIWAE -annealing False -n_val_samples 30 -gpu 0 -train_batch_size 500 -n_epoches 100
python3 main_pythae.py -data foursquare -model MultiCIWAE -annealing False -n_val_samples 30 -gpu 0 -train_batch_size 500 -n_epoches 100 -beta 1.
python3 main_pythae.py -data foursquare -model MultiCIWAE -annealing False -n_val_samples 30 -gpu 0 -train_batch_size 500 -n_epoches 100 -beta 0.2
python3 main_pythae.py -data foursquare -model MultiVAELinNF -annealing False -n_val_samples 30 -gpu 0 -train_batch_size 500 -n_epoches 100
python3 main_pythae.py -data foursquare -model MultiVAEIAF -annealing False -n_val_samples 30 -gpu 0 -train_batch_size 500 -n_epoches 100 -n_made_blocks 3
python3 main_pythae.py -data foursquare -model MultiVAEIAF -annealing False -n_val_samples 30 -gpu 0 -train_batch_size 500 -n_epoches 100 -n_made_blocks 5
python3 main_pythae.py -data foursquare -model MultiVAEIAF -annealing False -n_val_samples 30 -gpu 0 -train_batch_size 500 -n_epoches 100 -n_made_blocks 2
python3 main_pythae.py -data foursquare -model MultiVAE -annealing False -n_val_samples 30 -gpu 0 -train_batch_size 500 -n_epoches 100
