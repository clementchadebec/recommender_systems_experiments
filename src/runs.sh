python3 main.py -data ml20m -model MultiVAE -annealing False -n_val_samples 30 -gpu 0 -train_batch_size 500 -n_epoches 300
python3 main.py -data ml20m -model Multi_our_VAE -K 2 -N 1 -annealing False -lrdec 3e-3 -lrenc 1e-3 -n_val_samples 30 -learnscale True -gpu 0 -train_batch_size 500 -n_epoches 300
python3 main.py -data ml20m -model Multi_our_VAE -K 3 -N 1 -annealing False -lrdec 3e-3 -lrenc 1e-3 -n_val_samples 30 -learnscale True -gpu 0 -train_batch_size 500 -n_epoches 300
python3 main.py -data ml20m -model Multi_our_VAE -K 10 -N 1 -annealing False -lrdec 3e-3 -lrenc 1e-3 -n_val_samples 30 -learnscale True -gpu 0 -train_batch_size 500 -n_epoches 300

python3 main.py -data ml20m -model MultiVAE -annealing True -n_val_samples 30 -gpu 0 -train_batch_size 500 -n_epoches 300
python3 main.py -data ml20m -model Multi_our_VAE -K 2 -N 1 -annealing True -lrdec 3e-3 -lrenc 1e-3 -n_val_samples 30 -learnscale True -gpu 0 -train_batch_size 500 -n_epoches 300
python3 main.py -data ml20m -model Multi_our_VAE -K 3 -N 1 -annealing True -lrdec 3e-3 -lrenc 1e-3 -n_val_samples 30 -learnscale True -gpu 0 -train_batch_size 500 -n_epoches 300
python3 main.py -data ml20m -model Multi_our_VAE -K 10 -N 1 -annealing True -lrdec 3e-3 -lrenc 1e-3 -n_val_samples 30 -learnscale True -gpu 0 -train_batch_size 500 -n_epoches 300


python3 main.py -data gowalla -model MultiVAE -annealing False -n_val_samples 30 -gpu 0 -train_batch_size 500 -n_epoches 300
python3 main.py -data gowalla -model Multi_our_VAE -K 2 -N 1 -annealing False -lrdec 3e-3 -lrenc 1e-3 -n_val_samples 30 -learnscale True -gpu 0 -train_batch_size 300 -n_epoches 300
python3 main.py -data gowalla -model Multi_our_VAE -K 3 -N 1 -annealing False -lrdec 3e-3 -lrenc 1e-3 -n_val_samples 30 -learnscale True -gpu 0 -train_batch_size 300 -n_epoches 300
python3 main.py -data gowalla -model Multi_our_VAE -K 10 -N 1 -annealing False -lrdec 3e-3 -lrenc 1e-3 -n_val_samples 30 -learnscale True -gpu 0 -train_batch_size 300 -n_epoches 300

python3 main.py -data gowalla -model MultiVAE -annealing True -n_val_samples 30 -gpu 0 -train_batch_size 500 -n_epoches 300
python3 main.py -data gowalla -model Multi_our_VAE -K 2 -N 1 -annealing True -lrdec 3e-3 -lrenc 1e-3 -n_val_samples 30 -learnscale True -gpu 0 -train_batch_size 300 -n_epoches 300
python3 main.py -data gowalla -model Multi_our_VAE -K 3 -N 1 -annealing True -lrdec 3e-3 -lrenc 1e-3 -n_val_samples 30 -learnscale True -gpu 0 -train_batch_size 300 -n_epoches 300
python3 main.py -data gowalla -model Multi_our_VAE -K 10 -N 1 -annealing True -lrdec 3e-3 -lrenc 1e-3 -n_val_samples 30 -learnscale True -gpu 0 -train_batch_size 300 -n_epoches 300


python3 main.py -data foursquare -model MultiVAE -annealing True -n_val_samples 30 -gpu 0 -train_batch_size 500 -n_epoches 300
python3 main.py -data foursquare -model Multi_our_VAE -K 2 -N 1 -annealing True -lrdec 3e-3 -lrenc 1e-3 -n_val_samples 30 -learnscale True -gpu 0 -train_batch_size 300 -n_epoches 300
python3 main.py -data foursquare -model Multi_our_VAE -K 3 -N 1 -annealing True -lrdec 3e-3 -lrenc 1e-3 -n_val_samples 30 -learnscale True -gpu 0 -train_batch_size 300 -n_epoches 300
python3 main.py -data foursquare -model Multi_our_VAE -K 10 -N 1 -annealing True -lrdec 3e-3 -lrenc 1e-3 -n_val_samples 30 -learnscale True -gpu 0 -train_batch_size 300 -n_epoches 300

python3 main.py -data foursquare -model MultiVAE -annealing False -n_val_samples 30 -gpu 0 -train_batch_size 500 -n_epoches 300
python3 main.py -data foursquare -model Multi_our_VAE -K 2 -N 1 -annealing False -lrdec 3e-3 -lrenc 1e-3 -n_val_samples 30 -learnscale True -gpu 0 -train_batch_size 300 -n_epoches 300
python3 main.py -data foursquare -model Multi_our_VAE -K 3 -N 1 -annealing False -lrdec 3e-3 -lrenc 1e-3 -n_val_samples 30 -learnscale True -gpu 0 -train_batch_size 300 -n_epoches 300
python3 main.py -data foursquare -model Multi_our_VAE -K 10 -N 1 -annealing False -lrdec 3e-3 -lrenc 1e-3 -n_val_samples 30 -learnscale True -gpu 0 -train_batch_size 300 -n_epoches 300

python3 main.py -K 20 -N 3 -model Rezende -data Rezende -gpu 1 -learnscale True -learntransitions False
