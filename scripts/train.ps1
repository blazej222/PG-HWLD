# Train VGG on default dataset thrice

../resources/python/venv/Scripts/activate.ps1
cd ../models/vgg
python main.py --saved_model_path ./saved_models/run1
python main.py --saved_model_path ./saved_models/run2
python main.py --saved_model_path ./saved_models/run3
cd ../..
deactivate

# Train TextCaps on default dataset thrice

resources/python/venv-3.6.8/Scripts/activate.ps1
cd models/TextCaps
python textcaps_emnist_bal.py --cnt 5600 --save_dir ./saved_models/emnist_letters_5600_run1 --train_path ../../resources/datasets/dataset-EMNIST-mat/emnist-letters.mat --test_path ../../resources/datasets/dataset-EMNIST-mat/emnist-letters.mat
python textcaps_emnist_bal.py --cnt 5600 --save_dir ./saved_models/emnist_letters_5600_run2 --train_path ../../resources/datasets/dataset-EMNIST-mat/emnist-letters.mat --test_path ../../resources/datasets/dataset-EMNIST-mat/emnist-letters.mat
python textcaps_emnist_bal.py --cnt 5600 --save_dir ./saved_models/emnist_letters_5600_run3 --train_path ../../resources/datasets/dataset-EMNIST-mat/emnist-letters.mat --test_path ../../resources/datasets/dataset-EMNIST-mat/emnist-letters.mat