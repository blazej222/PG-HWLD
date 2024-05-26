Write-Output "Started Testing"
../resources/python/venv/Scripts/activate.ps1
cd ../models/vgg

Write-Output "VGG end result run1 ourset"
python main.py --test --test_path ../../resources/datasets/dataset-processed/test-images --rotate_images --saved_model_path ./saved_models/run1
Write-Output "`nVGG end result run2 ourset"
python main.py --test --test_path ../../resources/datasets/dataset-processed/test-images --rotate_images --saved_model_path ./saved_models/run2
Write-Output "`nVGG end result run3 ourset"
python main.py --test --test_path ../../resources/datasets/dataset-processed/test-images --rotate_images --saved_model_path ./saved_models/run3

Write-Output "`nVGG end result run1 emnist"
python main.py --test --saved_model_path ./saved_models/run1
Write-Output "`nVGG end result run2 emnist"
python main.py --test --saved_model_path ./saved_models/run2
Write-Output "`nVGG end result run3 emnist"
python main.py --test --saved_model_path ./saved_models/run3

Write-Output "`nVGG best result run1 ourset"
python main.py --test --test_path ../../resources/datasets/dataset-processed/test-images --rotate_images --saved_model_path ./saved_models/run1 --model1_filename best_model1.pth --model2_filename best_model2.pth
Write-Output "`nVGG best result run2 ourset"
python main.py --test --test_path ../../resources/datasets/dataset-processed/test-images --rotate_images --saved_model_path ./saved_models/run2 --model1_filename best_model1.pth --model2_filename best_model2.pth
Write-Output "`nVGG best result run3 ourset"
python main.py --test --test_path ../../resources/datasets/dataset-processed/test-images --rotate_images --saved_model_path ./saved_models/run3 --model1_filename best_model1.pth --model2_filename best_model2.pth

Write-Output "`nVGG best result run1 emnist"
python main.py --test --saved_model_path ./saved_models/run1 --model1_filename best_model1.pth --model2_filename best_model2.pth
Write-Output "`nVGG best result run2 emnist"
python main.py --test --saved_model_path ./saved_models/run2 --model1_filename best_model1.pth --model2_filename best_model2.pth
Write-Output "`nVGG best result run3 emnist"
python main.py --test --saved_model_path ./saved_models/run3 --model1_filename best_model1.pth --model2_filename best_model2.pth

cd ../..
deactivate
resources/python/venv-3.6.8/Scripts/activate.ps1
cd models/TextCaps

Write-Output "`nTextcaps end result run1 ourset"
python textcaps_emnist_bal.py --test --weights ./saved_models/emnist_letters_5600_run1/trained_model.h5 --train_path ../../resources/datasets/dataset-EMNIST-mat/emnist-letters.mat --test_path ../../resources/datasets/packed/dataset-processed/dataset-processed.mat
Write-Output "`nTextcaps end result run2 ourset"
python textcaps_emnist_bal.py --test --weights ./saved_models/emnist_letters_5600_run2/trained_model.h5 --train_path ../../resources/datasets/dataset-EMNIST-mat/emnist-letters.mat --test_path ../../resources/datasets/packed/dataset-processed/dataset-processed.mat
Write-Output "`nTextcaps end result run3 ourset"
python textcaps_emnist_bal.py --test --weights ./saved_models/emnist_letters_5600_run3/trained_model.h5 --train_path ../../resources/datasets/dataset-EMNIST-mat/emnist-letters.mat --test_path ../../resources/datasets/packed/dataset-processed/dataset-processed.mat

Write-Output "`nTextcaps end result run1 emnist"
python textcaps_emnist_bal.py --test --weights ./saved_models/emnist_letters_5600_run1/trained_model.h5 --train_path ../../resources/datasets/dataset-EMNIST-mat/emnist-letters.mat --test_path ../../resources/datasets/dataset-EMNIST-mat/emnist-letters.mat
Write-Output "`nTextcaps end result run2 emnist"
python textcaps_emnist_bal.py --test --weights ./saved_models/emnist_letters_5600_run2/trained_model.h5 --train_path ../../resources/datasets/dataset-EMNIST-mat/emnist-letters.mat --test_path ../../resources/datasets/dataset-EMNIST-mat/emnist-letters.mat
Write-Output "`nTextcaps end result run3 emnist"
python textcaps_emnist_bal.py --test --weights ./saved_models/emnist_letters_5600_run3/trained_model.h5 --train_path ../../resources/datasets/dataset-EMNIST-mat/emnist-letters.mat --test_path ../../resources/datasets/dataset-EMNIST-mat/emnist-letters.mat
deactivate
cd ../..