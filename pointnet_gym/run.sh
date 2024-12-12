mkdir data

curl -L -o ./data/3dmnist.zip https://www.kaggle.com/api/v1/datasets/download/daavoo/3d-mnist

unzip ./data/3dmnist.zip -d ./data

## param.zip is larger than 50M
# if [ -d ./param ]; then
#     :
# else
#     mkdir param
#     unzip param.zip -d param/
# fi