# Tennis-Shot-trajectory-modelling-GAIL

### Projectory structure

```
Tennis-Shot-trajectory-modelling-GALL/
|-- config                          #contain training and inferencing config file
├── dataset/
│   ├── data_prep.py                # TennisTrajectoryDataset class
│   └── dataloader.py               # DataLoader configuration
├── notebooks/                      
├── resources/                      
|-- dataset_collection_preparation  # dataset collection and preparation with human-in-loop
├── src/
│   └── trpo.py                     # TRPO algorithm implementation
├── utils/
│   ├── env.py                      # TennisEnvironment class
│   └── constants.py                # Physics constants
├── model.py                        # PolicyNetwork and Discriminator
├── plots.py                        # plot_discriminator_loss function
├── main.py                         # Training and evaluation loop and config
├── requirements.txt
|-- infer.py                        # inference code
|-- ball_data.csv 
└── README.md
```


### Usage

To create virtual env to work with:


```
conda create -n tennis_gail310 python=3.10
pip install -r requirements.txt
```


To **train** the model with default configurations,
Run
```
python main.py
```

To **infer** on trained model:
 ```
 python infer.py
 ```

 ### Dataset:
 [Training:validation:testing data](ball_data.csv)


 ## Model weights:
 [best model weights](https://drive.google.com/drive/folders/10Pmu1nzGWAoue5W1C-eEd5q11JWEs37m?usp=sharing)