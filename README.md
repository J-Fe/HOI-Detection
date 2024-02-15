# HOI-Detection
Our code and Doc. are based on Space-Time Interaction Graph Parsing Networks for Human-Object Interaction Recognition，ACM MM'21
### Installation
1. Clone this repository.   
    ```
    git clone https://github.com/GuangmingZhu/STIGPN](https://github.com/J-Fe/HOI-Detection.git
    ```
  
2. Install Python dependencies:   
    ```
    pip install -r requirements.txt
    ```
### Prepare Data
1. Follow [here](https://github.com/praneeth11009/LIGHTEN-Learning-Interactions-with-Graphs-and-Hierarchical-TEmporal-Networks-for-HOI) to prepare the original data of CAD120 dataset in `CAD120/datasets` folder.
2. You can also download the data we have processed directly from [here](https://drive.google.com/drive/folders/1ntHUZO8CBHV6-Wwci6XqcaQdK4EJUZJi?usp=sharing).
3. We also provide some checkpoints to the trained models. Download them [here](https://drive.google.com/drive/folders/1ntHUZO8CBHV6-Wwci6XqcaQdK4EJUZJi?usp=sharing) and put them in the checkpoints folder
### Training
For the CAD120 dataset:
    ```
        python train_CAD120.py --model VisualModelV
    ```
    ```
        python train_CAD120.py --model SemanticModelV 
    ```

### Testing
For the CAD120 dataset:
    ```
        python eval_CAD120.py
    ```
