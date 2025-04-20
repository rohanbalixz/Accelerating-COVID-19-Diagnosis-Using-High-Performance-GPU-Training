# COVID‑19 X‑Ray Classification: Comparative GPU Performance Study

## Project Overview  
This project investigates how different GPU configurations affect the training speed, resource utilization, and classification accuracy of a convolutional neural network (CNN) on a COVID‑19 chest X‑ray dataset. Our goal is to identify optimal hardware settings for rapid, reliable model development in medical imaging.

## Key Objectives  
- **Accelerate Training**: Leverage GPU acceleration to reduce end‑to‑end training time compared to CPU-only runs.  
- **Measure Efficiency**: Track GPU memory usage, throughput, and convergence speed across configurations.  
- **Maintain Accuracy**: Ensure that speed gains do not come at the expense of classification performance (accuracy, precision, recall).

## Repository Structure  
- **data/**  
  - *raw/*: Original COVID‑19 and Normal X‑ray images  
  - *preprocessed/*: Images after resizing, normalization, and augmentation  
- **scripts/**  
  - *preprocess_data.py*: Data cleaning and augmentation routines  
  - *train_model.py*: CNN training with configurable GPU selection and hyperparameters  
  - *evaluate_model.py*: Computation of test‑set metrics (accuracy, precision, recall)  
  - *animate_training.py*: Side‑by‑side convergence visualization for multiple GPUs  
- **results/**  
  - *logs/*: Per‑experiment timing and GPU utilization logs  
  - *models/*: Saved model checkpoints for each GPU setup  
  - *plots/*: Charts comparing training time vs. accuracy  
  - *animation.gif*: Consolidated training convergence animation  
- **gpu_configs.yaml**: Defines GPU names, batch sizes, and learning rates under test  
- **requirements.txt**: Python package dependencies  

## Prerequisites  
- Python 3.8 or higher  
- CUDA‑capable GPU drivers installed for target devices  
- Virtual environment or container with packages from requirements.txt  

## Workflow  
1. **Preprocess the Images**  
   Use the provided script to load raw X‑ray images, resize them, apply normalization and augmentations, and save them in a structured directory for training and testing.  
2. **Train the CNN**  
   Execute the training module with your GPU of choice, specifying desired epochs and batch size in the configuration file. Each run produces logs and a model checkpoint.  
3. **Evaluate Performance**  
   Run the evaluation script on the test split to compute accuracy, precision, and recall for each trained model.  
4. **Visualize Convergence**  
   Generate an animated comparison of training loss and accuracy curves across all GPU configurations to highlight differences in convergence speed.

## Results  
- **Performance Plots**: Visual comparisons of training time, GPU memory usage, and test accuracy in `results/plots/`.  
- **Convergence Animation**: A side‑by‑side GIF in `results/animation.gif` showing real‑time training behavior on each GPU.

## Contributing  
We welcome improvements! Please open an issue or submit a pull request to refine GPU configurations, add new metrics, or enhance visualization.

## License  
This project is released under the MIT License. See the included LICENSE file for details.

## Author  
**Rohan Bali** (@rohanbalixz)  
