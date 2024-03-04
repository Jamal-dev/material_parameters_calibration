import torch
import numpy as np
from src.preprocess import pipeline_preprocess
from pathlib import Path
from src.prepare_features import creating_invariances
from src.prepare_features import MyDataset
from src.utility import split_train_test
from src.model_pre import CustomModel
import re
from src.utility import validate_expression, modify_expression
from torch import optim
from torch import nn
from src.train import train, cal_loss
import matplotlib.pyplot as plt
from src.visualize import Visualization,plot_losses

def create_model(expression:str):
    params = re.findall(r'c\d+', expression)
    try:
        validate_expression(expression)
        model = CustomModel(modify_expression(expression), params)
        return model
    except ValueError as e:
        model = None
        print(e)
        return model
    except Exception as e:
        model = None
        print(f"Model is not created due to an error: {e}")
        return model

def main(filepath:Path,
         fiber_direction:np.ndarray,
         train_size:float=0.88,
         val_size:float=0.91,
         batch_size:int=128,
         expression:str = "c1 * (J-1)**2 + c2 * (I1bar-3) + c3 * (I1bar-3)**2 + c4 * (I2bar-3) + c5 * (I2bar-3)**2",
         lr:float=0.001,
         num_epochs:int=1000,
         print_every:int=100,
         patience:int=10000,
         save_models:bool=True,
         plot_losses_bool:bool=True,
         plot_stresses:bool=True,
         plot_psi:bool=True,
         update_ui = None,
         plot_losses_singal = None,
         visualize_data_signal = None,
         model_data_signal = None,):
    
    strain_lin, sigMatrixList, indices_dict = pipeline_preprocess(filepath)
    
    sigMatrixList = [torch.tensor(sigMatrixList[i],dtype=torch.float32) for i in range(len(sigMatrixList))]
    v_matrix,b_mat_lin,invarices,invarices_bar,bases_data_list = creating_invariances(strain_lin,fiber_direction)
    data_dict = {'invriants': invarices_bar, 'bases': bases_data_list, 'target_stress': sigMatrixList}
    dataset = MyDataset(data_dict)
    train_dataset, val_dataset, test_dataset = split_train_test(dataset, 
                                                                train_size, 
                                                                val_size)
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=batch_size,
                                                shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=batch_size,
                                                shuffle=False)
    model = create_model(expression)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    input_size = 5
    output_size = 1
    train_losses, val_losses, early_stop = train(model, 
                                            train_loader, 
                                            val_loader, 
                                            optimizer, 
                                            criterion, 
                                            num_epochs, 
                                            print_every=print_every, 
                                            save_models=save_models, 
                                            patience=patience,
                                            update_ui = update_ui)
    
    test_loss = cal_loss(model, test_loader, criterion)
    update_ui(f'Test Loss: {test_loss}')
    if plot_losses_bool:
        # plot_losses(train_losses, val_losses)
        plot_losses_singal(train_losses, val_losses)
    
    if plot_stresses or plot_psi:
        visualize_data_signal(data_dict, indices_dict)
        model_data_signal(model, strain_lin)

    # visualizer = Visualization(data_dict, indices_dict)
    # if plot_stresses:
    #     visualizer.plot_stresses(model,strain_lin)
    # if plot_psi:
    #     visualizer.plot_psi(model,strain_lin)
    
    return model

if __name__ == "__main__":
    
    expression:str = "c1 * (J-1)**2 + c2 * (I1bar-3) + c3 * (I1bar-3)**2 + c4 * (I2bar-3) + c5 * (I2bar-3)**2"
    fiber_direction = np.array([1,0,0]).reshape(3,1)
    lr:float=0.001,
    num_epochs:int=100
    print_every:int=100
    file_path = Path("data/avgStrainStressInfoNEPureShearLoading.csv")
    main(file_path,
            fiber_direction=fiber_direction, 
            expression=expression, 
            lr=lr, 
            num_epochs=num_epochs, 
            print_every=print_every)