import os
import sys
from os.path import join
import pickle as pkl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split
from neuro_data_analysis.neural_data_lib import load_neural_data, load_img_resp_pairs, load_latent_codes

def train_linear_model(df_img_resp_latent, test_size=0.2, random_state=42, alphas=[0.01, 0.1, 1, 10], projection_matrix=None, X_key="latent_code", y_key="resp"):
    """
    Train a Ridge regression model to predict responses from latent codes.

    Args:
        df_img_resp_latent: DataFrame containing latent codes and responses
        test_size: Fraction of data to use for testing (default 0.2)
        random_state: Random seed for train/test split (default 42)
        alphas: List of alpha values to try for Ridge regression (default [0.01, 0.1, 1, 10])
    
    Returns:
        model: Trained RidgeCV model
        train_score: R² score on training set
        test_score: R² score on test set
    """
    # Extract features (X) and target (y)
    X = np.array(df_img_resp_latent[X_key].tolist())
    if projection_matrix is not None:
        X = X @ projection_matrix
    y = df_img_resp_latent[y_key].values
    bsl = df_img_resp_latent['bsl'].values
    print(f"X.shape: {X.shape}, y.shape: {y.shape}")

    print(f"resp mean and std, min, max: {np.mean(y):.3f}, {np.std(y):.3f}, {np.min(y):.3f}, {np.max(y):.3f}")
    print(f"baseline mean and std, min, max: {np.mean(bsl):.3f}, {np.std(bsl):.3f}, {np.min(bsl):.3f}, {np.max(bsl):.3f}")
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Fit linear regression model
    model = RidgeCV(alphas=alphas)
    model.fit(X_train, y_train)
    # print final alpha
    print(f"Final alpha: {model.alpha_}")
    # Get model performance metrics
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"Linear model R² score on train set: {train_score:.3f}")
    print(f"Linear model R² score on test set: {test_score:.3f}")
    
    return {"model": model, "train_score": train_score, "test_score": test_score, 
            "X_key": X_key, "X_dim": X.shape[1], "sample_num": y.shape[0], "alpha": model.alpha_}



def load_and_merge_data(BFEStats, Expi, thread_id=1, stimdrive="S:", verbose=True):
    """
    Load latent codes and image response data and merge them into a single dataframe.
    
    Args:
        stim_path: Path to stimulus files
        BFEStats: BFE statistics object
        Expi: Experiment ID
        thread_id: Thread ID to load data from (default 1)
        stimdrive: Drive letter for stimulus files (default "S:")
        verbose: Whether to print debug info (default True)
        
    Returns:
        df_img_resp_latent: Merged dataframe containing image responses and latent codes
    """
    stim_path = BFEStats[Expi - 1].meta.stimuli
    stim_path = stim_path.replace("N:", "S:")
    # Load latent codes
    latent_codes_all, latent_image_ids, latent_gen_vec, _ = load_latent_codes(
        stim_path, thread_id=thread_id, verbose=verbose
    )
    # Load image response pairs
    imgfps, resp_vec, bsl_vec, gen_vec = load_img_resp_pairs(
        BFEStats, Expi, "Evol", thread=thread_id, stimdrive=stimdrive, output_fmt="vec"
    )
    # Create latent code dataframe
    df_latent = pd.DataFrame()
    df_latent["image_id"] = latent_image_ids
    df_latent["gen_vec"] = latent_gen_vec
    df_latent["latent_code"] = list(latent_codes_all)
    #TODO: potentially rename the founders to the true image ids to get more accurate results
    # if verbose:
        # print(df_latent.head())
    # Create image response dataframe

    df_img_resp = pd.DataFrame()
    df_img_resp["image_fps"] = imgfps
    df_img_resp["image_id"] = [name.split("\\")[-1] for name in imgfps]
    df_img_resp["resp"] = list(resp_vec)
    df_img_resp["bsl"] = list(bsl_vec)
    df_img_resp["gen_vec"] = gen_vec

    # Merge dataframes
    df_img_resp_latent = pd.merge(df_img_resp, df_latent, on="image_id", how="inner")
    return df_img_resp_latent


# Load and preprocess images
from torchvision.transforms import ToTensor, Resize
from PIL import Image
import torch
import tqdm

def process_images_batch_add_activations(df, caffenet, batch_size=128, layer_idx=-3):
    """Process images in batches through CaffeNet and extract fc6 activations
    
    Args:
        df: DataFrame containing image file paths in 'image_fps' column
        caffenet: CaffeNet model instance
        batch_size: Size of batches for processing (default 64)
        
    Returns:
        DataFrame with added 'caffenet_fc6_act' column containing activations
    """
    # Get list of image paths
    image_fps = df['image_fps'].tolist()
    activations = []

    # Process images in batches
    for i in tqdm.trange(0, len(image_fps), batch_size):
        batch_fps = image_fps[i:i+batch_size]
        batch_tensors = []
        # Prepare batch
        for fp in batch_fps:
            img = Image.open(fp)
            img = img.resize((224, 224), Image.BILINEAR)
            img_tensor = ToTensor()(img)
            batch_tensors.append(img_tensor)
        
        # Stack tensors into batch
        batch = torch.stack(batch_tensors).cuda()
        
        # Process batch
        with torch.no_grad():
            # Forward pass through caffenet up to relu6
            batch_acts = caffenet(batch, preproc=True, output_layer_idx=layer_idx)
            # Store flattened activation vectors
            activations.extend(batch_acts.cpu().numpy().reshape(len(batch_fps), -1))

    # Add activations as new column
    df['caffenet_fc6_act'] = activations
    return df



# load caffenet from 
from core.utils.GAN_utils import Caffenet
print("Loading caffenet...")
caffenet = Caffenet(pretrained=True)
caffenet.eval().cuda()

import sys 
if sys.platform == "linux":
    # rootdir = r"/scratch/binxu/BigGAN_Optim_Tune_new"
    # Hdir_BigGAN = r"/scratch/binxu/GAN_hessian/BigGAN/summary/H_avg_1000cls.npz"
    # Hdir_fc6 = r"/scratch/binxu/GAN_hessian/FC6GAN/summary/Evolution_Avg_Hess.npz"
    # O2 path interface
    scratchdir = "/n/scratch3/users/b/biw905"  # os.environ['SCRATCH1']
    # rootdir = join(scratchdir, "GAN_Evol_cmp")
    rootdir = join(scratchdir, "GAN_Evol_Dissection")
    Hdir_BigGAN = join("/home/biw905/Hessian", "H_avg_1000cls.npz")  #r"/scratch/binxu/GAN_hessian/BigGAN/summary/H_avg_1000cls.npz"
    Hdir_fc6 = join("/home/biw905/Hessian", "Evolution_Avg_Hess.npz")  #r"/scratch/binxu/GAN_hessian/FC6GAN/summary/Evolution_Avg_Hess.npz"
else:
    rootdir = r"F:\insilico_exps\GAN_Evol_Dissection"
    Hdir_BigGAN = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigGAN\H_avg_1000cls.npz"
    Hdir_fc6 = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\fc6GAN\Evolution_Avg_Hess.npz"

print("Loading hessian and projection matrices...")
H_data = np.load(Hdir_fc6, allow_pickle=True)
eigvals = H_data['eigv_avg']
eigvects = H_data['eigvect_avg']
assert eigvals[-1] > eigvals[-2]

proj_mat_hess_top128 = eigvects[:, -128:] # this is the top eigenspace of the hessian
proj_mat_hess_top256 = eigvects[:, -256:] # this is the top eigenspace of the hessian
proj_mat_hess_null128 = eigvects[:, :128] # this is the top eigenspace of the hessian
proj_mat_hess_null256 = eigvects[:, :256] # this is the top eigenspace of the hessian

np.random.seed(0)
proj_mat = np.random.randn(4096, 256)
proj_mat_rnd256 = np.linalg.qr(proj_mat)[0]
np.random.seed(1)
proj_mat = np.random.randn(4096, 128)
proj_mat_rnd128 = np.linalg.qr(proj_mat)[0]

# Define projection matrices and their names
projection_configs = [
    ('full', None),
    ('hess_top128', proj_mat_hess_top128),
    ('hess_top256', proj_mat_hess_top256), 
    ('hess_null128', proj_mat_hess_null128),
    ('hess_null256', proj_mat_hess_null256),
    ('rnd256', proj_mat_rnd256),
    ('rnd128', proj_mat_rnd128)
]

print("Loading neural data...")
BFEStats_merge, BFEStats = load_neural_data()
# iterate over all experiments
saveroot = "E:\OneDrive - Harvard University\BigGAN_latent_code_prediction"
for Expi in range(1, len(BFEStats) + 1):
    if BFEStats[Expi - 1]["evol"] is None:
        continue
    print(f"Processing experiment {Expi}...")
    expdir = join(saveroot, f"Both_Exp{Expi:03d}")
    os.makedirs(expdir, exist_ok=True)
    df_img_resp_latent_thr0 = load_and_merge_data(BFEStats, Expi, thread_id=0, stimdrive="S:", verbose=True)
    df_img_resp_latent_thr1 = load_and_merge_data(BFEStats, Expi, thread_id=1, stimdrive="S:", verbose=True)

    df_img_resp_latent_thr0 = process_images_batch_add_activations(df_img_resp_latent_thr0, caffenet, batch_size=128, )
    df_img_resp_latent_thr1 = process_images_batch_add_activations(df_img_resp_latent_thr1, caffenet, batch_size=128, )

    df_img_resp_latent_thr0.to_pickle(join(expdir, f"df_img_resp_latent_caffe_act_thr0_Expi{Expi}.pkl"))
    df_img_resp_latent_thr1.to_pickle(join(expdir, f"df_img_resp_latent_caffe_act_thr1_Expi{Expi}.pkl"))
    
    
    try:
        # Dictionary to store all results
        results = {
            'thread0': {},
            'thread1': {}
        }
        # Training loop for thread 0
        print("Training models for thread 0 data...")
        for proj_name, proj_matrix in projection_configs:
            print(f"- Regression on DeePSim latent codes with projection {proj_name} ...")
            results['thread0']["latent_"+proj_name] = train_linear_model(df_img_resp_latent_thr0, 
                                                                test_size=0.2, random_state=42, alphas=np.logspace(-3, 12, 100), 
                                                                projection_matrix=proj_matrix)

        for proj_name, proj_matrix in projection_configs:
            print(f"- Regression on DeePSim Caffenet activations with projection {proj_name} ...")
            results['thread0']["caffenet_fc6_act_"+proj_name] = train_linear_model(df_img_resp_latent_thr0, 
                                                                test_size=0.2, random_state=42, alphas=np.logspace(-3, 12, 100), 
                                                                projection_matrix=proj_matrix, X_key="caffenet_fc6_act")

        # Training loop for thread 1
        print("\nTraining models for thread 1 data...")
        print("- Training linear model on latent_code...")
        results['thread1']["latent_"+'full'] = train_linear_model(df_img_resp_latent_thr1, 
                                                                test_size=0.2, random_state=42, alphas=np.logspace(-3, 12, 100))
        # Then train all models with X_key="caffenet_fc6_act"
        for proj_name, proj_matrix in projection_configs[1:]:  # Skip 'full' as we already did it
            print(f"- Training {proj_name} model on caffenet_fc6_act...")
            results['thread1']["caffenet_fc6_act_"+proj_name] = train_linear_model(df_img_resp_latent_thr1, 
                                                                test_size=0.2, random_state=42, alphas=np.logspace(-3, 12, 100), 
                                                                X_key="caffenet_fc6_act", projection_matrix=proj_matrix )

        print("\nTraining complete! Results stored in 'results' dictionary")
        print("\nTest scores summary:")
        for thread in ['thread0', 'thread1']:
            print(f"\n{thread} results:")
            for proj_name in results[thread]:
                print(f"{proj_name}: {results[thread][proj_name]['test_score']:.4f}")
                
        # save the results
        with open(join(expdir, f"latent_code_linear_neural_pred_results_Expi{Expi}.pkl"), "wb") as f:
            pkl.dump(results, f)
    except Exception as e:
        print(f"Error in regression experiment {Expi}: {e}")
        continue
        