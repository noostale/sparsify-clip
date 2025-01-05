#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Subset
from typing import List, Dict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import random
import wandb
import datetime
import open_clip
import math
import umap


# In[2]:


def contrastive_loss(image_embeds, text_embeds, temperature=0.07):
    """
    image_embeds: (batch_size, embed_dim)
    text_embeds: (batch_size, embed_dim)
    temperature: scalar float for scaling similarities
    returns: scalar loss (contrastive)
    """
    
    # Similarity matrix, shape (bs, bs)
    logits = image_embeds @ text_embeds.t()
    logits = logits / temperature

    # Targets are just the diagonal (i.e. 0->0, 1->1, ...)
    batch_size = image_embeds.size(0)
    target = torch.arange(batch_size, device=logits.device)

    # CE loss for image->text
    loss_i2t = F.cross_entropy(logits, target)
    # CE loss for text->image
    loss_t2i = F.cross_entropy(logits.t(), target)

    # Average the two directions
    return (loss_i2t + loss_t2i) / 2

def lunif_loss(x, t=2):
    # Compute pairwise distances between all embeddings
    sq_pdist = torch.pdist(x, p=2).pow(2)
    
    # Apply the uniformity loss formula
    return sq_pdist.mul(-t).exp().mean().log()


# In[3]:


def visualize_embeddings(text_embeddings, vision_embeddings, 
                         sample_size=1000, method='pca', 
                         title="Embeddings Visualization",
                         save_path=None):
    """
    Visualizes text and vision embeddings in 2D or 3D using PCA, t-SNE, or UMAP.

    Args:
        text_embeddings (torch.Tensor): 
            Shape [N, D] containing text embeddings.
        vision_embeddings (torch.Tensor):
            Shape [N, D] containing vision/image embeddings.
        sample_size (int): 
            If the embeddings contain more than 'sample_size' samples, 
            randomly pick this many for faster plotting. Set -1 to use all.
        method (str): 
            "pca", "tsne", or "umap".
        title (str): 
            Title for the plot.
        save_path (str, optional): 
            If provided, saves the plot to this path instead of showing it.
    """
    # Detach from graph and bring to CPU if the tensors require grad
    text_np = text_embeddings.detach().cpu().numpy()
    vision_np = vision_embeddings.detach().cpu().numpy()

    # Optionally downsample for quicker plotting
    if sample_size != -1:
        n_text = text_np.shape[0]
        n_vision = vision_np.shape[0]

        n_samples = min(n_text, n_vision)

        if n_samples > sample_size:
            indices = np.random.choice(n_samples, size=sample_size, replace=False)
            text_np = text_np[indices]
            vision_np = vision_np[indices]

    # Combine for joint dimensionality reduction
    all_data = np.concatenate([text_np, vision_np], axis=0)
    
    
    # SHOULD BE NORMALIZED IN THE EVALUATE FUNCTION
    
    #NORMALIZATION
    #norms = np.linalg.norm(all_data, axis=1, keepdims=True)

    # Avoid division by zero
    #norms = np.where(norms == 0, 1, norms)

    # Normalize each vector
    #all_data = all_data / norms
    
    
    # Apply dimensionality reduction
    if method.lower() == "pca":
        reducer = PCA(n_components=3)
        reduced = reducer.fit_transform(all_data)
    elif method.lower() == "tsne":
        reducer = TSNE(n_components=3, perplexity=30, max_iter=1000, random_state=42)
        reduced = reducer.fit_transform(all_data)
    elif method.lower() == "umap":
        reducer = umap.UMAP(n_components=3) # TODO: check random state if is needed, or go with GPU version
        reduced = reducer.fit_transform(all_data)
    else:
        raise NotImplementedError("Only 'pca', 'tsne', and 'umap' are implemented.")

    # Split back into text and vision
    text_reduced = reduced[: len(text_np)]
    vision_reduced = reduced[len(text_np):]
    
    # Transform to numpy array
    text_reduced = np.array(text_reduced)
    vision_reduced = np.array(vision_reduced)
    
    # Transform to torch tensor
    text_reduced = torch.tensor(text_reduced)
    vision_reduced = torch.tensor(vision_reduced)
    
    # Normalize the reduced embeddings
    text_reduced = F.normalize(text_reduced, dim=-1)
    vision_reduced = F.normalize(vision_reduced, dim=-1)

    # Plot 3D visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(text_reduced[:, 0], text_reduced[:, 1], text_reduced[:, 2], 
               c='red', alpha=0.6, label='Text')
    ax.scatter(vision_reduced[:, 0], vision_reduced[:, 1], vision_reduced[:, 2], 
               c='blue', alpha=0.6, label='Vision')

    # set the axis limits to 1.0 and -1.0
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    
    

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    ax.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        wandb.log({method: wandb.Image(save_path)})
        plt.close()
    else:
        plt.show()


# In[4]:


def compute_centroids(text_embeddings, visual_embeddings):
    """
    Computes the centroid for each pair of samples between text embeddings and visual embeddings
    by calculating the mean of the corresponding feature vectors across the two modalities.

    Parameters:
    - text_embeddings (torch.Tensor): Tensor of shape (batch_size1, feature_dim) representing text embeddings.
    - visual_embeddings (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing visual embeddings.

    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2, feature_dim) representing the centroid for each pair.
    """

    # Compute centroids by averaging text and visual embeddings
    # Expand the dimensions to allow pairwise computation
    text_expanded = text_embeddings.unsqueeze(1)  # Shape: [batch_size1, 1, feature_dim]
    visual_expanded = visual_embeddings.unsqueeze(0)  # Shape: [1, batch_size2, feature_dim]

    # Compute the centroid by averaging the embeddings
    centroids = (text_expanded + visual_expanded) / 2.0

    # Compute norms of the centroids
    centroid_norms = torch.norm(centroids, dim=-1)

    return centroid_norms, centroids

def compute_metric_ret(score_matrix: torch.Tensor, ids: List[int], ids_txt: List[int], direction: str = 'forward') -> Dict[str, float]:
    """
    Compute retrieval metrics for either text-to-vision or vision-to-text retrieval.

    Args:
        score_matrix (torch.Tensor): Similarity matrix of shape [N_text, N_image].
        ids (List[int]): List of image IDs.
        ids_txt (List[int]): List of text IDs corresponding to images.
        direction (str): 'forward' for text-to-vision, 'backward' for vision-to-text.

    Returns:
        Dict[str, float]: Dictionary containing retrieval metrics.
    """
    assert score_matrix.shape == (len(ids_txt), len(ids)), f"Score matrix shape {score_matrix.shape} does not match (len(ids_txt), len(ids))"

    if direction == 'forward':  # Text-to-Vision Retrieval
        # Sort each row in descending order
        indice_matrix = score_matrix.sort(dim=-1, descending=True)[1].tolist()
        rank = []
        for i in range(len(ids_txt)):
            gt_indice = ids.index(ids_txt[i])
            rank.append(indice_matrix[i].index(gt_indice))

        rank = torch.tensor(rank).to(score_matrix.device)

        vr_r1 = (rank < 1).sum().item() / len(ids_txt)
        vr_r5 = (rank < 5).sum().item() / len(ids_txt)
        vr_r10 = (rank < 10).sum().item() / len(ids_txt)

        eval_log = {
            'forward_r1': round(vr_r1 * 100, 4),
            'forward_r5': round(vr_r5 * 100, 4),
            'forward_r10': round(vr_r10 * 100, 4),
            #'forward_recall': f'{round(vr_r1 * 100, 1)}/{round(vr_r5 * 100, 1)}/{round(vr_r10 * 100, 1)}',
            'forward_ravg': round((vr_r1 + vr_r5 + vr_r10) / 3 * 100, 4)
        }

    else:  # Vision-to-Text Retrieval
        # Sort each column in descending order
        indice_matrix = score_matrix.sort(dim=0, descending=True)[1].permute(1, 0).tolist()
        rank = []
        for i in range(len(ids)):
            gt_indices = [idx for idx, id_txt in enumerate(ids_txt) if id_txt == ids[i]]
            rank.append(min([indice_matrix[i].index(idx) for idx in gt_indices]))

        rank = torch.tensor(rank).to(score_matrix.device)

        tr_r1 = (rank < 1).sum().item() / len(ids)
        tr_r5 = (rank < 5).sum().item() / len(ids)
        tr_r10 = (rank < 10).sum().item() / len(ids)

        eval_log = {
            'backward_r1': round(tr_r1 * 100, 4),
            'backward_r5': round(tr_r5 * 100, 4),
            'backward_r10': round(tr_r10 * 100, 4),
            #'backward_recall': f'{round(tr_r1 * 100,1)}/{round(tr_r5 * 100,1)}/{round(tr_r10 * 100,1)}',
            'backward_ravg': round((tr_r1 + tr_r5 + tr_r10) / 3 * 100, 4)
        }

    return eval_log

def compute_gap(feat_modality1: torch.Tensor, feat_modality2: torch.Tensor) -> float:
    """
    Compute the Euclidean distance between the centroids of two modalities.

    Args:
        feat_modality1 (torch.Tensor): Feature matrix of modality 1 with shape [N, D].
        feat_modality2 (torch.Tensor): Feature matrix of modality 2 with shape [N, D].

    Returns:
        float: Euclidean distance between centroids.
    """
    # Ensure features are normalized if required
    modality1_centroid = torch.mean(feat_modality1, dim=0)
    modality2_centroid = torch.mean(feat_modality2, dim=0)

    gap = modality1_centroid - modality2_centroid
    norm_gap = torch.norm(gap).item()

    return norm_gap

def compute_mean_angular_value_of_a_modality(feat_modality: torch.Tensor) -> float:
    """
    Compute the mean angular value (mean cosine similarity) of a modality.

    Args:
        feat_modality (torch.Tensor): Feature matrix with shape [N, D].

    Returns:
        float: Mean angular value.
    """
    # Compute cosine similarity matrix
    cos_sim = feat_modality @ feat_modality.T

    # Exclude diagonal elements by creating a mask
    mask = ~torch.eye(cos_sim.size(0), dtype=torch.bool, device=cos_sim.device)
    cos_sim_no_diag = cos_sim[mask]

    mean_cos_sim = cos_sim_no_diag.mean().item()

    return mean_cos_sim

def uniformity(features_modality1: torch.Tensor, features_modality2: torch.Tensor) -> float:
    x = torch.cat([features_modality1, features_modality2], dim=0)
    N = x.size(0)
    dim = x.size(1)

    x_center = torch.mean(x, dim=0, keepdim=True)
    covariance = torch.mm((x - x_center).t(), x - x_center) / N

    mean =  x.mean(0)
    np_mean = mean.data.cpu().numpy()
    np_covariance = covariance.data.cpu().numpy()
   
    ##calculation of part1
    part1 = np.sum(np.multiply(np_mean, np_mean))

    ##calculation of part2
    eps = 1e-8 
    S, Q = np.linalg.eig(np_covariance)
    S = S + eps

    mS = np.sqrt(np.diag(S.clip(min=0)))

    covariance_2 = np.dot(np.dot(Q, mS), Q.T)

    part2 = np.trace(np_covariance - 2.0/np.sqrt(dim) * covariance_2)
    wasserstein_distance = math.sqrt(part1 + 1 + part2)
    return -wasserstein_distance 

def mean_distance_of_true_pairs(features_modality1: torch.Tensor, features_modality2: torch.Tensor) -> float:
    """
    Compute the mean cosine similarity of true pairs between two modalities.

    Args:
        features_modality1 (torch.Tensor): Normalized feature matrix of modality 1 with shape [N, D].
        features_modality2 (torch.Tensor): Normalized feature matrix of modality 2 with shape [N, D].

    Returns:
        float: Mean cosine similarity of true pairs.
    """
    # Compute cosine similarity matrix
    cosine_sim = torch.matmul(features_modality1, features_modality2.T)

    # Extract diagonal elements (true pairs)
    cosine_sim_diag = torch.diag(cosine_sim)

    # Compute mean cosine similarity of true pairs
    cosine_tv_mean = torch.mean(cosine_sim_diag).item()

    return cosine_tv_mean


# In[5]:


def evaluate_model(model: torch.nn.Module, test_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Evaluate the (OpenCLIP) model on the given test_loader by computing
    text-to-image and image-to-text retrieval metrics, along with additional metrics.

    Args:
        model (torch.nn.Module): The trained (DataParallel) model.
        test_loader (DataLoader): A DataLoader for the evaluation set.
        device (torch.device): The device (CPU or GPU).

    Returns:
        Dict[str, float]: Dictionary containing all evaluation metrics.
    """
    # Put model into eval mode
    model.eval()

    # Prepare storage for embeddings
    all_image_embeds = []
    all_text_embeds = []

    # IDs for retrieval
    ids_img = []
    ids_txt = []

    current_index = 0
    
    tokenizer = open_clip.get_tokenizer('RN50')

    # No gradient needed during evaluation
    with torch.no_grad():
        for images, captions_list in tqdm.tqdm(test_loader, desc="Evaluating"):
            # Move images to device
            images = images.to(device)

            # Tokenize captions
            text_tokens = tokenizer(captions_list)
            text_tokens = text_tokens.to(device)

            # Extract embeddings using the .module references in DataParallel
            image_embeds = model.module.encode_image(images)
            text_embeds = model.module.encode_text(text_tokens)
            
            # Normalize embeddings
            image_embeds = F.normalize(image_embeds, dim=-1)
            text_embeds  = F.normalize(text_embeds, dim=-1)

            # Move embeddings to CPU for later concatenation
            image_embeds = image_embeds.cpu()
            text_embeds = text_embeds.cpu()

            # Store embeddings
            all_image_embeds.append(image_embeds)
            all_text_embeds.append(text_embeds)

            # Assign unique IDs
            bs = images.size(0)
            sample_ids = list(range(current_index, current_index + bs))
            ids_img.extend(sample_ids)
            ids_txt.extend(sample_ids)
            current_index += bs

    # Concatenate all embeddings
    all_image_embeds = torch.cat(all_image_embeds, dim=0)  # Shape: [N, D]
    all_text_embeds = torch.cat(all_text_embeds, dim=0)    # Shape: [N, D]
    
    
    visualize_embeddings(all_text_embeds, 
                                all_image_embeds, 
                                sample_size=1000, 
                                method='umap', 
                                title="CLIP Embeddings Visualization",
                                save_path="embeddings_plot_umap.png")
    """visualize_embeddings(all_text_embeds, 
                                all_image_embeds, 
                                sample_size=1000, 
                                method='tsne',
                                title="CLIP Embeddings Visualization",
                                save_path="embeddings_plot_tsne.png")"""

    # should be already normalized
    # Normalize embeddings for more stable retrieval and metric computations
    #all_image_embeds = F.normalize(all_image_embeds, dim=-1)
    #all_text_embeds = F.normalize(all_text_embeds, dim=-1)

    # Compute pairwise similarity: [N_text, N_image]
    similarity_matrix = all_text_embeds @ all_image_embeds.t()

    # Compute retrieval metrics
    log_forward = compute_metric_ret(similarity_matrix, ids_img, ids_txt, direction='forward')   # Text-to-Vision
    log_backward = compute_metric_ret(similarity_matrix, ids_img, ids_txt, direction='backward') # Vision-to-Text

    # Compute additional metrics
    gap = compute_gap(all_image_embeds, all_text_embeds)
    mean_ang_image = compute_mean_angular_value_of_a_modality(all_image_embeds)
    mean_ang_text = compute_mean_angular_value_of_a_modality(all_text_embeds)
    uniformity_metric = uniformity(all_image_embeds, all_text_embeds)
    mean_cos_true_pairs = mean_distance_of_true_pairs(all_image_embeds, all_text_embeds)

    # Combine all metrics into final_log
    final_log = {
        **log_forward,
        **log_backward,
        'gap': round(gap, 4),
        'mean_angular_value_image': round(mean_ang_image, 4), # round to 4 decimal places
        'mean_angular_value_text': round(mean_ang_text, 4),
        'uniformity': round(uniformity_metric, 4),
        'mean_cosine_similarity_true_pairs': round(mean_cos_true_pairs, 4)
    }

    print("Evaluation Results:", final_log)
    print()
    
    wandb.log(final_log)

    return final_log


# In[6]:


def train_model(config, train_loader, test_loader, device):

    # Create model & transforms from scratch (no pretrained weights)
    model, _, preprocess = open_clip.create_model_and_transforms(
        config["model"],
        pretrained=None,
        device=device
    )
    
    tokenizer = open_clip.get_tokenizer(config["model"])

    # Put the model into training mode
    model.train()

    # If you want to fine-tune *everything* from scratch, ensure all parameters require grad:
    for param in model.parameters():
        param.requires_grad = True

    # Set up training parameters from the config
    lr = config["learning_rate"]
    epochs = config["epochs"]
    temperature = config["temperature"]

    # Move the model to multiple GPUs
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])  # Use 4 GPUs
    
    contrastive_temperature_learnable = torch.nn.Parameter(torch.tensor(temperature))

    optimizer = optim.AdamW(list(model.parameters())+[contrastive_temperature_learnable], lr=lr)

    current_batch = 0
    loss = 0

    for epoch in range(epochs):
        for images, captions_list in tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}"):
            
            current_batch += 1
            
            # Move data to the primary device
            images = images.to(device)
            captions = captions_list

            # Tokenize text
            text_tokens = tokenizer(captions)
            text_tokens = text_tokens.to(device)

            # Encode image and text
            image_embeds = model.module.encode_image(images)  # Use .module for methods inside DataParallel
            text_embeds = model.module.encode_text(text_tokens)
            
            # Normalize embeddings
            image_embeds = F.normalize(image_embeds, dim=-1)
            text_embeds  = F.normalize(text_embeds, dim=-1)
            
            # Compute loss based on the experiment type
            if config["loss_type"] == "anchor":
                loss = contrastive_loss(image_embeds, text_embeds, temperature=contrastive_temperature_learnable)
            elif config["loss_type"] == "anchor+lunif":
                lunif_img = lunif_loss(image_embeds)
                lunif_txt = lunif_loss(text_embeds)
                lunif = (lunif_img + lunif_txt) / 2
                loss = contrastive_loss(image_embeds, text_embeds, temperature=contrastive_temperature_learnable) + lunif
            elif config["loss_type"] == "lunif_n_iters+frozen(text_embed)":
                if current_batch <= config["lunif_n_iters"]:
                    lunif_img = lunif_loss(image_embeds)
                    lunif_txt = lunif_loss(text_embeds)
                    lunif = (lunif_img + lunif_txt) / 2
                    loss = lunif
                else: # train on anchor loss with frozen text embeddings
                    text_embeds = text_embeds.detach()
                    loss = contrastive_loss(image_embeds, text_embeds, temperature=contrastive_temperature_learnable)
                    
            wandb.log({"train_loss": loss.item()})

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if config["evaluate_and_visualize_every_epoch"] == True:
            evaluate_model(model, test_loader, device)
            
        if (epoch+1) % config["save_checkpoint_every_n_epochs"]  == 0:
            torch.save(model.state_dict(), f"models/model_" + config["run_name"] + f"_epoch_{epoch+1}.pt")
            print(f"Model saved at epoch {epoch+1}")
        
                
    return model


# In[7]:


def dataset_loader(config):

    # Path to train images and annotations
    train_image_dir = './data/coco/images/train2017/'                          # Path to train2017 images
    train_annotation_file = './data/coco/annotations/captions_train2017.json'  # Path to train2017 captions

    # Path to test (val) images and annotations
    test_image_dir = './data/coco/images/val2017/'                          # Path to val2017 images
    test_annotation_file = './data/coco/annotations/captions_val2017.json'  # Path to val2017 captions
    
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    
    # Define the transform to be applied to the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to the model's required input size
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Create the training dataset
    train_coco = dset.CocoCaptions(
        root=train_image_dir,
        annFile=train_annotation_file,
        transform=transform
    )

    # Create the test dataset
    test_coco = dset.CocoCaptions(
        root=test_image_dir,
        annFile=test_annotation_file,
        transform=transform
    )
    
    if config["num_train_samples"] != -1:
        print(f"Subsetting the training dataset to {config['num_train_samples']} samples")
        # Subset the training dataset
        num_training_samples = config["num_train_samples"]
        subset_indices = list(range(num_training_samples))
        train_coco = Subset(train_coco, subset_indices)
    
    if config["num_test_samples"] != -1:
        print(f"Subsetting the test dataset to {config['num_test_samples']} samples")
        # Subset the test dataset
        num_test_samples = config["num_test_samples"]
        subset_indices = list(range(num_test_samples))
        test_coco = Subset(test_coco, subset_indices)

    # Every image has 5 captions at max, we need to sample one of them
    # Create collate function to sample one caption per image 
    def collate_fn(batch):
        images, captions = zip(*batch)
        images = torch.stack(images, 0)
        sel_captions = []
        
        for list_captions in captions:
            caption = random.choice(list_captions)
            sel_captions.append(caption)

        return images, sel_captions

    # Create DataLoader
    batch_size = config["batch_size"]
    train_loader = DataLoader(train_coco, batch_size=batch_size, shuffle=True , drop_last=True, collate_fn=collate_fn, num_workers=12, persistent_workers=True)
    test_loader  = DataLoader(test_coco , batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn, num_workers=12, persistent_workers=True)
    
    return train_loader, test_loader


# In[8]:


def set_seed(seed: int):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy random module
    torch.manual_seed(seed)  # PyTorch CPU random numbers
    torch.cuda.manual_seed(seed)  # PyTorch GPU random numbers for a single GPU
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU random numbers for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.benchmark = False  # Disable benchmark for deterministic behavior


# In[9]:


def main(config):
    # Set the seed for reproducibility
    set_seed(config["seed"])
    
    # Finish any existing W&B runs before starting a new one
    wandb.finish()

    # Initialize your W&B run
    wandb.init(project="sparsify-clip", config=config, name=config["run_name"])
    
    # Print the config
    print("Config:", config)
    
    # Set the device
    device_id = config["device_id"]
    device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
    
    # Load the dataset
    print("\nLoading the dataset...")
    train_loader, test_loader = dataset_loader(config)
    print("Dataset loaded.\n")
    
    # Train the model
    print("Training the model...")
    model = train_model(config, train_loader, test_loader, device)
    print("Training complete.\n")
    
    # Final evaluation of the model
    print("Final evaluation of the model...")
    final_log = evaluate_model(model, test_loader, device)
    print("Evaluation complete.\n")
    
    # Save the model and upload it to W&B
    torch.save(model.state_dict(), config["run_name"] + ".pt")
    wandb.save(config["run_name"] + ".pt")    
    
    wandb.finish()


# In[ ]:

#####%%prun

config = {
    "run_name": "{}".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),  # A readable name for this run
    "device_id": 2,      # GPU id
    "seed": 42,     # Random seed
    
    "learning_rate": 1e-4,
    "batch_size": 256,
    "epochs": 30, # TODO: 100
    "model": "RN50",
    
    "temperature": 0.07,
    
    "loss_type": "MANUAL",   # anchor, anchor+lunif
    "lunif_n_iters": 200,
    
    
    "save_checkpoint_every_n_epochs": 10,
    "evaluate_and_visualize_every_epoch": True,
    
    "num_train_samples":            -1,            # -1 for all TODO: -1
    "num_test_samples":             -1,            # -1 for all TODO: -1
}

if __name__ == "__main__":
    
    # Baseline
    #config["loss_type"] = "anchor"
    #config["run_name"] = config["model"] + "_" + config["loss_type"] + "_" + config["run_name"] 
    #print("\nTraining Baseline model")
    #main(config)
    
    
    # Anchor + Lunif
    config["loss_type"] = "anchor+lunif"
    config["run_name"] = config["model"] + "_" + config["loss_type"] + "_" + config["run_name"] 
    print("\nTraining Anchor + Lunif model")
    main(config)
    
    
    # Lunif(50itr)+frozen(text_embed)
    #config["loss_type"] = "lunif_n_iters+frozen(text_embed)"
    #config["run_name"] = config["model"] + "_" + config["loss_type"] + "_" + config["run_name"] 
    #print("\nTraining lunif_n_iters+frozen(text_embed) model")
    #main(config)
# %%
