import torch

def preprocess_obs(obs):
    """
    Converts MiniGrid observation dict into a flat tensor.
    """
    img = torch.tensor(obs["image"], dtype=torch.float32) / 255.0
    direction = torch.tensor([obs["direction"]], dtype=torch.float32)

    img = img.flatten()
    obs_vec = torch.cat([img, direction], dim=0)

    return obs_vec
