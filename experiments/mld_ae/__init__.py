"""
Uni-modal deterministic autoencoders for the modalities of the HS data set (test_data/data/multimodal), as the
first stage of Multi-modal Latent Diffusion (MLD, Bounoua, Franzese and Michiardi, ICLR 2024;
reference/theory_papers/arXiv-2306.04445v2, code reference/MLD).

MLD trains one autoencoder per modality, each alone, on the reconstruction loss of its modality (eq. 1 of the
paper): z_i = e_i(x_i), x_i_hat = d_i(z_i). A later stage then models the concatenated latents [z_1, ..., z_M].
This folder does not import src/.

    data.py         per-modality tables, column kinds and transforms, the patient split
    model.py        ModalityAE (encoder, decoder, loss for each column kind)
    train.py        trains the autoencoders: runs/mld_ae/<modality>/model.pt
    reconstruct.py  encodes and decodes each modality, writes the latents, the metrics and the plots
    plots.py        the figures of reconstruct.py
"""
