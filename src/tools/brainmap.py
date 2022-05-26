#!/usr/bin/env python3

import os
import shutil
from typing import Tuple
import numpy as np
from PIL import Image
import nilearn
from nilearn import input_data, surface, plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


def clear_matplotlib_fig_cache() -> None:
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    return None


def _save_surfaces_to_workdir(
    img,
    workdir,
    mask_img=None,
    axsize=(7,5),
    dpi=300,
    cmap='inferno',
    vmax=None,
    vmin=None,
    threshold=None,
    threshold_percentile=None,
    darkness=0.8,
    contour_parcellation=None,
    bg_on_data=False
    ) -> Tuple[str]:

    fsaverage_surface = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')

    if mask_img is None:
        masker = input_data.NiftiMasker(mask_strategy='background')
        fitted_masker = masker.fit(imgs=img)
        mask_img = fitted_masker.mask_img_

    img = nilearn.masking.unmask(
        X=nilearn.masking.apply_mask(
            imgs=img,
            mask_img=mask_img
        ),
        mask_img=mask_img
    )

    if (
        threshold is None and
        threshold_percentile is not None
    ):
        assert np.logical_and(
            threshold_percentile>0,
            threshold_percentile<100
        ), 'threshold_percentile needs to be >0 and <100'
        threshold = np.percentile(
            a=nilearn.masking.apply_mask(
                imgs=img,
                mask_img=mask_img
            ).ravel(),
            q=threshold_percentile
        )

    plt.rcParams['axes.facecolor'] = 'black'
    image_paths = []

    for view, hemi in zip(
        [
            'lateral', 'lateral',
            'medial', 'medial',
            'ventral', 'ventral'
        ],
        [
            'left', 'right',
            'left', 'right',
            'left', 'right'
        ]
    ):
        fig = plt.figure(
            figsize=axsize,
            dpi=dpi
        )
        ax = fig.add_subplot(
            1, 1, 1,
            projection='3d'
        )
        surface_img = surface.vol_to_surf(
            img=img,
            surf_mesh=fsaverage_surface[f'pial_{hemi}']
        )
        figure = plotting.plot_surf(
            surf_mesh=fsaverage_surface[f'infl_{hemi}'],
            surf_map=surface_img,
            hemi=hemi,
            view=view,
            threshold=threshold,
            colorbar=False,
            bg_map=fsaverage_surface[f'sulc_{hemi}'],
            cmap=cmap,
            vmax=vmax,
            vmin=vmin,
            darkness=darkness,
            bg_on_data=bg_on_data,
            axes=ax
        )
        if contour_parcellation is not None:
            parcelation_surf = surface.vol_to_surf(
                img=contour_parcellation,
                surf_mesh=fsaverage_surface[f'pial_{hemi}']
            )
            plotting.plot_surf_contours(
                surf_mesh=fsaverage_surface[f'infl_{hemi}'],
                roi_map=parcelation_surf,
                # labels=labels,
                levels=[1, 2, 3, 4, 5, 6, 7],
                figure=figure,
                colors=sns.color_palette()[:7]
            )
        plt.savefig(
            os.path.join(
                workdir,
                '{}_{}.png'.format(
                    view, hemi
                )
            ),
            dpi=dpi,
            facecolor='black',
            edgecolor='none'
        )
        image_paths.append(
            os.path.join(
                workdir,
                '{}_{}.png'.format(
                    view, hemi
                )
            )
        )
        clear_matplotlib_fig_cache()

    return image_paths


def _save_colorbar_to_workdir(
    workdir,
    axsize=(7,5),
    cmap='inferno',
    vmax=None,
    vmin=None,
    dpi=300,
    ) -> str:
    fig, ax = plt.subplots(
            1,1,
            figsize=axsize,
            dpi=dpi
    )
    vmin = -vmax if vmin is None else vmin
    img = ax.imshow(
        np.array([[vmin, vmax]]),
        cmap=cmap
        )
    img.set_visible(False)
    cbar = plt.colorbar(
        img,
        orientation="horizontal",
        ax=ax
    )
    cbar.ax.tick_params(
        labelsize=30,
        colors='white',
        width=0
    )
    ax.remove()
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            workdir,
            'colorbar.png'
        ),
        dpi=dpi,
        facecolor='black',
        edgecolor='none'
    )
    return os.path.join(
        workdir,
        'colorbar.png'
    )


def plot_brain_map(
    img,
    path,
    mask_img=None,
    workdir=None,
    axsize=(7, 5),
    dpi=300,
    w_pad=0,
    h_pad=0,
    threshold=None,
    threshold_percentile=None,
    vmax=None,
    vmin=None,
    darkness=0.8,
    cmap='inferno',
    contour_parcellation=None,
    bg_on_data=False,
    colorbar=True
    ) -> None:
    
    if workdir is None:
        workdir = os.path.join(
            os.path.dirname(path),
            '.plot_brain_map_cache'
        )

    vmax = np.percentile(img.get_fdata(), 99.75) if vmax is None else vmax

    os.makedirs(workdir, exist_ok=True)
    image_paths = _save_surfaces_to_workdir(
        img=img,
        workdir=workdir,
        mask_img=mask_img,
        axsize=axsize,
        dpi=dpi,
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
        threshold=threshold,
        threshold_percentile=threshold_percentile,
        darkness=darkness,
        contour_parcellation=contour_parcellation,
        bg_on_data=bg_on_data
    )

    if colorbar:
        cbar_path = _save_colorbar_to_workdir(
            workdir=workdir,
            axsize=axsize,
            cmap=cmap,
            vmax=vmax,
            vmin=vmin,
            dpi=dpi,
        )

    positions = np.zeros((7,4)) if colorbar else np.zeros((6,4))
    W = [
        (w_pad, w_pad+axsize[0]*dpi),
        (w_pad*2+axsize[0]*dpi, w_pad*2+2*axsize[0]*dpi)
    ]
    H = [
        (h_pad, h_pad+axsize[1]*dpi),
        (h_pad*2+axsize[1]*dpi, h_pad*2+2*axsize[1]*dpi),
        (h_pad*4+2*axsize[1]*dpi, h_pad*4+2.5*axsize[1]*dpi)
    ]
    
    if colorbar:
        W.append(
            (w_pad*2+0.15*axsize[0]*dpi, w_pad*2+1.85*axsize[0]*dpi)
        )
        H.append(
            (h_pad*6+2.5*axsize[1]*dpi, h_pad*6+3*axsize[1]*dpi)
        )
    
    positions[0] = (W[0][0], H[0][0] ,W[0][1], H[0][1])
    positions[1] = (W[1][0], H[0][0] ,W[1][1], H[0][1])
    positions[2] = (W[0][0], H[1][0] ,W[0][1], H[1][1])
    positions[3] = (W[1][0], H[1][0] ,W[1][1], H[1][1])
    positions[4] = (W[0][0], H[2][0] ,W[0][1], H[2][1])
    positions[5] = (W[1][0], H[2][0] ,W[1][1], H[2][1])
    
    if colorbar:
        positions[6] = (W[2][0], H[3][0] ,W[2][1], H[3][1])

    background_w = 2*axsize[0]*dpi+(4*w_pad)
    background_h = 2.5*axsize[1]*dpi+(6*h_pad)
    
    if colorbar:
        background_h += 0.5*axsize[1]*dpi+(2*h_pad)

    background_img = Image.new(
        mode='RGBA',
        size=(int(background_w), int(background_h)),
        color="BLACK"
    )
        
    for i, img in enumerate(image_paths):
        img_i = Image.open(
            img,
            mode='r'
        )
        pos_i = positions[i].astype(int)
        w, h = (
            int(pos_i[2]-pos_i[0]),
            int(pos_i[3]-pos_i[1])
        )
        
        if 'ventral' in img:
            img_i = img_i.crop(
                (
                    int(img_i.size[0]*0.25),
                    int(img_i.size[1]*0.33),
                    int(img_i.size[0]*0.75),
                    int(img_i.size[1]*0.66)
                )
            )
            img_i = img_i.resize((w, h))
            
            if 'right' in img:
                img_i = img_i.rotate(180)
                pos_i[0] += w//19
                pos_i[2] += w//19

        else:
            img_i = img_i.crop(
                (
                    int(img_i.size[0]*0.25),
                    int(img_i.size[1]*0.25),
                    int(img_i.size[0]*0.75),
                    int(img_i.size[1]*0.75)
                )
            )
            img_i = img_i.resize((w, h))
        
        background_img.paste(img_i, pos_i)

    if colorbar:
        img_i = Image.open(
            cbar_path,
            mode='r'
        )
        pos_i = positions[-1].astype(int)
        w, h = (
            int(pos_i[2]-pos_i[0]),
            int(pos_i[3]-pos_i[1])
        )
        img_i = img_i.crop(
            (
                0,
                int(img_i.size[1]*0.7),
                img_i.size[0],
                img_i.size[1]
            )
        )
        img_i = img_i.resize((w, h))
        background_img.paste(img_i, pos_i)

    background_img.save(path)
    shutil.rmtree(workdir)
    clear_matplotlib_fig_cache()