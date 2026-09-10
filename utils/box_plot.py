import torch
from utils.box_ops import box_cxcywh_to_xyxy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_bboxes_on_img(
    img: torch.tensor,
    target,
    idx_to_lesion_fn,
    img_size,
    cmap={
        "Enlarged cardiac silhouette": "yellow",
        "Atelectasis": "red",
        "Pleural abnormality": "orange",
        "Consolidation": "lightgreen",
        "Pulmonary edema": "dodgerblue",
    },
):
    fig, ax = plt.subplots(
        dpi=128,
    )

    plt.imshow(img.permute(1, 2, 0).numpy())

    for label, bbox in zip(
        target["labels"].detach().cpu().numpy(),
        target["boxes"].detach().cpu().numpy(),
    ):
        bbox = box_cxcywh_to_xyxy(torch.tensor(bbox * img_size)).numpy()
        disease = idx_to_lesion_fn(label)
        c = cmap[disease]
        ax.add_patch(
            Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                fill=False,
                color=c,
                linewidth=2,
            )
        )
        ax.text(
            bbox[0],
            bbox[1],
            disease,
            color="black",
            backgroundcolor=c,
        )
