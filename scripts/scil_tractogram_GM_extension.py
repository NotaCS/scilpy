#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lengthen the streamlines so that they reach deeper in the grey matter.
The lengthened endpoints follow linearly the direction of the last streamline step.
The added length can be specified, and the script requires a grey matter mask
to ensure that the streamline end in the grey matter:
    - if the lengthening gets out of the grey matter, the protruding points will be cut.
    - if despite the lengthening, the endpoint does not reach the grey matter, the
    lengthening will be canceled for that streamline end.

The "with_atlas" option consider that a point is "protruding" (and thus must be shaved)
if the extension changes parcel (or get out of the grey matter). This help avoid the
extension crossing hemishere, for example.
"""

import nibabel as nib
import numpy as np
import argparse
import time
from nibabel.affines import apply_affine
from dipy.io.streamline import load_tractogram
from scilpy.io.utils import add_overwrite_arg, assert_inputs_exist
from nibabel.streamlines.array_sequence import ArraySequence
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_trk

# %%


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )

    p.add_argument("in_tractogram", help="Path of the input tractograms (.trk).")
    p.add_argument(
        "in_grey_matter",
        help="Path of the brain mask or atlas of the grey matter (.nii, nii.gz)."
        "\n(used as anatomy reference for the tractogram)",
    )
    p.add_argument("out_file", help="Output file path (.trk).")

    p.add_argument(
        "-l",
        "--length",
        default=3,
        help="Length to add to the streamlines ends, in mm (Default=3mm).",
    )
    p.add_argument(
        "-s",
        "--step",
        default=0.5,
        help="Step size for the added bit (in mm). Should be a multiple"
        "\nof the length to add (Default=0.5mm).",
    )
    p.add_argument(
        "-a",
        "--with_atlas",
        action="store_true",
        help="If chosen, the grey matter volume must be an atlas, and the "
        "shaving step of the extension will take into account if the "
        "extension changes parcel (and will cut it).",
    )
    p.add_argument(
        "-c",
        "--compressed",
        action="store_true",
        help="Change the end-points of the streamlines instead of adding "
        "multiple points after the previous end-points. This option "
        "will preserve the compressed state of compressed streamlines "
        "and should only be used with such compressed streamlines.",
    )
    add_overwrite_arg(p)

    return p


def compute_end_bits(strml, step_nb, step_sz, nback=1):
    """
    Compute the end bit to (potentially) add to the end of streamlines.
    (in rasmm space, necessary for euclidian distances in case of non-isotropic voxels?)

    Parameters
    ----------
    strml : ArraySequence
        Streamlines in RAS space.
    step_nb : int
        Number of points to add.
    step_sz : float
        Step-size between two points.
    nback : int
        Number of the point away from the extremity to compute the vector

    Returns
    -------
    end_bits : Array
        4D array of shape (strml_nb, 2, step_nb, 3) contaning the added points for the
        two end-points of each streamline. Each extension are ordered from streamline -> outward

    """
    end_bits = np.zeros((len(strml), 2, step_nb, 3), dtype="f")
    for i, strm in enumerate(strml):
        end1 = strm[[0, nback]]  # First bit
        dv = end1[0] - end1[1]  # Vector to follow for the added bit
        dv_mm = dv / np.sqrt((dv ** 2).sum())  # Normalized vector (mm)
        # np.arange because we are doing a linear extrapolation
        end_bits[i, 0, :] = (
            np.dot(
                np.arange(step_sz, (step_nb + 1) * step_sz, step_sz, dtype="f").reshape(
                    -1, 1
                ),
                dv_mm.reshape(1, -1),
            )
            + end1[0]
        )
        end2 = strm[[-nback - 1, -1]]  # Second bit
        dv = end2[1] - end2[0]  # Vector to follow for the added bit
        dv_mm = dv / np.sqrt((dv ** 2).sum())  # Normalized vector (mm)
        # np.arange because we are doing a linear extrapolation
        end_bits[i, 1, :] = (
            np.dot(
                np.arange(step_sz, (step_nb + 1) * step_sz, step_sz, dtype="f").reshape(
                    -1, 1
                ),
                dv_mm.reshape(1, -1),
            )
            + end2[1]
        )
    return end_bits


def generate_longer_streamlines(strml, end_bits_vox, atlas, comp):
    """
    Generator that uses end_bits (in voxel space) and lengthen the streamlines while checking if the added length
    reach the GM or get out of the GM. If it doesn't reach the GM, the streamline is not lengthened,
    and if the added end get out of the GM, the part going out is shaved.
    """
    parcels = np.unique(atlas).tolist()
    try:
        parcels.remove(0)
    except ValueError:
        pass
    for i, strm in enumerate(strml):

        end1 = end_bits_vox[i, 0, :]
        nv = 0  # Voxels to keep in the extension
        valParc = 0
        for v in end1:
            if np.any(v < 0):
                break  # Voxel out of the bounding box
            v = tuple(v.astype(int))
            valv = atlas[v]
            if not valParc and valv not in parcels:
                pass  # GM not reached yet
            elif not valParc and valv in parcels:
                valParc = valv  # Just reached the GM
            elif valParc and valv == valParc:
                pass  # in the GM
            elif valParc and valv != valParc:
                break  # Changed parcel or got out of GM
            nv += 1
        if valParc:
            end1 = end1[:nv]
        else:  # If it never reached GM
            end1 = end1[:0]

        end2 = end_bits_vox[i, 1, :]
        nv = 0  # Voxels to keep in the extension
        valParc = 0
        for v in end2:
            v = tuple(v.astype(int))
            valv = atlas[v]
            if not valParc and valv not in parcels:
                pass  # GM not reached yet
            elif not valParc and valv in parcels:
                valParc = valv  # Just reached the GM
            elif valParc and valv == valParc:
                pass  # in the GM
            elif valParc and valv != valParc:
                break  # Changed parcel or got out of GM
            nv += 1
        if valParc:
            end2 = end2[:nv]
        else:  # If it never reached GM
            end2 = end2[:0]

        if comp:  # If the streamlines are compressed, we just change the end-points
            newStrm = strm
            if len(end1):
                newStrm[0] = end1[-1]
            if len(end2):
                newStrm[-1] = end2[-1]
        else:  # Otherwise we concatenate the new end bits
            newStrm = np.concatenate((end1[::-1], strm, end2))
        yield newStrm


# %% def main():
t0 = time.time()
parser = _build_arg_parser()
args = parser.parse_args()

assert_inputs_exist(parser, [args.in_tractogram, args.in_grey_matter])

gm_F = args.in_grey_matter
trk_F = args.in_tractogram
added_len = float(args.length)
step_size = args.step

if step_size > added_len:
    raise ValueError("Step size bigger that the max length to add.")

step_number = int(added_len // step_size)

gm_im = nib.load(gm_F)
gm = gm_im.get_fdata(dtype="f")
if not args.with_atlas:
    gm = np.where(gm, 1, 0)

print("Loading the tractogram...")
trk_sft = load_tractogram(trk_F, "same", bbox_valid_check=False)

trk_sft.to_rasmm()
trk_sft.to_corner()

print("Computing all extensions...")
endBits = compute_end_bits(trk_sft.streamlines, step_number, step_size)
endBits = apply_affine(np.linalg.inv(trk_sft.affine), endBits)  # To voxel space
trk_sft.to_vox()

print("Shaving bad points...")

strm_gen = generate_longer_streamlines(
    trk_sft.streamlines, endBits, gm, args.compressed
)
new_trk_sft = StatefulTractogram(ArraySequence(strm_gen), trk_F, trk_sft.space)

save_trk(new_trk_sft, args.out_file, bbox_valid_check=False)
