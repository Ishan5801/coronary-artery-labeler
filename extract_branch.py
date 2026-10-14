#!/usr/bin/env python3
"""
Split a coronary artery mask (.nii.gz) into LEFT and RIGHT coronary artery,
then split each of them into individual branches.

    python extract_branches.py                      # uses the default mask path set in main()
    python extract_branches.py coronary_mask.nii.gz # or pass a mask explicitly

Output layout
-------------
    branches/
        left_artery/
            branch_1_l.nii.gz, branch_2_l.nii.gz, ...
        right_artery/
            branch_1_r.nii.gz, branch_2_r.nii.gz, ...

(binary masks, same shape / affine as the input; inside each artery folder
branch 1 is the largest branch by volume, then 2, 3, ... in decreasing size.)

Step A - left / right artery
----------------------------
* The mask is cut into connected components.
* The significant components (>= --min-component-frac of the mask) are
  grouped into TWO arteries by single-linkage clustering: the two groups are
  the ones separated by the biggest gap, so a fragmented left tree still ends
  up in one group. Small leftover islands join the nearest group.
* Which group is left / right: the NIfTI world space is RAS+ (x grows towards
  the patient's RIGHT), so the group whose centroid has the smaller x is the
  LEFT artery. This assumes a normal anatomy (no dextrocardia / situs
  inversus) and a correct affine; the decision is printed so you can check
  it, and --swap-sides flips it.
* Sanity check: if the smaller group holds < --min-artery-frac (10%) of the
  mask, the split is rejected as implausible.
* If the two arteries are joined into one connected mask (e.g. through the
  aortic root) or only one is present, the script cannot split them by itself:
  it writes everything to branches/unsplit/ and tells you. Give it two
  centerlines with --split-centerlines A.nii.gz B.nii.gz to split by nearest
  centerline instead.

Step B - branches inside each artery
------------------------------------
0. Fill closed cavities inside the mask (cleanup only).
1. Skeletonize the artery (or use --centerline).
2. Find junctions (skeleton voxels with >= 3 neighbours whose cluster joins
   >= 3 segments).
3. Merge junctions closer than --merge-junction-mm.
4. Remove short dead-end spurs (--min-spur-mm).
5. Label every voxel with the branch it is connected to (watershed seeded
   from the branch segments, flooding inside the mask only).
6. Report the number of independent LOOPS in the skeleton graph. A coronary
   tree has none; loops > 0 means touching vessels / tunnels / blobs (use
   --debug to inspect).

Requires: numpy, scipy, scikit-image, nibabel
"""
import argparse
import glob
import os
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from skimage.segmentation import watershed

S26 = np.ones((3, 3, 3), dtype=bool)
_KERNEL = np.ones((3, 3, 3), dtype=np.uint8)
_KERNEL[1, 1, 1] = 0


# ============================================================================
# STEP B helpers: skeleton graph
# ============================================================================
def _neighbour_count(skel):
    return ndi.convolve(skel.astype(np.uint8), _KERNEL, mode="constant") * skel


def _grow(sl, shape, g=1):
    return tuple(
        slice(max(s.start - g, 0), min(s.stop + g, n)) for s, n in zip(sl, shape)
    )


def _cut(skel, node_mask):
    """
    Cut the skeleton at node_mask. Clusters touching < 3 segments are not real
    junctions and are put back into the skeleton.
    Returns seg_lab, n_seg, node_lab (0 = not a node).
    """
    node_lab, n = ndi.label(node_mask, structure=S26)
    seg0, _ = ndi.label(skel & ~node_mask, structure=S26)
    keep = np.zeros(n + 1, dtype=bool)
    for j, sl in enumerate(ndi.find_objects(node_lab), start=1):
        sl = _grow(sl, skel.shape)
        cluster = node_lab[sl] == j
        halo = ndi.binary_dilation(cluster, structure=S26)
        keep[j] = len(np.unique(seg0[sl][halo & (seg0[sl] > 0)])) >= 3
    kept = keep[node_lab]
    seg_lab, n_seg = ndi.label(skel & ~kept, structure=S26)
    return seg_lab, n_seg, np.where(kept, node_lab, 0)


def _segment_nodes(seg_lab, node_lab):
    """{segment id: [node ids it touches]}"""
    out = {}
    for b, sl in enumerate(ndi.find_objects(seg_lab), start=1):
        if sl is None:
            continue
        sl = _grow(sl, seg_lab.shape)
        seg = seg_lab[sl] == b
        halo = ndi.binary_dilation(seg, structure=S26)
        out[b] = np.unique(node_lab[sl][halo & (node_lab[sl] > 0)]).tolist()
    return out


def _analyse_skeleton(skel, spacing, merge_mm):
    """Cut skeleton into segments + junction nodes; merge nearby junctions."""
    node_mask = skel & (_neighbour_count(skel) >= 3)
    seg_lab, n_seg, node_lab = _cut(skel, node_mask)
    if merge_mm > 0:
        step = float(np.mean(spacing))
        for _ in range(5):
            sizes = np.bincount(seg_lab.ravel(), minlength=n_seg + 1)
            absorb = np.zeros(n_seg + 1, dtype=bool)
            for b, nodes in _segment_nodes(seg_lab, node_lab).items():
                if len(nodes) >= 2 and sizes[b] * step < merge_mm:
                    absorb[b] = True  # short segment between two junctions
            if not absorb.any():
                break
            seg_lab, n_seg, node_lab = _cut(skel, (node_lab > 0) | absorb[seg_lab])
    return seg_lab, n_seg, node_lab


def _prune_spurs(skel, spacing, min_len_mm, merge_mm, max_iter=10):
    """Delete dead-end segments (touch exactly one junction) shorter than min_len_mm."""
    if min_len_mm <= 0:
        return skel
    skel = skel.copy()
    step = float(np.mean(spacing))
    for _ in range(max_iter):
        seg_lab, n_seg, node_lab = _analyse_skeleton(skel, spacing, merge_mm)
        if not node_lab.any():
            break
        sizes = np.bincount(seg_lab.ravel(), minlength=n_seg + 1)
        remove = [b for b, nodes in _segment_nodes(seg_lab, node_lab).items()
                  if len(nodes) == 1 and sizes[b] * step < min_len_mm]
        if not remove:
            break
        skel[np.isin(seg_lab, remove)] = False
    return skel


# ============================================================================
# STEP B: branches inside ONE artery
# ============================================================================
def split_mask_into_branches(mask, spacing=(1.0, 1.0, 1.0), centerline=None,
                             min_spur_mm=3.0, merge_mm=3.0, fill_holes=True):
    """
    Returns
    -------
    labels : int32 volume, 0 = background, 1..N = branch id (1 = largest)
    info   : dict (branches, junctions, loops, filled_voxels, unassigned_voxels,
             skeleton, nodes)
    """
    mask = mask > 0
    work = mask
    filled_voxels = 0
    if fill_holes:
        work = ndi.binary_fill_holes(mask)
        filled_voxels = int(work.sum() - mask.sum())

    src = work if centerline is None else (centerline > 0)
    skel = skeletonize(src).astype(bool) & work
    if not skel.any():
        raise ValueError("Empty skeleton: the centerline does not overlap the mask.")

    skel = _prune_spurs(skel, spacing, min_spur_mm, merge_mm)
    seg_lab, n_seg, node_lab = _analyse_skeleton(skel, spacing, merge_mm)

    # connectivity-aware assignment of every mask voxel to a branch
    dist = ndi.distance_transform_edt(~skel, sampling=spacing)
    labels = watershed(dist, markers=seg_lab, mask=work, connectivity=3).astype(np.int32)

    # junction summary + loop count (ids are pre-renumbering here)
    junctions, pairs = [], []
    for j, sl in enumerate(ndi.find_objects(node_lab), start=1):
        if sl is None:
            continue
        sl = _grow(sl, skel.shape)
        cluster = node_lab[sl] == j
        halo = ndi.binary_dilation(cluster, structure=S26)
        ids = np.unique(seg_lab[sl][halo & (seg_lab[sl] > 0)]).tolist()
        centre = np.argwhere(cluster).mean(axis=0) + [s.start for s in sl]
        junctions.append({
            "type": {3: "bifurcation", 4: "trifurcation"}.get(len(ids), f"{len(ids)}-way"),
            "centre_vox": centre.round(1).tolist(),
            "branches": ids,
        })
        pairs += [(b, len(junctions) - 1) for b in ids]

    # independent loops of the bipartite (segment <-> junction) graph
    V = n_seg + len(junctions)
    rows = [b - 1 for b, _ in pairs]
    cols = [n_seg + k for _, k in pairs]
    adj = coo_matrix((np.ones(len(pairs)), (rows, cols)), shape=(V, V))
    n_comp = connected_components(adj, directed=False)[0]
    loops = len(pairs) - V + n_comp

    # renumber: branch 1 = largest by volume
    counts = np.bincount(labels.ravel(), minlength=n_seg + 1)
    seg_vox = np.bincount(seg_lab.ravel(), minlength=n_seg + 1)
    step = float(np.mean(spacing))
    order = [b for b in np.argsort(-counts[1:]) + 1 if counts[b] > 0]
    lut = np.zeros(n_seg + 1, dtype=np.int32)
    for new_id, old_id in enumerate(order, start=1):
        lut[old_id] = new_id
    labels = lut[labels]
    for j in junctions:
        j["branches"] = sorted(int(lut[b]) for b in j["branches"])

    branches = [{"id": i, "mask_voxels": int(counts[old]),
                 "length_mm": float(seg_vox[old] * step)}
                for i, old in enumerate(order, start=1)]
    info = {
        "junctions": junctions,
        "branches": branches,
        "loops": int(loops),
        "filled_voxels": filled_voxels,
        "unassigned_voxels": int((work & (labels == 0)).sum()),
        "skeleton": skel,
        "nodes": node_lab > 0,
    }
    return labels, info


# ============================================================================
# STEP A: left / right artery
# ============================================================================
def _world_x(mask, affine):
    """x coordinate (mm, RAS+: grows towards the patient's RIGHT) of the centroid."""
    c = np.argwhere(mask).mean(axis=0)
    return float(nib.affines.apply_affine(affine, c)[0])


def split_arteries_by_components(mask, spacing, min_frac=0.03, max_pts=4000):
    """
    Group the connected components of the mask into two arteries.

    Returns (mask_a, mask_b, table) or (None, None, table) if the mask has
    fewer than two significant components.
    table = list of dicts (component id, voxels, group) for reporting.
    """
    lab, n = ndi.label(mask, structure=S26)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    sizes[0] = 0
    total = int(sizes.sum())
    sp = np.asarray(spacing, dtype=float)

    objs = ndi.find_objects(lab)
    pts = {}
    for i in range(1, n + 1):
        sl = objs[i - 1]
        p = np.argwhere(lab[sl] == i) + np.array([s.start for s in sl])
        pts[i] = p[:: max(1, len(p) // max_pts)] * sp  # physical (mm), sub-sampled

    sig = [i for i in range(1, n + 1) if sizes[i] >= min_frac * total]
    table = [{"id": i, "voxels": int(sizes[i]), "group": None}
             for i in np.argsort(-sizes)[:12] if i > 0 and sizes[i] > 0]
    if len(sig) < 2:
        return None, None, table

    # single-linkage into 2 groups = drop the longest edge of the MST
    k = len(sig)
    trees = [cKDTree(pts[i]) for i in sig]
    D = np.zeros((k, k))
    for a in range(k):
        for b in range(a + 1, k):
            D[a, b] = D[b, a] = trees[a].query(pts[sig[b]])[0].min()
    mst = minimum_spanning_tree(csr_matrix(np.triu(D))).toarray()
    a, b = np.unravel_index(np.argmax(mst), mst.shape)
    mst[a, b] = 0
    _, grp = connected_components(csr_matrix(mst), directed=False)

    group_of = {i: int(grp[j]) for j, i in enumerate(sig)}
    group_pts = [np.vstack([pts[i] for i in sig if group_of[i] == g]) for g in (0, 1)]
    group_trees = [cKDTree(p) for p in group_pts]
    for i in range(1, n + 1):  # leftover small components -> nearest group
        if i not in group_of:
            d = [t.query(pts[i])[0].min() for t in group_trees]
            group_of[i] = int(np.argmin(d))

    lut = np.zeros(n + 1, dtype=np.int8)
    for i, g in group_of.items():
        lut[i] = g + 1
    for row in table:
        row["group"] = group_of[row["id"]]
    vol = lut[lab]
    return vol == 1, vol == 2, table


def split_arteries_by_centerlines(mask, cl1, cl2, spacing):
    """Assign each mask voxel to the artery whose centerline is nearest (mm)."""
    sp = np.asarray(spacing, dtype=float)
    vox = np.argwhere(mask)
    p1, p2 = np.argwhere(cl1 > 0), np.argwhere(cl2 > 0)
    if len(p1) == 0 or len(p2) == 0:
        sys.exit("One of the --split-centerlines files is empty.")
    d1 = cKDTree(p1 * sp).query(vox * sp)[0]
    d2 = cKDTree(p2 * sp).query(vox * sp)[0]
    a = np.zeros(mask.shape, dtype=bool)
    b = np.zeros(mask.shape, dtype=bool)
    v1, v2 = vox[d1 <= d2], vox[d1 > d2]
    a[tuple(v1.T)] = True
    b[tuple(v2.T)] = True
    return a, b


# ============================================================================
# IO / CLI
# ============================================================================
def _load_3d(path):
    nii = nib.load(path)
    data = np.squeeze(np.asanyarray(nii.dataobj))
    if data.ndim != 3:
        sys.exit(f"{path}: expected a 3D volume, got shape {data.shape}")
    return nii, data


def _save(vol, ref_nii, path, dtype):
    hdr = ref_nii.header.copy()
    hdr.set_data_dtype(dtype)
    try:
        nib.save(nib.Nifti1Image(vol.astype(dtype), ref_nii.affine, hdr), path)
    except PermissionError:
        print(f"Warning: could not overwrite '{path}' because it is in use; skipping.")


def process_artery(name, art_mask, nii, spacing, centerline, args, side_code=None):
    """Run step B on one artery and save into <out_dir>/<name>/."""
    out_dir = os.path.join(args.out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(out_dir, "branch_*.nii.gz")):
        try:
            os.remove(old)  # avoid stale files from a previous run
        except PermissionError:
            print(f"Warning: could not delete stale output '{old}' because it is in use; continuing.")

    idx = np.argwhere(art_mask)
    lo = np.maximum(idx.min(axis=0) - 3, 0)
    hi = np.minimum(idx.max(axis=0) + 4, art_mask.shape)
    crop = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    mask_c = art_mask[crop]

    labels_c, info = split_mask_into_branches(
        mask_c, spacing,
        None if centerline is None else centerline[crop],
        args.min_spur_mm, args.merge_junction_mm, not args.no_fill_holes,
    )

    n = len(info["branches"])
    for b in range(1, n + 1):
        vol = np.zeros(art_mask.shape, dtype=np.uint8)
        vol[crop] = (labels_c == b) & mask_c  # restricted to the ORIGINAL mask
        suffix = side_code or {"left_artery": "l", "right_artery": "r"}.get(name, "u")
        _save(vol, nii, os.path.join(out_dir, f"branch_{b}_{suffix}.nii.gz"), np.uint8)

    if args.save_labels:
        full = np.zeros(art_mask.shape, dtype=np.uint16)
        full[crop] = np.where(mask_c, labels_c, 0)
        _save(full, nii, os.path.join(out_dir, "all_branches_labeled.nii.gz"), np.uint16)

    if args.debug:
        for key, fname in (("skeleton", "debug_skeleton"), ("nodes", "debug_junctions")):
            vol = np.zeros(art_mask.shape, dtype=np.uint8)
            vol[crop] = info[key]
            _save(vol, nii, os.path.join(out_dir, f"{fname}.nii.gz"), np.uint8)

    # ---- report (voxel coordinates are 0-based, in the ORIGINAL volume) ---
    print(f"\n{'=' * 60}\n{name.upper()}  ({int(art_mask.sum())} voxels)  ->  {n} branches "
          f"saved to '{out_dir}/'\n{'=' * 60}")
    print(f"  {'branch':<8}{'voxels':>9}{'length_mm':>11}")
    for b in info["branches"]:
        print(f"  {b['id']:<8}{b['mask_voxels']:>9}{b['length_mm']:>11.1f}")
    print(f"\n  Junctions: {len(info['junctions'])}")
    for j in info["junctions"]:
        c = [round(float(v) + float(o), 1) for v, o in zip(j["centre_vox"], lo)]
        print(f"    {j['type']:<12} at voxel {c}  joins branches {j['branches']}")
    print(f"\n  Holes filled before skeletonizing: {info['filled_voxels']} voxels")
    print(f"  Independent loops in skeleton graph: {info['loops']}"
          + ("   (0 = clean tree)" if info["loops"] == 0 else
             "   <-- a vessel tree should have 0; inspect with --debug"))
    if info["unassigned_voxels"]:
        print(f"  WARNING: {info['unassigned_voxels']} voxels not assigned to any branch "
              f"(isolated islands with no centerline).")


def main():
    default_mask = r"D:\AICOE- Ishan\Codes\data\ICAS\masks\Img_003_Image.nii.gz"
    ap = argparse.ArgumentParser(description="Split a coronary mask into left/right artery "
                                             "and branches.")
    ap.add_argument("mask", nargs="?", default=default_mask,
                    help="coronary artery mask (.nii / .nii.gz)")
    ap.add_argument("-o", "--out-dir", default="branches",
                    help="output folder (default: ./branches)")
    ap.add_argument("--split-centerlines", nargs=2, metavar=("CL_A", "CL_B"), default=None,
                    help="split into two arteries by nearest of these two centerlines "
                         "instead of by connected components (use when left and right "
                         "are joined in the mask)")
    ap.add_argument("--swap-sides", action="store_true",
                    help="swap the left/right decision")
    ap.add_argument("--min-component-frac", type=float, default=0.03,
                    help="components smaller than this fraction of the mask do not decide "
                         "the left/right grouping; they join the nearest artery (default 0.03)")
    ap.add_argument("--min-artery-frac", type=float, default=0.10,
                    help="sanity check: if the smaller of the two arteries holds less than "
                         "this fraction of the mask the split is rejected (likely a joined "
                         "mask or fragments of one artery); 0 disables (default 0.10)")
    ap.add_argument("--centerline", default=None,
                    help="optional centerline .nii.gz used as the skeleton for step B; "
                         "if omitted each artery is skeletonized")
    ap.add_argument("--min-spur-mm", type=float, default=3.0,
                    help="remove dead-end branches shorter than this (mm); 0 = keep all "
                         "(default 3.0)")
    ap.add_argument("--merge-junction-mm", type=float, default=3.0,
                    help="junctions connected by a segment shorter than this (mm) are "
                         "merged into one; 0 = off (default 3.0)")
    ap.add_argument("--no-fill-holes", action="store_true",
                    help="do not fill closed cavities inside the mask")
    ap.add_argument("--save-labels", action="store_true",
                    help="also save all_branches_labeled.nii.gz in each artery folder")
    ap.add_argument("--debug", action="store_true",
                    help="also save left_right_labels.nii.gz (1=left, 2=right) and, per "
                         "artery, debug_skeleton / debug_junctions for ITK-SNAP / 3D Slicer")
    args = ap.parse_args()

    if not os.path.exists(args.mask):
        sys.exit(f"Mask file not found: {args.mask}")

    nii, data = _load_3d(args.mask)
    mask = data > 0
    if not mask.any():
        sys.exit("The mask is empty.")
    spacing = tuple(float(z) for z in nii.header.get_zooms()[:3])
    print(f"Mask: {args.mask}\n  shape={mask.shape}  spacing={spacing}  orientation="
          f"{''.join(nib.aff2axcodes(nii.affine))}  voxels={int(mask.sum())}  "
          f"volume={mask.sum() * np.prod(spacing) / 1000:.2f} mL")

    centerline = None
    if args.centerline:
        _, centerline = _load_3d(args.centerline)
        if centerline.shape != mask.shape:
            sys.exit("Centerline and mask have different shapes.")

    # ---- STEP A: left / right --------------------------------------------
    if args.split_centerlines:
        cls = []
        for p in args.split_centerlines:
            if not os.path.exists(p):
                sys.exit(f"Centerline not found: {p}")
            _, c = _load_3d(p)
            if c.shape != mask.shape:
                sys.exit(f"{p}: shape differs from the mask.")
            cls.append(c)
        m_a, m_b = split_arteries_by_centerlines(mask, cls[0], cls[1], spacing)
        table = []
        print("\nArtery split: nearest of the two given centerlines.")
    else:
        m_a, m_b, table = split_arteries_by_components(mask, spacing, args.min_component_frac)
        if table:
            print("\nLargest connected components of the mask:")
            print(f"  {'id':<6}{'voxels':>9}   group")
            for r in table:
                print(f"  {r['id']:<6}{r['voxels']:>9}   "
                      f"{'-' if r['group'] is None else r['group']}")

    reason = "the mask has fewer than two significant connected components"
    if m_a is not None and not args.split_centerlines and args.min_artery_frac > 0:
        small = min(int(m_a.sum()), int(m_b.sum())) / float(mask.sum())
        if small < args.min_artery_frac:
            reason = (f"the smaller of the two groups holds only {small:.1%} of the mask "
                      f"(< --min-artery-frac {args.min_artery_frac:.0%}), which looks like "
                      f"fragments of ONE artery or a joined mask, not a left/right pair")
            m_a = m_b = None

    if m_a is None:
        print(f"\nWARNING: cannot separate left and right automatically: {reason}.\n"
              "         Writing everything to 'unsplit/'. To separate them, pass two "
              "centerlines with --split-centerlines A.nii.gz B.nii.gz\n"
              "         (or lower --min-artery-frac if one artery really is that small).")
        arteries = {"unsplit": mask}
    else:
        xa, xb = _world_x(m_a, nii.affine), _world_x(m_b, nii.affine)
        left, right = (m_a, m_b) if xa < xb else (m_b, m_a)
        if args.swap_sides:
            left, right = right, left
        xl, xr = _world_x(left, nii.affine), _world_x(right, nii.affine)
        print(f"\nLeft  artery: {int(left.sum())} voxels, centroid x = {xl:.1f} mm (world, RAS+)")
        print(f"Right artery: {int(right.sum())} voxels, centroid x = {xr:.1f} mm (world, RAS+)")
        print("  (RAS+: smaller x = patient's left. Check with ITK-SNAP / Slicer; "
              "use --swap-sides if reversed.)")
        arteries = {"left_artery": left, "right_artery": right}
        if args.debug:
            os.makedirs(args.out_dir, exist_ok=True)
            lr = np.zeros(mask.shape, dtype=np.uint8)
            lr[left] = 1
            lr[right] = 2
            _save(lr, nii, os.path.join(args.out_dir, "left_right_labels.nii.gz"), np.uint8)

    # ---- STEP B: branches inside each artery -----------------------------
    for name, art in arteries.items():
        if not art.any():
            print(f"\n{name}: empty, skipped.")
            continue
        side_code = {"left_artery": "l", "right_artery": "r"}.get(name, "u")
        process_artery(name, art, nii, spacing, centerline, args, side_code)

    print(f"\nDone. Output folder: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
